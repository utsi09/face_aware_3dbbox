#!/usr/bin/env python3
import rclpy
import numpy as np
import cv2
import torch
import os
import sys
import math
from cv_bridge import CvBridge
from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo
from torchvision.models import vgg
from dataclasses import dataclass

@dataclass
class PredictionResult:
    idx: int
    cls: str
    box_2d: np.ndarray
    mask: np.ndarray
    dim: np.ndarray
    location: np.ndarray
    orient: float
    alpha: float
    theta_ray: float


class Inference():
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

        model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
        if len(model_lst) == 0:
            self.get_logger().error('No previous model')
            raise FileNotFoundError('No .pkl model in weights/')

        my_vgg = vgg.vgg19_bn(pretrained=True)
        self.model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s' % model_lst[-1])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.yolo = cv_Yolo(weights_path)
        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(2)

    def predict(self,truth_img, proj_matrix):
        img = np.copy(truth_img)
        detections = self.yolo.detect(img)
        results = []
        for idx, detection in enumerate(detections):
            if not self.averages.recognized_class(detection.detected_class):
                continue
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, proj_matrix)
            except:
                continue

            
            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img

            with torch.no_grad():
                [orient, conf, dim] = self.model(input_tensor)
                orient = orient.cpu().data.numpy()[0, :, :]
                conf = conf.cpu().data.numpy()[0, :]
                dim = dim.cpu().data.numpy()[0, :]

            dim += self.averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += self.angle_bins[argmax]
            alpha -= np.pi

            location, X = calc_location(dim, proj_matrix, box_2d, alpha, theta_ray)
            orient = alpha + theta_ray

            res = PredictionResult(
                idx=idx,
                cls=detection.detected_class,
                box_2d=detection.box_2d,
                mask=detection.mask,
                dim=dim,
                location=location,
                orient=orient,
                alpha=alpha,
                theta_ray=theta_ray
            )
            results.append(res)

        return results
