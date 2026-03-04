import cv2
import numpy as np
import os
from ultralytics import YOLO

class Detection:
    def __init__(self, box_2d, class_, mask=None):
        self.box_2d = box_2d
        self.detected_class = class_
        self.mask = mask

class cv_Yolo:                                                      
    def __init__(self, yolo_path, confidence=0.5, threshold=0.3):       
        self.confidence = float(confidence)                                 
        self.threshold = float(threshold)                                   
        engine_path = os.path.sep.join([yolo_path, "yolo26x-seg.engine"])       
        self.model = YOLO(engine_path, task='segment')

    def _clean_instance_mask(self, mask_bool, x1, y1, x2, y2, erode_iter=3):
        h, w = mask_bool.shape[:2]
        x1c = max(0, min(w - 1, int(x1)))
        y1c = max(0, min(h - 1, int(y1)))
        x2c = max(0, min(w, int(x2)))
        y2c = max(0, min(h, int(y2)))
        if x2c <= x1c or y2c <= y1c:
            return np.zeros((h, w), dtype=bool)

        cropped = np.zeros((h, w), dtype=np.uint8)
        cropped[y1c:y2c, x1c:x2c] = (mask_bool[y1c:y2c, x1c:x2c] > 0).astype(np.uint8) * 255

        if erode_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            cropped = cv2.erode(cropped, kernel, iterations=erode_iter)

        roi = cropped[y1c:y2c, x1c:x2c]
        num, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), connectivity=8)
        if num <= 1:
            return cropped > 0

        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        keep = np.zeros_like(roi, dtype=np.uint8)
        keep[labels == best] = 255
        cropped[y1c:y2c, x1c:x2c] = keep

        return cropped > 0

    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        TARGET_CLASSES = {"car", "truck", "pedestrian", "bus", "cyclist"}
        CLASS_MAP = {0: "pedestrian", 1: "cyclist", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        results = self.model.predict(image, conf=self.confidence, verbose=False)
        detections = []

        if len(results) == 0 or results[0].boxes is None:
            return detections

        result = results[0]
        boxes = result.boxes
        masks = result.masks

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            class_name = CLASS_MAP.get(cls_id, None)
            if class_name is None or class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            mask_bool = None
            if masks is not None and i < len(masks):
                mask_data = masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (orig_w, orig_h))
                mask_bool = mask_resized > 0.5
                mask_bool = self._clean_instance_mask(mask_bool, x1, y1, x2, y2, erode_iter=1)

            box_2d = [(x1, y1), (x2, y2)]
            detections.append(Detection(box_2d, class_name, mask_bool))

        return detections
