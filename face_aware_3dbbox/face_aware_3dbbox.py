#!/usr/bin/env python3
import rclpy
import numpy as np
import cv2
import torch
import os
import sys
import math
bbox_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, bbox_path)
from rclpy.time import Time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup 
from rclpy.executors import MultiThreadedExecutor     
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from dataclasses import dataclass
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo
from inference import Inference
from utils.visualizer import BBoxVisualizer
from face_aware_3dbbox.utils.geometry_utils import trans_global, count_face, closest_plane_center, get_offset,get_face_len,cal_face_len,get_inner_product,get_visible_faces_cam ,rotate_offset_to_global

@dataclass
class Ego():
    x : float = None
    y : float = None
    z : float = None
    or_w : float = None
    or_x : float = None
    or_y : float = None
    or_z : float = None

class FaceAwareBBox3D(Node):
    def __init__(self):
        super().__init__('bbox3d_node_upgrade')

        self.declare_parameter('lidar_topic', '/carla/hero/lidar')
        self.declare_parameter('camera_topic', '/carla/hero/rgb_front/image')
        self.declare_parameter('camera_info_topic', '/carla/hero/rgb_front/camera_info')
        self.declare_parameter('odometry', '/carla/hero/odometry')

        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        odometry = self.get_parameter('odometry').value

        self.bridge = CvBridge()
        self.proj_matrix = None
        self.ego = Ego()
        self.engine = Inference()
        self.marker = BBoxVisualizer()
        self.odom_cb_group = MutuallyExclusiveCallbackGroup()

        self.tf_buffer = Buffer()
        self.dynamic_tf_broadcaster = TransformBroadcaster(self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        lidar_sub  = Subscriber(self, PointCloud2, self.lidar_topic)
        front_sub  = Subscriber(self, Image, self.camera_topic)
        front_info_sub  = Subscriber(self, CameraInfo, self.camera_info_topic)

        self.ts = ApproximateTimeSynchronizer (
            [lidar_sub, front_sub, front_info_sub],
            queue_size= 10,
            slop = 0.05
        )
        self.ts.registerCallback(self.sync_callback)

        self.odom_sub = self.create_subscription(
            Odometry, odometry, self.odometry_callback, 20, 
            callback_group=self.odom_cb_group)

        self.prev_marker_ids = set()
        self.marker_pub = self.create_publisher(MarkerArray, '/bbox3d/markers', 20)
        self.image_pub = self.create_publisher(Image, '/bbox3d/image', 20)
        self.pts_pub = self.create_publisher(PointCloud2, '/bbox3d/lidar',20)
        self.pts_pub_refine = self.create_publisher(PointCloud2, '/bbox3d/lidar/refine',20)

    def quat_to_rot(self, qx, qy, qz, qw):
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        return np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
        ], dtype=np.float32)

    def tf_to_T(self, tf_msg):
        tr = tf_msg.transform.translation
        qr = tf_msg.transform.rotation
        R = self.quat_to_rot(qr.x, qr.y, qr.z, qr.w)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        return T

    def odometry_callback(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id ="map"
        t.child_frame_id = "hero"

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation
        self.dynamic_tf_broadcaster.sendTransform(t)

    def get_ego_pose_at_time(self,timestamp):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'hero',
                Time.from_msg(timestamp),
                rclpy.duration.Duration(seconds=0.01)
            )
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            tz = t.transform.translation.z
            
            rx = t.transform.rotation.x
            ry = t.transform.rotation.y
            rz = t.transform.rotation.z
            rw = t.transform.rotation.w
            
            siny_cosp = 2 * (rw * rz + rx * ry)
            cosy_cosp = 1 - 2 * (ry * ry + rz * rz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return tx, ty, tz, yaw
        
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def sync_callback(self, lidar_msg, msg: Image, front_info_msg: CameraInfo):
        try:
            truth_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        img_h, img_w = truth_img.shape[:2]

        if self.proj_matrix is None:
            P = np.array(front_info_msg.p).reshape(3, 4)
            self.proj_matrix = P

        if self.proj_matrix is None:
            self.get_logger().warn('Waiting for camera info', throttle_duration_sec=2.0)
            return

        ego_pose = self.get_ego_pose_at_time(lidar_msg.header.stamp)
        if ego_pose is None:
            self.get_logger().warn("TF data 기다리는중", throttle_duration_sec=1.0)
            return
        
        ego_x, ego_y, ego_z, ego_yaw = ego_pose

        marker_array = MarkerArray()
        refined_points_to_pub = []
        pts_np = pc2.read_points_numpy(lidar_msg, field_names=("x", "y", "z"))
        pts_lidar = np.hstack([pts_np, np.ones((pts_np.shape[0], 1))]).T

        K = np.array(front_info_msg.k, dtype=np.float32).reshape(3, 3)

        lidar_frame = lidar_msg.header.frame_id
        cam_frame = msg.header.frame_id

        if front_info_msg.header.frame_id:
            cam_frame = front_info_msg.header.frame_id

        try:
            tf_lidar2cam = self.tf_buffer.lookup_transform(
                cam_frame,
                lidar_frame,
                Time.from_msg(lidar_msg.header.stamp),
                rclpy.duration.Duration(seconds=0.05)
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {lidar_frame} -> {cam_frame}: {e}",
                throttle_duration_sec=1.0
            )
            return

        T_lidar2cam = self.tf_to_T(tf_lidar2cam)
        pts_cam = T_lidar2cam @ pts_lidar 

        proj = K @ pts_cam[:3, :]
        z = proj[2, :]
        u_f = proj[0, :] / z
        v_f = proj[1, :] / z

        in_img = (z > 0) & (u_f >= 0) & (u_f < img_w) & (v_f >= 0) & (v_f < img_h)
        if not np.any(in_img):
            return
        pts_cam = pts_cam[:, in_img]
        u = u_f[in_img].astype(np.int32)
        v = v_f[in_img].astype(np.int32)

        uvs = (u,v)
        total_lidar_points = []
        TARGET_CLASSES = ['car', 'truck', 'bus', 'person', 'pedestrian', 'cyclist']

        inference_results = self.engine.predict(truth_img, self.proj_matrix)
        
        for res in inference_results:
            idx = res.idx
            detected_class = res.cls
            box_2d = res.box_2d
            mask_source = res.mask
            dim = res.dim
            location = res.location
            orient = res.orient
            alpha = res.alpha
            theta_ray = res.theta_ray

            global_corners_refine = None
            my_local_center = None

            global_location_center = trans_global(location, ego_x, ego_y, ego_z, ego_yaw)

            R = rotation_matrix(orient)

            corners = create_corners(dim, location=location, R=R)

            global_corners = []
            for corner in corners:
                g_pt = trans_global(corner, ego_x, ego_y, ego_z, ego_yaw)
                global_corners.append(g_pt)
            
            global_orient = ego_yaw - orient
            relative_location = [ego_x, ego_y, ego_z]

            inner_product = get_inner_product(global_orient, global_location_center, relative_location)
            face_num, faces = count_face(inner_product)
            self.get_logger().info(f'{face_num}개의 면이 보임, faces={faces}')

            front_back_len, left_right_len, corner_len = get_face_len(corners, location)
            raw_offset = get_offset(face_num, faces, front_back_len, left_right_len, corner_len)

            mask_uint8 = (mask_source.astype(np.uint8)) * 255
            mask_r = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_r > 127

            in_mask = mask_bool[v, u]
            if np.any(in_mask):
                pts_obj_cam = pts_cam[:, in_mask] 
                center_cam = closest_plane_center(pts_obj_cam[:3, :])

                x_cam, y_cam, z_cam = map(float, center_cam)
                self.get_logger().info(f"[CENTER_CAM] x={x_cam:.2f}, y={y_cam:.2f}, z={z_cam:.2f}")
                dists_center=[x_cam, y_cam, z_cam]
                pts3 = pts_obj_cam[:3, :]
                uclidian = np.linalg.norm(pts3, axis=0)

                sorted_idx = np.argsort(uclidian)
                N = uclidian.shape[0]
                top_k = min(N, max(int(N * 0.15), 1))
                nearest_pts = pts3[:, sorted_idx[:top_k]]

                min_cam = np.mean(nearest_pts, axis=1)
                x_cam, y_cam, z_cam = map(float, min_cam)
                self.get_logger().info(f"[MIN_CAM_5%] x={x_cam:.2f}, y={y_cam:.2f}, z={z_cam:.2f}, k={top_k}")
                dists_min = [x_cam, y_cam, z_cam]

                target_pt = None
                if face_num == 1:
                    target_pt = dists_center
                elif face_num == 2:
                    target_pt = dists_min
                else:
                    continue

                if raw_offset is not None:
                    my_local_center = [
                        target_pt[0] - raw_offset[0],
                        target_pt[1] - raw_offset[1],
                        target_pt[2] - raw_offset[2]
                    ]

                if target_pt is not None:
                    refined_points_to_pub.append([float(target_pt[0]), float(target_pt[1]), float(target_pt[2])])

                N = pts_obj_cam.shape[1]
                pts_obj_h = np.vstack((pts_obj_cam[:3,:], np.ones((1,N))))

                T_cam2lidar = np.linalg.inv(T_lidar2cam)
                pts_obj_lidar = (T_cam2lidar @ pts_obj_h)[:3, :].T

                pc_header = lidar_msg.header
                pc_header.frame_id = lidar_msg.header.frame_id
                N = pts_obj_cam.shape[1]
                pts_obj_h = np.vstack((pts_obj_cam[:3,:], np.ones((1,N))))

                if my_local_center is None:
                    continue

                self.get_logger().info(f"{location} \ {my_local_center}")
                corners= plot_3d_box(truth_img, self.proj_matrix, orient, dim, my_local_center)

                global_corners_refine = []
                for corner in corners:
                    g_pt = trans_global(corner, ego_x, ego_y, ego_z, ego_yaw)
                    global_corners_refine.append(g_pt)

                total_lidar_points.extend(pts_obj_lidar.tolist())
                
                global_offset = rotate_offset_to_global(raw_offset, ego_yaw)
                refine_point = [float(target_pt[0]), float(target_pt[1]), float(target_pt[2])]
                refine_point = trans_global(refine_point, ego_x, ego_y, ego_z, ego_yaw)

            if global_corners_refine is not None:
                for i in range(3):
                    self.get_logger().info(f'refine_point{i} : {refine_point[i]}, global_offset{i} : {global_offset[i]}')
                markers_list = self.marker.create_3d_marker(idx, location, global_corners_refine, dim, global_orient, detected_class, lidar_msg.header.stamp,refine_point, global_offset)
                marker_array.markers.extend(markers_list)

        if len(total_lidar_points) > 0:
            try:
                tf_lidar2map = self.tf_buffer.lookup_transform(
                    "map",
                    lidar_frame,
                    Time.from_msg(lidar_msg.header.stamp),
                    rclpy.duration.Duration(seconds=0.05)
                )
                T_lidar2map = self.tf_to_T(tf_lidar2map)
                pts = np.asarray(total_lidar_points, dtype=np.float32)
                pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])  
                pts_map = (T_lidar2map @ pts_h.T).T[:, :3].tolist()

                pc_header = lidar_msg.header
                pc_header.frame_id = "map"
                pc_msg = pc2.create_cloud_xyz32(pc_header, pts_map)
                self.pts_pub.publish(pc_msg)
            except Exception as e:
                self.get_logger().warn(f"TF lidar->map lookup failed: {e}", throttle_duration_sec=1.0)

        if len(refined_points_to_pub) > 0:
            try:
                tf_cam2map = self.tf_buffer.lookup_transform(
                    "map",
                    cam_frame,  
                    Time.from_msg(lidar_msg.header.stamp),
                    rclpy.duration.Duration(seconds=0.05)
                )
                T_cam2map = self.tf_to_T(tf_cam2map)

                pts = np.asarray(refined_points_to_pub, dtype=np.float32)
                pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
                pts_map = (T_cam2map @ pts_h.T).T[:, :3].tolist()

                pc_header = lidar_msg.header
                pc_header.frame_id = "map"
                refined_msg = pc2.create_cloud_xyz32(pc_header, pts_map)
                self.pts_pub_refine.publish(refined_msg)
            except Exception as e:
                self.get_logger().warn(f"TF cam->map lookup failed: {e}", throttle_duration_sec=1.0)

        cur_marker_ids = set()
        for m in marker_array.markers:
            cur_marker_ids.add((m.ns, m.id))

        stale = self.prev_marker_ids - cur_marker_ids
        for ns, mid in stale:
            del_marker = Marker()
            del_marker.header.frame_id = "map"
            del_marker.header.stamp = lidar_msg.header.stamp
            del_marker.ns = ns
            del_marker.id = mid
            del_marker.action = Marker.DELETE
            marker_array.markers.append(del_marker)

        self.prev_marker_ids = cur_marker_ids
        self.marker_pub.publish(marker_array)

        vis_msg = self.bridge.cv2_to_imgmsg(truth_img, 'bgr8')
        vis_msg.header = msg.header
        self.image_pub.publish(vis_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FaceAwareBBox3D()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()