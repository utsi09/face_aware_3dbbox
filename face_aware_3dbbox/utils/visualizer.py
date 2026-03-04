from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

class BBoxVisualizer:
    def __init__(self):
        self.color_map = {
            'car': (0.0, 1.0, 0.0),
            'truck': (1.0, 1.0, 1.0),
            'pedestrian': (1.0, 0.0, 0.0),
            'cyclist': (0.0, 0.0, 1.0),
            'default': (1.0, 1.0, 0.0)
        }
        self.line_indices = [
            [7, 5], [5, 1], [1, 3], [3, 7],
            [6, 4], [4, 0], [0, 2], [2, 6],
            [7, 6], [5, 4], [1, 0], [3, 2]
        ]

    def _create_base_marker(self, id, stamp, ns, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = int(id)
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        marker.pose.orientation.w = 1.0
        return marker

    def _to_point(self, coord):
        return Point(x=float(coord[0]), y=float(coord[1]), z=float(coord[2]))

    def create_3d_marker(self, id, location, corners, dimensions, orientation, class_name, stamp, refine_pt, global_offset):
        main_color = self.color_map.get(class_name, self.color_map['default'])
        marker = self._create_base_marker(id, stamp, "bbox3d_main", main_color)
        for start_idx, end_idx in self.line_indices:
            marker.points.append(self._to_point(corners[start_idx]))
            marker.points.append(self._to_point(corners[end_idx]))

        offset_marker = self._create_base_marker(id + 10000, stamp, "bbox3d_offset", (1.0, 0.0, 0.0))
        offset_marker.points.append(self._to_point(refine_pt))
        offset_marker.points.append(self._to_point([
            refine_pt[0] - global_offset[0],
            refine_pt[1] - global_offset[1],
            refine_pt[2] - global_offset[2]
        ]))

        front_marker = self._create_base_marker(id + 1000, stamp, "bbox3d_front", (0.0, 0.0, 1.0))
        for s, e in [(3, 0), (2, 1)]:
            front_marker.points.append(self._to_point(corners[s]))
            front_marker.points.append(self._to_point(corners[e]))

        return [marker, front_marker, offset_marker]