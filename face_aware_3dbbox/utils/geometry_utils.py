import numpy as np
import math
from sklearn.cluster import DBSCAN

def closest_plane_center(pts, z_eps=0.15, xy_eps=0.5, min_samples=3):
        z_vals = pts[2, :].reshape(-1, 1)
        db_z = DBSCAN(eps=z_eps, min_samples=min_samples).fit(z_vals)
        labels_z = db_z.labels_
        unique_z = set(labels_z) - {-1}
        
        if len(unique_z) == 0:
            return np.median(pts, axis=1)
        
        best_label = min(unique_z, key=lambda l: np.median(z_vals[labels_z == l]))
        z_mask = labels_z == best_label
        pts_z_filtered = pts[:, z_mask]
        
        if pts_z_filtered.shape[1] < min_samples:
            return np.median(pts_z_filtered, axis=1)
        
        xy_vals = pts_z_filtered[:2, :].T
        db_xy = DBSCAN(eps=xy_eps, min_samples=min_samples).fit(xy_vals)
        labels_xy = db_xy.labels_
        unique_xy = set(labels_xy) - {-1}
        
        if len(unique_xy) == 0:
            return np.median(pts_z_filtered, axis=1)
        
        best_xy = max(unique_xy, key=lambda l: np.sum(labels_xy == l))
        final_mask = labels_xy == best_xy
        
        return np.median(pts_z_filtered[:, final_mask], axis=1)

def get_offset(face_num, faces, front_back_len, left_right_len, corner_len):
    if face_num == 1:
        for i in range(4):
            if faces[i] == 0:
                continue
            if i==0:
                return left_right_len[0]
            if i==1:
                return left_right_len[1]
            if i==2:
                return front_back_len[1]
            if i==3:
                return front_back_len[0]
    elif face_num == 2:
        if faces[3]==1 and faces[0]==1:
            return corner_len[0]
        if faces[3]==1 and faces[1]==1:
            return corner_len[1]
        if faces[0]==1 and faces[2]==1:
            return corner_len[3]
        if faces[1]==1 and faces[2]==1:
            return corner_len[2]
    else:
        return None

def get_face_len(corners, location):
    y = location[1]

    front_x = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4
    front_z = (corners[0][2] + corners[1][2] + corners[2][2] + corners[3][2]) / 4
    back_x = (corners[4][0] + corners[5][0] + corners[6][0] + corners[7][0]) / 4
    back_z = (corners[4][2] + corners[5][2] + corners[6][2] + corners[7][2]) / 4
    front_back = [[front_x, y, front_z], [back_x, y, back_z]]

    left_x = (corners[0][0] + corners[2][0] + corners[4][0] + corners[6][0]) / 4
    left_z = (corners[0][2] + corners[2][2] + corners[4][2] + corners[6][2]) / 4
    right_x = (corners[1][0] + corners[3][0] + corners[5][0] + corners[7][0]) / 4
    right_z = (corners[1][2] + corners[3][2] + corners[5][2] + corners[7][2]) / 4
    left_right = [[left_x, y, left_z], [right_x, y, right_z]]

    corner = [
        [corners[0][0], y, corners[0][2]],
        [corners[1][0], y, corners[1][2]],
        [corners[5][0], y, corners[5][2]],
        [corners[4][0], y, corners[4][2]],
    ]

    front_back_len = [cal_face_len(p, location) for p in front_back]
    left_right_len = [cal_face_len(p, location) for p in left_right]
    corner_len = [cal_face_len(p, location) for p in corner]

    return front_back_len, left_right_len, corner_len

def cal_face_len(point, location):
    point_offset = []
    for i in range(3):
        point_offset.append(point[i] - location[i])
    return point_offset

def count_face(inner_product):
    num = 0
    threshold = -0.3
    faces = [0,0,0,0]
    for i in range(4):
        if inner_product[i] < threshold:
            num+=1
            faces[i] = 1
    return num, faces 

def get_inner_product(global_orient, global_location_center,relative_loaction):
    front_nv = [math.cos(global_orient), math.sin(global_orient),0]
    back_nv = [-math.cos(global_orient), -math.sin(global_orient),0]
    left_nv = [-math.sin(global_orient), math.cos(global_orient),0]
    right_nv = [math.sin(global_orient), -math.cos(global_orient),0]
    nv_list = [front_nv, back_nv, left_nv, right_nv]
    inner_product = []
    view_vec_x = global_location_center[0] - relative_loaction[0]
    view_vec_y = global_location_center[1] - relative_loaction[1]
    mag = math.sqrt(view_vec_x**2 + view_vec_y**2)
    if mag > 0:
        view_vec_x /= mag
        view_vec_y /= mag
    for nv in nv_list:
        dot_val = (view_vec_x * nv[0]) + (view_vec_y * nv[1])
        inner_product.append(dot_val)
    return inner_product
        
def get_visible_faces_cam(corners, location):
    corners_np = np.array(corners)
    loc = np.array(location)
    side_faces = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ]
    face_names = ['FRONT', 'BACK', 'LEFT', 'RIGHT']

    visible = []
    for fi, corner_ids in enumerate(side_faces):
        c = corners_np[corner_ids]
        face_center = np.mean(c, axis=0)
        v1 = c[1] - c[0]
        v2 = c[3] - c[0]
        normal = np.cross(v1, v2)
        if np.dot(normal, face_center - loc) < 0:
            normal = -normal
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
        view_dir = -face_center
        dot_val = np.dot(normal, view_dir)
        if dot_val > 0:
            visible.append((corner_ids, face_center, face_names[fi], dot_val))

    face_num = len(visible)
    if face_num == 2:
        dot0 = visible[0][3]
        dot1 = visible[1][3]
        minor_dot = min(dot0, dot1)
        if minor_dot < 0.15:
            if dot0 >= dot1:
                visible = [visible[0]]
            else:
                visible = [visible[1]]
            face_num = 1

    if face_num == 1:
        fc = visible[0][1]
        offset = [float(fc[0] - loc[0]), 0.0, float(fc[2] - loc[2])]
        return face_num, offset
    elif face_num == 2:
        dot0 = visible[0][3]
        dot1 = visible[1][3]
        if dot0 >= dot1:
            dominant_fc, dominant_name = visible[0][1], visible[0][2]
        else:
            dominant_fc, dominant_name = visible[1][1], visible[1][2]
        offset = [float(dominant_fc[0] - loc[0]), 0.0, float(dominant_fc[2] - loc[2])]
        return face_num, offset

    return face_num, None

def rotate_offset_to_global(offset, ego_yaw):
    local_x = float(offset[2])
    local_y = float(-offset[0])
    local_z = float(-offset[1])
    gx = local_x * math.cos(ego_yaw) - local_y * math.sin(ego_yaw)
    gy = local_x * math.sin(ego_yaw) + local_y * math.cos(ego_yaw)
    gz = local_z
    return [gx, gy, gz]

def trans_global(location,ego_x, ego_y, ego_z,ego_yaw):
    CAMERA_X_OFFSET = 2.0
    CAMERA_Y_OFFSET = 0.0 
    CAMERA_Z_OFFSET = 2.0 
    local_x = float(location[2]) + CAMERA_X_OFFSET
    local_y = float(-location[0])+ CAMERA_Y_OFFSET
    gl_x = ego_x + (local_x * math.cos(ego_yaw) - local_y * math.sin(ego_yaw))
    gl_y = ego_y + (local_x * math.sin(ego_yaw) + local_y * math.cos(ego_yaw))
    gl_z = ego_z + float(-location[1]) + CAMERA_Z_OFFSET
    return [gl_x, gl_y, gl_z]