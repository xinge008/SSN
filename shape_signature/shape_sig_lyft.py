# -*- coding:utf-8 -*-
# author: Xinge
# @file: shape_sig_lyft.py 






####################### compute three views for the shape embedding ####################

########### get three views of points in box ############
import numpy.polynomial.chebyshev as chebyshev
from scipy.spatial import ConvexHull
import numpy as np
from nuscenes.eval.detection.utils import quaternion_yaw
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

nusc = NuScenes(version='v1.01-train', dataroot='/path/to/lyft/v1.01-train/', verbose=True)

def get_points(sample_id):
    sample = nusc.sample[sample_id]
    lidar_path, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])


    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)

    if np.remainder(len(points), 5) != 0:
        cnt = len(points)-np.remainder(len(points), 5)
        print("Abnormal data:")
        print(str(lidar_path))
        points = points[:cnt].reshape([-1, 5])
    else:
        points = points.reshape([-1, 5])

    points[:, 3] /= 255
    points[:, 4] = 0

    return points, boxes
## 1. get the points in the box

number_of_less = 0

def get_points_after_view_transform(points, box_t):
#     box_t = boxes[box_id]
    wlh = box_t.wlh
#     print(wlh)
    box_len = np.sqrt(wlh[0]**2 + wlh[1]**2)
#     print(box_t)
    points_new = points[:, 0:3].transpose()
    mask = points_in_box(box_t, points_new, wlh_factor=1.0)
#     print(mask.sum())
    if mask.sum() <= 1:
#         print("points in box are less than 2!!!")
        global number_of_less
        number_of_less += 1
        return None
#     print("points number is :", mask.sum())
    points_in = points_new.transpose()[mask]
    # print(points_in)
    ### normalize the 3D points
    points_in = points_in - box_t.center
#     points_in = np.concatenate((points_in, np.array([[0.0,0.0,0.0]]),np.array([[0.01,0.0,0.0]])), 0)
    points_in = points_in.transpose()

## 2. get normal vector of forwarding direction
    box_coors = box_t.corners().transpose()
    pq = box_coors[3] - box_coors[0]
    pr = box_coors[4] - box_coors[0]
    normal_1 = np.cross(pq, pr)

## 3. get the rotation angle between normal vector and base vector
    def unit_vector(vector):
        """ Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    base_vector = np.array([0,1,0])
    angle_t = angle_between(normal_1, base_vector)

    ### get the axis of rotation
    axis_t = np.cross(normal_1, base_vector)

    ## 4. get the matrix representation of view; angle --> Quaternion --> matrix representation

    quat = Quaternion._from_axis_angle(axis_t, angle_t)
#     print(quat)
    FLOAT_EPS = np.finfo(np.float).eps
    def _quat_to_matrix(quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        Nq = w*w + x*x + y*y + z*z
        if Nq < FLOAT_EPS:
            return np.eye(3)
        s = 2.0/Nq
        X = x*s
        Y = y*s
        Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
        return np.array(
               [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

    view_t = _quat_to_matrix(quat)

    ## 5. view transformation
    points_t = view_points(points_in, view_t, normalize=False)
#     print(points_t.shape)
    return points_t

################### compute the symmetry  and covtex hull #################

# points_t2 = points_t.transpose() * [-1,-1,-1]
# print(points_t.transpose())
# print(points_t2)
# points_t = np.concatenate((points_t.transpose(), points_t2), 0).transpose()
# print(points_t)
# bird_view = points_t[[0, 1], :].transpose() + points_t[[0, 1], :].transpose() * [-1.0, -1.0]
# front_view = points_t[[1, 2], :].transpose() + points_t[[1, 2], :].transpose() * [-1.0, 1.0]
# profile_view = points_t[[0, 2], :].transpose() + points_t[[0, 2], :].transpose() * [1.0, -1.0]

def get_hull_from_three_views(points_t):
    points_trans = points_t.transpose()
#     print(points_trans.shape)
    bird_view = np.concatenate((points_trans[:, [0, 1]], points_trans[:, [0, 1]]*[-1, -1]), 0)
#     print(bird_view.shape)
    front_view = np.concatenate((points_trans[:, [1, 2]], points_trans[:, [1, 2]]*[-1, -1]), 0)
    profile_view = np.concatenate((points_trans[:, [0, 2]], points_trans[:, [0, 2]]*[-1, -1]), 0)

#     print(bird_view.shape)

    hull_bird = ConvexHull(bird_view)

#     for simplex in hull_bird.simplices:
#         plt.plot(bird_view[simplex, 0], bird_view[simplex, 1], 'k-')

    hull_forw = ConvexHull(front_view)

#     for simplex in hull_forw.simplices:
#         plt.plot(front_view[simplex, 0], front_view[simplex, 1], 'k-')

    hull_prof = ConvexHull(profile_view)


#     for simplex in hull_prof.simplices:
#         plt.plot(profile_view[simplex, 0], profile_view[simplex, 1], 'k-')

    return hull_bird, hull_forw, hull_prof

################### compute the distance between line and polygan  #################

def hit(U,hull):
    eq=hull.equations.T
#     print("eq.shape: ", eq.shape)
    V,b=eq[:-1],eq[-1]
#     print(U.shape)
#     print("v:shape: ", V.shape)
    alpha=-b/np.dot(V.transpose(),U)
#     print(alpha)
#     print(alpha.shape)
#     if(len(alpha))
    return np.min(alpha[alpha>0])*U

# dist_p_hull = hit(np.array([-1,0]), hull)
# print(dist_p_hull)


################## compute the rotated distance ############

def compute_dist(degs, hull):
    dist_p_hull_list = []
#     print(len(degs))
    for deg in degs:
#         print(deg)
        if deg < 90:
            y_ = np.tan(np.radians(deg))
            dist_p_hull = hit(np.array([1.0, y_]), hull)
        elif deg == 90:
            dist_p_hull = hit(np.array([0, 1.0]), hull)
        elif deg >90 and deg < 180:
            y_ = -1.0 * np.tan(np.radians(deg))
            dist_p_hull = hit(np.array([-1, y_]), hull)
        elif deg == 180:
            dist_p_hull = hit(np.array([-1.0, 0.0]), hull)
        elif deg > 180 and deg < 270:
            y_ = -1.0 * np.tan(np.radians(deg))
            dist_p_hull = hit(np.array([-1, y_]), hull)
        elif deg == 270:
            dist_p_hull = hit(np.array([0.0, -1.0]), hull)
        elif deg > 270 and deg < 360:
            y_ = np.tan(np.radians(deg))
            dist_p_hull = hit(np.array([1.0, y_]), hull)
        elif deg == 0 or deg == 360:
            dist_p_hull = hit(np.array([1.0, 0.0]), hull)
        else:
            print("deg error !!!")
        dist_p_hull_list.append([dist_p_hull[0], dist_p_hull[1]])

    return dist_p_hull_list

def get_coff(hull, degs):


    dist_degs = np.array(compute_dist(degs, hull))

    dist_len = np.sqrt(dist_degs[:,0]**2 + dist_degs[:,1]**2)
    # dist_len[dist_len>3] = 5.3
    # dist_len = dist_len / box_len
#     print(dist_len)


    coefficient,Res = chebyshev.chebfit(degs,dist_len,8,full=True)
    return coefficient

# degs = np.arange(360) * (360/360)

# coff_bird = get_coff(hull, degs)
# coff_forw = get_coff(hull_forw, degs)
# coff_prof = get_coff(hull_prof, degs)

# print("coefficient===========")
# print(coff_bird)
# print(coff_forw)
# print(coff_prof)
# print(Res)

########### render the points and boxes in the image ###############
# NameMapping = {
#         'movable_object.barrier': 'barrier',
#         'vehicle.bicycle': 'bicycle',
#         'vehicle.bus.bendy': 'bus',
#         'vehicle.bus.rigid': 'bus',
#         'vehicle.car': 'car',
#         'vehicle.construction': 'construction_vehicle',
#         'vehicle.motorcycle': 'motorcycle',
#         'human.pedestrian.adult': 'pedestrian',
#         'human.pedestrian.child': 'pedestrian',
#         'human.pedestrian.construction_worker': 'pedestrian',
#         'human.pedestrian.police_officer': 'pedestrian',
#         'movable_object.trafficcone': 'traffic_cone',
#         'vehicle.trailer': 'trailer',
#         'vehicle.truck': 'truck'
#     }
NameMapping = {
        'bicycle': 'bicycle',
        'bus': 'bus',
        'car': 'car',
        'emergency_vehicle': 'emergency_vehicle',
        'motorcycle': 'motorcycle',
        'other_vehicle': 'other_vehicle',
        'pedestrian': 'pedestrian',
        'truck': 'truck',
        'animal': 'animal'
    }
# sample_id = 3
# box_id = 2
# import pickle

shape_pattern = {'barrier': [], 'bicycle': [], 'bus': [], 'car': [], 'motorcycle': [], 'pedestrian': [],
                  'truck': [], 'other_vehicle': [], 'emergency_vehicle': [], 'animal': []}

def get_coff_three_views(sample_id, shape_emb):

    global shape_pattern

    points, boxes = get_points(sample_id)
#     coff_list = []
    print("len of boxes: ", len(boxes))
    for box_id in range(len(boxes)):
#         print(box_id)
        box_t = boxes[box_id]
#         print(box_t.name)
#         print(box_t)
        token_t = box_t.token
        if box_t.name not in NameMapping.keys():
            continue
        name_t = NameMapping[box_t.name]
        xyz = box_t.center
#         print(xyz)
        dist_t = np.sqrt(xyz[0]**2 + xyz[1]**2)
        point_trans = get_points_after_view_transform(points, box_t)
#         print("points in box:")

        item_t = {'name': name_t, 'dist': dist_t}

        if token_t in shape_emb.keys():
            print("Not Unique Token: ", token_t)

        if point_trans is None:
#             coff_list.append(np.zeros(9))
            item_t['shape'] = None
            shape_emb[token_t] = item_t
            continue
#         print(len(point_trans))
        bird_hull, front_hull, prof_hull = get_hull_from_three_views(point_trans)
        degs = np.arange(360) * (360/360)
        coff_bird = get_coff(bird_hull, degs)
        coff_forw = get_coff(front_hull, degs)
        coff_prof = get_coff(prof_hull, degs)

        coff = np.concatenate((coff_bird[0:3], coff_forw[0:3], coff_prof[0:3]), 0)
#         print(coff)
#         coff_list.append(coff)
        item_t['shape'] = coff
        shape_pattern[name_t].append(list(coff))
        shape_emb[token_t] = item_t
#     print(len(boxes))
#     print()
#     coff_list = np.array(coff_list)
#     print(coff_list)
#     return coff_list

# coff_list = get_coff_three_views(sample_id)


import pickle
def get_all_shape_embedding():
    shape_emb = {}
    for sample_id in range(len(nusc.sample)):
        get_coff_three_views(sample_id, shape_emb)
    print("total boxes: ", len(shape_emb.keys()))

    pickle.dump(shape_emb, open("./box_shape.pkl", "wb"))

get_all_shape_embedding()
print("less of 2 points: ", number_of_less)



for keys_ in shape_pattern.keys():
    shape_com = shape_pattern[keys_]
    if len(shape_com) > 0:
        print("Keys: ", keys_)
        print("Common Values: ", np.array(shape_com).sum(0)/len(shape_com))