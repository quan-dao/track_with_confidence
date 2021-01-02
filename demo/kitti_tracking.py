'''
Demo tracking in one sequence of KITTI
'''

import cv2
import os
import numpy as np

from utils.front_end.kitti_parser import load_oxts_packets_and_poses
from utils.front_end.kitti_parser import parse_detection_file, kitti_obj_to_bbox3d, Calibration
from utils.front_end.kitti_parser import get_box_to_cam_trans
from utils.back_end.drawing_functions import draw_bbox3d
from tracklet_manager import TrackletManager
from measurement import cvt_bbox3d_to_measurement
from state import cvt_state_to_bbox3d


seq_name = '0000'

# get OXTS data
oxts_file = '/home/user/Downloads/kitti/tracking/data_tracking_oxts/training/oxts/{}.txt'.format(seq_name)
oxts = load_oxts_packets_and_poses([oxts_file])
print('num poses: ', len(oxts))
print('first pose: \n', oxts[0].imu_to_world)

# get sequence image
img_dir = '/home/user/Downloads/kitti/tracking/data_tracking_image_2/training/image_02/{}'.format(seq_name)
img_names = [file for file in os.listdir(img_dir) if file.endswith('png')]
img_names.sort()

# get sensor calib
calib_file = '/home/user/Downloads/kitti/tracking/data_tracking_calib/training/calib/{}.txt'.format(seq_name)
calib = Calibration(calib_file)

# get detection
detect_file = '/home/user/Desktop/python_ws/AB3DMOT/data/KITTI/pointrcnn_Car_val/{}.txt'.format(seq_name)
detection = parse_detection_file(detect_file)

# init tracklet manager
manager = TrackletManager('car')

# main loop
world_to_c0 = np.eye(4)
for frame_idx, name in enumerate(img_names):
    img = cv2.imread(os.path.join(img_dir, name))
    # get ego vehicle pose
    imu_to_world = oxts[frame_idx].imu_to_world

    if frame_idx == 0:
        # get mapping from world to c0
        c0_to_world = imu_to_world @ calib.cam_to_imu
        world_to_c0 = np.linalg.inv(c0_to_world)

    ci_to_world = imu_to_world @ calib.cam_to_imu
    ci_to_c0 = world_to_c0 @ ci_to_world

    # get 3D object detection in camera frame
    boxes_3d = []
    if frame_idx in detection.keys():
        boxes_3d = [kitti_obj_to_bbox3d(o, timestamp=frame_idx, obj_type='Car') for o in detection[frame_idx]]

    # map boxes from current camera frame to first camera frame
    for box in boxes_3d:
        box.transform_(ci_to_c0, 'c0')

    # ------------------------------------------------------------------#
    # tracking code
    # should go
    # here
    # convert boxes to measurements
    all_measurements = [cvt_bbox3d_to_measurement(box) for box in boxes_3d]
    manager.run_(all_measurements, frame_idx)

    #------------------------------------------------------------------#
    c0_to_cam = np.linalg.inv(ci_to_c0)
    for tracklet in manager.all_tracklets:
        if tracklet.tail.stamp == frame_idx and not tracklet.just_born:
            # convert tracklet's tail to a Bbox3D
            box = cvt_state_to_bbox3d(tracklet.tail, tracklet.id, tracklet.most_recent_meas_score)
            # map box back to camera frame for drawing
            box.transform_(c0_to_cam, 'camera')
            # draw boxes
            box_to_cam = get_box_to_cam_trans(box)
            proj = box.project_on_image(box_to_cam, calib.cam_proj_mat)
            draw_bbox3d(img, proj, tracklet.id)

    cv2.imshow('left_color', img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

