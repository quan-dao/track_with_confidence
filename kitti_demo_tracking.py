'''
Demo tracking in one sequence of KITTI
'''

import cv2
import os
import sys
import numpy as np

from utils.front_end.kitti_parser import load_oxts_packets_and_poses
from utils.front_end.kitti_parser import parse_detection_file, kitti_obj_to_bbox3d, Calibration
from utils.front_end.kitti_parser import get_box_to_cam_trans
from utils.back_end.drawing_functions import draw_bbox3d
from tracking.tracklet_manager import TrackletManager
from tracking.measurement import cvt_bbox3d_to_measurement
from tracking.state import cvt_state_to_bbox3d
from global_config import GlobalConfig


assert GlobalConfig.dataset == 'kitti', \
    "Change GlobalConfig.dataset (from {}) to 'kitti'".format(GlobalConfig.dataset)
kitti_root = GlobalConfig.kitti_tracking_root


def main(seq_name, tracked_obj_class, create_video=True):
    print("Demo tracking {} on sequence {}. Press 'q' to stop".format(tracked_obj_class, seq_name))

    # get OXTS data
    oxts_file = os.path.join(kitti_root, 'data_tracking_oxts', 'training', 'oxts', '{}.txt'.format(seq_name))
    oxts = load_oxts_packets_and_poses([oxts_file])

    # get sequence image
    img_dir = os.path.join(kitti_root, 'data_tracking_image_2', 'training', 'image_02', seq_name)
    img_names = [file for file in os.listdir(img_dir) if file.endswith('png')]
    img_names.sort()

    if create_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        img = cv2.imread(os.path.join(img_dir, img_names[0]))
        video = cv2.VideoWriter('video.avi', fourcc, 1, (img.shape[1], img.shape[0]))

    # get sensor calib
    calib_file = os.path.join(kitti_root, 'data_tracking_calib', 'training', 'calib', '{}.txt'.format(seq_name))
    calib = Calibration(calib_file)

    # get detection
    detect_file = './data/kitti/pointrcnn_detection/pointrcnn_{}_val/{}.txt'.format(tracked_obj_class, seq_name)
    detection = parse_detection_file(detect_file)

    # init tracklet manager
    manager = TrackletManager(tracked_obj_class)

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

        # ------------------------------------------------------------------ #
        # tracking code
        # should go
        # here
        # convert boxes to measurements
        all_measurements = [cvt_bbox3d_to_measurement(box) for box in boxes_3d]
        manager.run_(all_measurements, frame_idx)
        # ------------------------------------------------------------------ #

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
        if create_video:
            video.write(img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if create_video:
        video.release()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: $python kitti_demo_tracking.py sequence_name tracked_obj_class")
        exit(0)
    seq_name = sys.argv[1]
    tracked_obj_class = sys.argv[2]
    assert len(seq_name) == 4 and int(seq_name[-2:]) in range(21), \
        'sequence_name must have the form 00xx, xx is integer from 00 to 20'
    assert tracked_obj_class in ('Car', 'Cyclist', 'Pedestrian')
    main(seq_name, tracked_obj_class)
