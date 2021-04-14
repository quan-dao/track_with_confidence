import os
import glob
import sys
import numpy as np
from tqdm import tqdm

from utils.front_end.kitti_parser import load_oxts_packets_and_poses
from utils.front_end.kitti_parser import parse_detection_file, kitti_obj_to_bbox3d, Calibration
from utils.front_end.kitti_parser import get_box_to_cam_trans
from utils.data_classes import Bbox3D
from tracking.tracklet_manager import TrackletManager
from tracking.measurement import cvt_bbox3d_to_measurement
from tracking.state import cvt_state_to_bbox3d
from global_config import GlobalConfig


def format_result(frame_idx, track_id, obj_type, box_3d, projection_2d):
    """Convert a Bbox3D into a line in tracking result file

    Args:
        frame_idx (int): frame index
        track_id (int): tracklet's id
        obj_type (str): object type
        box_3d (Bbox3D): a 3D bounding box
        projection_2d (list): axis-aligned bounding box of projection of 3D bbox onto image, [x1, y1, x2, y2]
    Returns:
        str: a line including new line symbolic to write into tracking result file
    """
    truncated = 0
    occluded = 0
    alpha = box.cam_obs_angle
    bbox = '{} {} {} {} '.format(*projection_2d)
    dimensions = '{} {} {} '.format(box_3d.h, box_3d.w, box_3d.l)
    location = '{} {} {} '.format(*box_3d.center.tolist())
    rotation_y = box_3d.yaw
    score =box_3d.score
    return '{} {} {} '.format(frame_idx, track_id, obj_type) + '{} {} '.format(truncated, occluded) + '{} '.format(alpha) \
           + bbox + dimensions + location + '{} '.format(rotation_y) + '{}\n'.format(score)


assert GlobalConfig.dataset == 'kitti', \
    "Change GlobalConfig.dataset (from {}) to 'kitti'".format(GlobalConfig.dataset)
kitti_root = GlobalConfig.kitti_tracking_root

# prepare data directories
oxts_root = os.path.join(kitti_root, 'data_tracking_oxts', 'training', 'oxts')
calib_root = calib_file = os.path.join(kitti_root, 'data_tracking_calib', 'training', 'calib')
detection_root = './data/kitti/pointrcnn_detection'
result_root = './results/kitti'

# get all sequence names
seq_names = [name[:-4] for name in os.listdir(oxts_root) if name.endswith('.txt')]
seq_names.sort()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: $python generate_kitti_re.py t_sha obj_type data_split\n"
              "\t t_sha is an arbitrary string to format the result name")
        exit(0)
    # python generate_kitti_re.py t_sha obj_type data_split
    t_sha = sys.argv[1]
    obj_type = sys.argv[2]
    data_split = sys.argv[3]
    assert obj_type in ['Car', 'Pedestrian', 'Cyclist']
    assert data_split in ['val', 'test']
    result_dir = os.path.join(result_root, t_sha, 'data')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        # clear result in this folder
        files = glob.glob(os.path.join(result_dir, '*.txt'))
        for f in files:
            os.remove(f)

    for seq_idx, seq in enumerate(seq_names):
        print('\n--------------------------------------------')
        print('Generate tracking result for sequence {}/{}'.format(seq_idx, len(seq_names)))
        # prepare result file
        result_file = open(os.path.join(result_dir, '{}.txt'.format(seq)), 'w')
        # parse data file
        oxts = load_oxts_packets_and_poses([os.path.join(oxts_root, '{}.txt'.format(seq))])
        calib = Calibration(os.path.join(calib_root, '{}.txt'.format(seq)))
        detection = parse_detection_file(os.path.join(detection_root, 'pointrcnn_{}_{}/{}.txt'.format(
            obj_type, data_split, seq)))
        # init
        manager = TrackletManager(obj_type)
        world_to_c0 = np.eye(4)
        for frame_idx in tqdm(range(len(oxts))):
            # get ego vehicle pose
            imu_to_world = oxts[frame_idx].imu_to_world
            # get mapping from world to c0
            if frame_idx == 0:
                c0_to_world = imu_to_world @ calib.cam_to_imu
                world_to_c0 = np.linalg.inv(c0_to_world)
            # compute pose of i-th camera w.r.t first camera
            ci_to_world = imu_to_world @ calib.cam_to_imu
            ci_to_c0 = world_to_c0 @ ci_to_world

            # get 3D object detection in camera frame
            boxes_3d = []
            if frame_idx in detection.keys():
                boxes_3d = [kitti_obj_to_bbox3d(o, timestamp=frame_idx, obj_type=obj_type) for o in detection[frame_idx]]

            # map boxes from current camera frame to first camera frame
            for box in boxes_3d:
                box.transform_(ci_to_c0, 'c0')

            # invoke tracking functionalities
            all_measurements = [cvt_bbox3d_to_measurement(box) for box in boxes_3d]
            manager.run_(all_measurements, frame_idx)

            # log result
            c0_to_cam = np.linalg.inv(ci_to_c0)
            for tracklet in manager.all_tracklets:
                if tracklet.tail.stamp == frame_idx and not tracklet.just_born:
                    # convert tracklet's tail to a Bbox3D
                    box = cvt_state_to_bbox3d(tracklet.tail, tracklet.id, tracklet.most_recent_meas_score,
                                              tracklet.kitti_meas_alpha)
                    # map box back to camera frame for drawing
                    box.transform_(c0_to_cam, 'camera')
                    # project box onto image for 2D tracking benchmarking
                    box_to_cam = get_box_to_cam_trans(box)
                    proj = box.project_on_image(box_to_cam, calib.cam_proj_mat)
                    # write result
                    box_2d = np.append(np.amin(proj, axis=0), np.amax(proj, axis=0)).tolist()
                    result_file.write(format_result(frame_idx, tracklet.id, obj_type, box, box_2d))

        result_file.close()
        print('\n--------------------------------------------')
