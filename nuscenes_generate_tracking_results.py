import os
import sys
from pyquaternion import Quaternion
from tqdm import tqdm
import json

from nuscenes.nuscenes import NuScenes
from utils.front_end.nuscenes_parser import parse_detection_file, nuscenes_object_to_bbox3d
from utils.data_classes import Bbox3D
from tracking.measurement import cvt_bbox3d_to_measurement
from tracking.tracklet_manager import TrackletManager
from tracking.state import cvt_state_to_bbox3d
from global_config import nuscenes_tracking_names, GlobalConfig


assert GlobalConfig.dataset == 'nuscenes', \
    "Change GlobalConfig.dataset (from {}) to 'nuscenes'".format(GlobalConfig.dataset)
nuscenes_root_trainval = GlobalConfig.nuscenes_root_trainval


def format_result(sample_token, box):
    """

    Args:
        sample_token (str): sample token
        box (Bbox3D): an instance of Bbox3D represents the reported state of a tracklet

    Note: only report tracklet which is active at current time step (tail.stamp == current_timestamp)
    Format tracking result for 1 single target as following
    sample_result {
        "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
        "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
        "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
        "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
        "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
        "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
        "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                           Note that the tracking_name cannot change throughout a track.
        "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                           We average over frame level scores to compute the track level score.
                                           The score is used to determine positive and negative tracks via thresholding.
    }
    """
    assert box.id is not None and box.score is not None and box.obj_type is not None
    rotation = Quaternion(axis=[0, 0, 1], angle=box.yaw).elements
    sample_result = {
        'sample_token': sample_token,
        'translation': box.center.tolist(),
        'size': [box.w, box.l, box.h],
        'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
        'velocity': [0, 0],
        'tracking_id': box.id,
        'tracking_name': box.obj_type,
        'tracking_score': box.score
    }
    return sample_result


if __name__ == '__main__':
    report_conf_thresh = GlobalConfig.nuscenes_tracklet_report_conf_threshold
    # Usage: python nuscenes_generate_tracking_results data_split datetime_str
    if len(sys.argv) != 3:
        print("Usage: python nuscenes_generate_tracking_results data_split datetime_str\n"
              "\trefer to nuscenes_run.sh for example")
        exit(1)
    data_split = sys.argv[1]
    datetime = sys.argv[2]

    results_dir = os.path.join('./results/nuscenes', '{}_{}'.format(data_split, datetime))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        raise ValueError("{} already exists. Choose a different datetime or datasplit".format(results_dir))

    detection_dir = './data/nuscenes/megvii_detection'
    val_scene_tokens = [name[:-4] for name in os.listdir(detection_dir) if name.endswith('.txt')]
    if len(val_scene_tokens) < 2:
        print("Execute script nuscenes_detection_extractor.py in utils/font_end before running tracking in NuScenes")
        exit(0)

    # initialize tracking results
    tracking_results = {}  # {sample_token: List[sample_result]}

    nusc = NuScenes(dataroot=nuscenes_root_trainval, version='v1.0-trainval', verbose=True)
    for sidx, scene_token in enumerate(val_scene_tokens):
        print('\n---------------------------------------------')
        print('Generate tracking results for scene {}/{}'.format(sidx, len(val_scene_tokens)))
        # get scene
        scene = nusc.get('scene', scene_token)

        # get scene detetections
        detection_file = os.path.join(detection_dir, '{}.txt'.format(scene['token']))
        detections = parse_detection_file(detection_file)

        # init tracklet managers
        tracklet_managers = {obj_type: TrackletManager(obj_type) for obj_type in nuscenes_tracking_names}

        # main for for each scene
        sample_token = scene['first_sample_token']
        for frame_idx in tqdm(range(scene['nbr_samples'])):
            # get detection of this frame as list of Bbox3D
            boxes_3d = []
            if frame_idx in detections.keys():
                boxes_3d = [nuscenes_object_to_bbox3d(o) for o in detections[frame_idx]]

            # convert boxes_3d to measurements
            all_measurements = {obj_type: [] for obj_type in nuscenes_tracking_names}
            for box in boxes_3d:
                all_measurements[box.obj_type].append(cvt_bbox3d_to_measurement(box))

            # invoke tracklet_managers for tracking
            for obj_type, manager in tracklet_managers.items():
                manager.run_(all_measurements[obj_type], frame_idx)

            # log tracking results
            tracking_results[sample_token] = []
            for obj_type, manager in tracklet_managers.items():
                for tracklet in manager.all_tracklets:
                    if tracklet.tail.stamp == frame_idx and not tracklet.just_born and \
                            tracklet.conf > report_conf_thresh[obj_type]:
                        box = cvt_state_to_bbox3d(tracklet.tail, tracklet.id, score=tracklet.most_recent_meas_score,
                                                  frame='world', obj_type=tracklet.obj_type)
                        # find camera this box is visible on and draw
                        tracking_results[sample_token].append(format_result(sample_token, box))

            # move on to next frame
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']

        # end-of-scene
        print('---------------------------------------------\n')

    # save tracking result
    meta = {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False}
    output_data = {'meta': meta, 'results': tracking_results}
    with open(os.path.join(results_dir, 'nuscenes_{}_{}.json'.format(data_split, datetime)), 'w') as outfile:
        json.dump(output_data, outfile)

    print('Finish generating tracking results')
