import os
import sys
import numpy as np
from tqdm import tqdm

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

from utils.front_end.waymo_parser import parse_detection_file, waymo_object_to_bbox3d, parse_ego_pose_file
from utils.data_classes import Bbox3D
from tracking.tracklet_manager import TrackletManager
from tracking.state import cvt_state_to_bbox3d
from tracking.measurement import cvt_bbox3d_to_measurement
from global_config import waymo_to_nuscenes, GlobalConfig


# global constants in this script
assert GlobalConfig.dataset == 'waymo', "dataset must be 'waymo' instead of {}".format(GlobalConfig.dataset)
report_conf_thres = GlobalConfig.tracklet_report_conf_threshold
waymo_tracking_names = {
    'CYCLIST': label_pb2.Label.TYPE_CYCLIST,
    'PEDESTRIAN': label_pb2.Label.TYPE_PEDESTRIAN,
    'VEHICLE': label_pb2.Label.TYPE_VEHICLE
}
data_root = './data/waymo'
result_root = './results/waymo/val_pointpillars_ppba'
nbr_terminals = GlobalConfig.nbr_terminals


def format_tracking_result(box, context_name, timestamp_micros):
    """ Convert a Bbox3D into an instance of class metrics_pb2.Object so that it can be serialized in
    Waymo OD 's format

    Args:
        box (Bbox3D): box converted from tracklet's State
        context_name (str): name of context
        timestamp_micros (int): time stamp of the frame where this box appears
    Returns:
        metrics_pb2.Object
    """
    assert box.frame == 'ego_vehicle', "box must be in 'ego_vehicle' frame, while right now it is in {}".format(box.frame)
    o = metrics_pb2.Object()
    o.context_name = context_name
    o.frame_timestamp_micros = timestamp_micros
    # populate box & score
    waymo_box = label_pb2.Label.Box()
    waymo_box.center_x = box.center[0]
    waymo_box.center_y = box.center[1]
    waymo_box.center_z = box.center[2]
    waymo_box.length = box.l
    waymo_box.width = box.w
    waymo_box.height = box.h
    waymo_box.heading = box.yaw
    o.object.box.CopyFrom(waymo_box)
    o.score = box.score
    o.object.id = '{}-{}'.format(box.obj_type, box.id)
    o.object.type = waymo_tracking_names[box.obj_type]
    return o


def generate_tracking_results_for_one_record(context_name, context_idx, nbr_contexts):
    """Generate tracking results for one record represents by the context_name

    Args:
        context_name (str): name of the context
        context_idx (int): context index
        nbr_contexts (int): total number of contexts

    Returns:
        metrics_pb2.Objects: the object stores all the tracking result of this record
    """
    print('\nGenerate tracking result for context {}/{}'.format(context_idx, nbr_contexts))

    # get path to ego_poses & detections file
    ego_poses_file = os.path.join(data_root, context_name, '{}_stamp_and_pose.txt'.format(context_name))
    detections_file = os.path.join(data_root, context_name, '{}_unpacked_detection.txt'.format(context_name))

    # parse these files to get the dict {timestamp_micros: information}
    ego_poses = parse_ego_pose_file(ego_poses_file)
    detections = parse_detection_file(detections_file)

    # make sure timestamp are in ascending order
    all_timestamps = sorted(list(ego_poses.keys()))

    # init tracklet managers
    managers = {name: TrackletManager(name) for name in waymo_to_nuscenes.keys()}
    tracking_results = metrics_pb2.Objects()  # to store tracking results

    # main loop
    frame_idx = 0
    for timestamp in tqdm(all_timestamps):
        # get detection of this frame
        boxes_3d = []
        if timestamp in detections.keys():
            boxes_3d = [waymo_object_to_bbox3d(o, frame_idx) for o in detections[timestamp]]

        # convert detection in 'ego_vehicle' frame to measurement in 'world' frame
        ego_to_world = ego_poses[timestamp].ego_pose
        all_measurements = {name: [] for name in managers.keys()}
        for box in boxes_3d:
            box.transform_(ego_to_world, 'world')
            all_measurements[box.obj_type].append(cvt_bbox3d_to_measurement(box))

        # invoke tracking
        for obj_type, manager in managers.items():
            manager.run_(all_measurements[obj_type], frame_idx)

        # log tracking result
        world_to_ego = np.linalg.inv(ego_to_world)
        for obj_type, manager in managers.items():
            for tracklet in manager.all_tracklets:
                if tracklet.tail.stamp == frame_idx and not tracklet.just_born and tracklet.conf > 0.1:
                    box = cvt_state_to_bbox3d(tracklet.tail, tracklet.id, tracklet.most_recent_meas_score,
                                              frame='world',
                                              obj_type=obj_type)
                    box.transform_(world_to_ego, 'ego_vehicle')  # map back to ego_vehicle frame
                    o = format_tracking_result(box, context_name, timestamp)
                    tracking_results.objects.append(o)

        # move on
        frame_idx += 1

    print('writing tracking results of {}'.format(context_name))
    with open(os.path.join(result_root, '{}_tracking_result.bin'.format(context_name)), 'wb') as f:
        f.write(tracking_results.SerializeToString())
    print('----------------------------------------------------------------------------\n')


if __name__ == '__main__':
    # Usage: python waymo_generate_tracking_results.py index_of_process
    process_idx = int(sys.argv[1])
    assert process_idx < nbr_terminals, 'maximum value of process_idx is {}'.format(nbr_terminals-1)

    context_names = [folder for folder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, folder))]
    context_names.sort()
    nbr_contexts_per_process = int(len(context_names) * 1.0 / nbr_terminals)

    _start = process_idx * nbr_contexts_per_process
    _end = (process_idx+1) * nbr_contexts_per_process if process_idx < nbr_terminals - 1 else len(context_names)
    for idx, context in enumerate(context_names[_start: _end]):
        generate_tracking_results_for_one_record(context, _start + idx, len(context_names))
