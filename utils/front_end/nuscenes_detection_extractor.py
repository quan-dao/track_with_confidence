# extract objects in detection file written in nuscenes-format & write them
# in plain text file (.txt) in kitti-format
import os
import json
from tqdm import tqdm
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from global_config import nuscenes_tracking_names


def format_detection(frame_idx, d):
    """Convert a sample_result of nuscenes-format detections into a string in kitti-format

    Args:
        frame_idx (int): index (in the scene) of the frame when d appears
        d (dict): {
            "sample_token":       <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
            "translation":        <float> [3]   -- box location in m in the global frame: center_x, center_y, center_z.
            "size":               <float> [3]   -- box size in m: width, length, height.
            "rotation":           <float> [4]   -- box orientation as quaternion in the global frame: w, x, y, z.
            "velocity":           <float> [2]   -- box velocity in m/s in the global frame: vx, vy.
            "detection_name":     <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
            "detection_score":    <float>       -- Object prediction score between 0 and 1
            "attribute_name":     <str>         -- dont' care
        }
    Returns:
        str: frame_index, type, dimensions (h, w, l), location, yaw, score, frame of reference (world in case of nuscenes)
    """
    q = Quaternion(d['rotation'])
    yaw = q.angle if q.axis[2] > 0 else -q.angle
    re = '{},{},{},{},{},{},{},{},{},{},world\n'.format(
        frame_idx, d['detection_name'], d['size'][2], d['size'][0], d['size'][1], *d['translation'], yaw, d['detection_score']
    )
    return re


nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/v1.0-trainval', version='v1.0-trainval', verbose=True)

detection_file = '/home/user/dataset/nuscenes/nusc-detection/detection-megvii/megvii_val.json'
with open(detection_file, 'r') as f:
    detections = json.load(f)
print(detections['meta'])

unpack_dir = '../../data/nuscenes/megvii_detection'

processed_samples = set()
for sample_token in tqdm(detections['results'].keys()):
    if sample_token in processed_samples:
        continue

    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    scene_detections_file = open(os.path.join(unpack_dir, '{}.txt'.format(sample['scene_token'])), 'w')

    # main loop for processing a scene
    scene_sample_token = scene['first_sample_token']
    scene_timestamp = 0
    while scene_sample_token != '':
        # get detections of this sample
        sample_dets = detections['results'][scene_sample_token]  # list(sample_results)
        for det in sample_dets:
            if det['detection_name'] in nuscenes_tracking_names:
                scene_detections_file.write(format_detection(scene_timestamp, det))

        # save this sample token into processes_samples
        processed_samples.add(scene_sample_token)

        # move on
        scene_sample = nusc.get('sample', scene_sample_token)
        scene_timestamp += 1
        scene_sample_token = scene_sample['next']
    # finish for this scene
    scene_detections_file.close()


