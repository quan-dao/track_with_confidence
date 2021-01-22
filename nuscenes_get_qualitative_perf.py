import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from utils.front_end.nuscenes_parser import parse_detection_file, nuscenes_object_to_bbox3d
from utils.back_end.nuscenes_visualization import draw_bbox3d_bev, nuscenes_annotation_to_tracking, gt_to_box
from tracking.measurement import cvt_bbox3d_to_measurement
from tracking.tracklet_manager import TrackletManager
from tracking.state import cvt_state_to_bbox3d
from global_config import GlobalConfig


np.random.seed(23)

nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
detection_dir = './data/nuscenes/megvii_detection'
val_scene_tokens = [name[:-4] for name in os.listdir(detection_dir) if name.endswith('.txt')]
print('num_val_scenes: ', len(val_scene_tokens))
for i, scene in enumerate(nusc.scene):
    if scene['token'] in val_scene_tokens:
        print('{}:\t {}'.format(i, scene['token']))


scene = nusc.scene[1]
detection_file = os.path.join(detection_dir, '{}.txt'.format(scene['token']))
detections = parse_detection_file(detection_file)

# init tracklet managers
# tracklet_managers = {obj_type: TrackletManager(obj_type) for obj_type in nuscenes_tracking_names}
track_class = 'pedestrian'
tracklet_manager = TrackletManager(track_class)
report_conf_thresh = GlobalConfig.nuscenes_tracklet_report_conf_threshold

# prepare for rendering BEV
fig, ax = plt.subplots(1, 3)
ax[0].set_title('Ours')
ax[1].set_title('Ground Truth')
ax[2].set_title('MEGVII Detection')
tracklet_color = {}
gt_color = {}

# main loop
sample_token = scene['first_sample_token']
for frame_idx in tqdm(range(scene['nbr_samples'])):
    sample = nusc.get('sample', sample_token)

    # get ground truth of this frame
    gt_boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['category_name'] in nuscenes_annotation_to_tracking.keys():
            box = gt_to_box(ann)
            if box.obj_type == track_class:
                if box.id not in gt_color.keys():
                    gt_color[box.id] = tuple(np.random.uniform(size=3).tolist())
                draw_bbox3d_bev(box, gt_color[box.id], ax[1])


    # get detection of this frame
    boxes_3d = []
    if frame_idx in detections.keys():
        boxes_3d = [nuscenes_object_to_bbox3d(o) for o in detections[frame_idx] if o.obj_type == track_class]

    # convert boxes to measurements
    all_measurements = [cvt_bbox3d_to_measurement(box) for box in boxes_3d]

    # invoke tracklet manager
    tracklet_manager.run_(all_measurements, frame_idx)

    # draw tracklet
    for tracklet in tracklet_manager.all_tracklets:
        if tracklet.tail.stamp == frame_idx and not tracklet.just_born and tracklet.conf > report_conf_thresh[track_class]:
            box = cvt_state_to_bbox3d(tracklet.tail, tracklet.id, tracklet.most_recent_meas_score, frame='world')
            if tracklet.id not in tracklet_color.keys():
                tracklet_color[tracklet.id] = tuple(np.random.uniform(size=3).tolist())
            draw_bbox3d_bev(box, tracklet_color[tracklet.id], ax[0])

    # draw box on ax
    for box in boxes_3d:
        draw_bbox3d_bev(box, 'r', ax[2])

    # move on
    sample_token = sample['next']

plt.show()

