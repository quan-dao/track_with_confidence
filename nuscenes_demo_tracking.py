import os
import cv2
import numpy as np

from nuscenes.nuscenes import NuScenes
from utils.front_end.nuscenes_parser import parse_detection_file, nuscenes_object_to_bbox3d
from utils.back_end.nuscenes_visualization import NuCamera, find_camera, draw_bbox3d_on_image
from tracking.measurement import cvt_bbox3d_to_measurement
from tracking.tracklet_manager import TrackletManager
from tracking.state import cvt_state_to_bbox3d
from global_config import nuscenes_tracking_names


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
tracklet_managers = {obj_type: TrackletManager(obj_type) for obj_type in nuscenes_tracking_names}

# prepare for rendering 6 cameras
imsize = (640, 360)
window_name = '{}'.format(scene['name'])
cv2.namedWindow(window_name)
cv2.moveWindow(window_name, 0, 0)
canvas = np.ones((2 * imsize[1], 3 * imsize[0], 3), np.uint8)
layout = {
    'CAM_FRONT_LEFT': (0, 0),
    'CAM_FRONT': (imsize[0], 0),
    'CAM_FRONT_RIGHT': (2 * imsize[0], 0),
    'CAM_BACK_LEFT': (0, imsize[1]),
    'CAM_BACK': (imsize[0], imsize[1]),
    'CAM_BACK_RIGHT': (2 * imsize[0], imsize[1]),
}
horizontal_flip = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']  # Flip these for aesthetic reasons.

frame_idx = 0
sample_token = scene['first_sample_token']
while sample_token != '':
    sample = nusc.get('sample', sample_token)

    # get camera information
    camera_tokens = [token for k, token in sample['data'].items() if 'CAM' in k]
    cam_infos = tuple([NuCamera(cam_token, nusc) for cam_token in camera_tokens])

    # get detection of this frame
    boxes_3d = []
    if frame_idx in detections.keys():
        boxes_3d = [nuscenes_object_to_bbox3d(o) for o in detections[frame_idx]]

    # ------------------------------------------------------------------#
    # tracking code
    # should go
    # here
    # convert boxes to measurements
    all_measurements = {obj_type: [] for obj_type in nuscenes_tracking_names}
    for box in boxes_3d:
        all_measurements[box.obj_type].append(cvt_bbox3d_to_measurement(box))

    for obj_type, manager in tracklet_managers.items():
        manager.run_(all_measurements[obj_type], frame_idx)

    # ------------------------------------------------------------------#

    # draw boxes on camera images
    for obj_type, manager in tracklet_managers.items():
        for tracklet in manager.all_tracklets:
            if tracklet.tail.stamp == frame_idx and not tracklet.just_born:
                box = cvt_state_to_bbox3d(tracklet.tail, tracklet.id, tracklet.most_recent_meas_score, frame='world')
                # find camera this box is visible on and draw
                find_camera(box, cam_infos)
                draw_bbox3d_on_image(box, cam_infos, '{}:{}'.format(obj_type[:3], tracklet.id))

    for cam in cam_infos:
        cam.im = cv2.resize(cam.im, imsize)
        if cam.name in horizontal_flip:
            cam.im = cam.im[:, ::-1, :]
        canvas[
            layout[cam.name][1]: layout[cam.name][1] + imsize[1],
            layout[cam.name][0]: layout[cam.name][0] + imsize[0], :
        ] = cam.im

    cv2.imshow(window_name, canvas)
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

    # move on
    frame_idx += 1
    sample_token = sample['next']

cv2.destroyAllWindows()
