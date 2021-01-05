import os
import cv2

from nuscenes.nuscenes import NuScenes
from utils.front_end.nuscenes_parser import parse_detection_file, nuscenes_object_to_bbox3d
from utils.back_end.nuscenes_visualization import NuCamera, find_camera, draw_bbox3d_on_image


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

    for i, box in enumerate(boxes_3d):
        find_camera(box, cam_infos)
        if box.score > 0.3:
            draw_bbox3d_on_image(box, cam_infos, i)

    # display CAM_FRONT
    cam_front_idx = 0
    for i, cam in enumerate(cam_infos):
        if cam.name == 'CAM_FRONT':
            cam_front_idx = i

    cv2.imshow('CAM_FRONT', cam_infos[cam_front_idx].im)
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

    # move on
    frame_idx += 1
    sample_token = sample['next']

