import os
import tensorflow.compat.v1 as tf
import cv2
import numpy as np

from waymo_open_dataset import dataset_pb2 as open_dataset

from utils.front_end.waymo_parser import parse_detection_file, waymo_object_to_bbox3d, parse_ego_pose_file
from utils.back_end.waymo_visualization import WaymoCamera, find_camera, draw_bbox3d_on_image


record_dir = '/home/user/dataset/waymo-open/segments/val/'
records = [file for file in os.listdir(record_dir) if file.endswith('.tfrecord')]
records.sort()
dataset = tf.data.TFRecordDataset(os.path.join(record_dir, records[7]), compression_type='')

# prepare canvas for rendering sequence
resized_width, resized_height_front, resized_height_back= (192*2, 128*2, 104*2)
layout = {
    'FRONT_LEFT': (0, 0),
    'FRONT': (0, resized_width),
    'FRONT_RIGHT': (0, 2 * resized_width),
    'SIDE_LEFT': (resized_height_front, 0),
    'SIDE_RIGHT': (resized_height_front, 2 * resized_width)
}
canvas = np.ones((resized_height_front + resized_height_back, 3 * resized_width, 3), np.uint8)
window_name = 'Sample context'
cv2.namedWindow(window_name)
cv2.moveWindow(window_name, 0, 0)

context_name = None
ego_poses_file = './data/waymo'
detections_file = './data/waymo'
detections = None
ego_poses = None  # TODO: parse ego_pose file to make a dict {timestamp_micro: ego_pose}

waymo_cameras = []
frame_idx = 0
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    if frame_idx == 0:
        # get context name & context folder
        context_name = frame.context.name
        ego_poses_file = os.path.join(ego_poses_file, context_name, '{}_stamp_and_pose.txt'.format(context_name))
        detections_file = os.path.join(detections_file, context_name, '{}_unpacked_detection.txt'.format(context_name))
        detections = parse_detection_file(detections_file)
        ego_poses = parse_ego_pose_file(ego_poses_file)
        # get camera calibration
        for camera_calibration in frame.context.camera_calibrations:
            name = open_dataset.CameraName.Name.Name(camera_calibration.name)
            intrinsic = list(camera_calibration.intrinsic)
            extrinsic = np.asarray(camera_calibration.extrinsic.transform).reshape(4, 4)
            width = camera_calibration.width
            height = camera_calibration.height
            waymo_cameras.append(WaymoCamera(name, intrinsic, extrinsic, width, height))

    # get detection of this frame
    boxes_3d = []
    if frame.timestamp_micros in detections.keys():
        boxes_3d = [waymo_object_to_bbox3d(o, frame_idx) for o in detections[frame.timestamp_micros]]

    #------------------------------------------------------------------
    # tracking code goes here
    # TODO: map box to global frame
    ego_to_world = ego_poses[frame.timestamp_micros].ego_pose
    for box in boxes_3d:
        box.transform_(ego_to_world, 'world')

    # TODO: invoke tracking

    # TODO: map back to ego_vehicle frame for rendering
    world_to_ego = np.linalg.inv(ego_to_world)
    for box in boxes_3d:
        box.transform_(world_to_ego, 'ego_vehicle')
    # ------------------------------------------------------------------

    all_images = {}
    for camera_image in frame.images:
        camera_name = open_dataset.CameraName.Name.Name(camera_image.name)
        all_images[camera_name] = tf.image.decode_jpeg(camera_image.image).numpy()
    # attach these images into waymo_cameras
    for cam in waymo_cameras:
        cam.im = all_images[cam.name]

    # draw
    for i, box in enumerate(boxes_3d):
        if box.score > 0.3 and box.obj_type == 'VEHICLE':
            find_camera(box, waymo_cameras)
            draw_bbox3d_on_image(box, waymo_cameras, label=i)
    for cam in waymo_cameras:
        _h = resized_height_front if 'FRONT' in cam.name else resized_height_back
        cam.im = cv2.resize(cam.im, (resized_width, _h))
        cam.im = cam.im[:, :, ::-1]  # RGB to BGR
        canvas[
            layout[cam.name][0]: layout[cam.name][0] + _h,
            layout[cam.name][1]: layout[cam.name][1] + resized_width, :
        ] = cam.im

    cv2.imshow(window_name, canvas)
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

    # move on
    frame_idx += 1

cv2.destroyAllWindows()
