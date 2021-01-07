'''
Extract objects in detection file written in nuscenes-format & write them
in plain text file (.txt) in kitti-format
    timestamp_micro, obj_type, h, w, l, x, y, z, yaw, score, ref_frame
'''
import os
from tqdm import tqdm
from waymo_open_dataset.protos import metrics_pb2

detection_dir = '/home/user/dataset/waymo-open/zip_baseline_waymo_tracking_val_detection'
bin_files = [file for file in os.listdir(detection_dir) if file.endswith('.bin')]

context_root = '../../data/waymo'

OBJECT_TYPES = [
    'UNKNOWN',  # 0
    'VEHICLE',  # 1
    'PEDESTRIAN',  # 2
    'SIGN',  # 3
    'CYCLIST',  # 4
]

for bfile in bin_files:
    print('Extracting ', bfile)
    detections = metrics_pb2.Objects()
    with open(os.path.join(detection_dir, bfile), 'rb') as f:
        detections.ParseFromString(f.read())

    for o in tqdm(detections.objects):
        context_dir = os.path.join(context_root, o.context_name)
        assert os.path.exists(context_dir), 'Run waymo_context_parser.py before using this script'
        object_line = '%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%s\n' % (
            o.frame_timestamp_micros,
            OBJECT_TYPES[o.object.type],
            o.object.box.height, o.object.box.width, o.object.box.length,
            o.object.box.center_x, o.object.box.center_y, o.object.box.center_z,
            o.object.box.heading,
            o.score,
            'ego_vehicle'
        )
        detection_file = os.path.join(context_dir, '{}_unpacked_detection.txt'.format(o.context_name))
        if os.path.isfile(detection_file):
            with open(detection_file, 'a') as f:
                f.write(object_line)
        else:
            with open(detection_file, 'w') as f:
                f.write(object_line)

    print('------------------------------------')


