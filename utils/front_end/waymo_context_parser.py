'''
Extract timestamps & vehicle poses from a tfrecord file into a plain text file
Each line is:
    timestamp_micros, 16 elements of ego_vehicle pose (row major order)
'''
import os
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset


record_dir = '/home/user/dataset/waymo-open/segments/val/'
contexts = [file for file in os.listdir(record_dir) if file.endswith('.tfrecord')]
contexts.sort()

for c in tqdm(contexts):
    dataset = tf.data.TFRecordDataset(os.path.join(record_dir, c), compression_type='')
    context_name = ''
    context_dir = '../../data/waymo'
    stamp_and_pose_file = None
    frame_idx = 0
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if frame_idx == 0:
            context_name = frame.context.name
            context_dir = os.path.join(context_dir, context_name)
            if not os.path.exists(context_dir):
                os.makedirs(context_dir)
                stamp_and_pose_file = open(os.path.join(context_dir, '{}_stamp_and_pose.txt'.format(context_name)), 'w')
            else:
                raise ValueError("context is already parsed")
        # extract timestamp & ego_vehicle pose
        line = '%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (frame.timestamp_micros, *list(frame.pose.transform))
        stamp_and_pose_file.write(line)

        # move on to the next frame
        frame_idx += 1

    # end of context
    stamp_and_pose_file.close()
