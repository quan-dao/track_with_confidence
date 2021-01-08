from waymo_open_dataset.protos import metrics_pb2
import os
from tqdm import tqdm

results_dir = 'results/waymo/val_pointpillars_ppba'
results_file = [file for file in os.listdir(results_dir) if file.endswith('.bin')]

all_results = metrics_pb2.Objects()
for file_name in tqdm(results_file):
    with open(os.path.join(results_dir, file_name), 'rb') as f:
        tracking_stream = f.read()
    # extract tracking result of this segment
    segment_res = metrics_pb2.Objects()
    segment_res.ParseFromString(tracking_stream)
    # save segment result of all_results
    for o in segment_res.objects:
        all_results.objects.append(o)

print('writing all_results to disk')
with open(os.path.join('results/waymo/', 'val_tracking.bin'), 'wb') as f:
    f.write(all_results.SerializeToString())


'''
bazel-bin/waymo_open_dataset/metrics/tools/create_submission  \
--input_filenames='/home/user/Desktop/python_ws/track-with-confidence-v1.0/results/waymo/val_tracking.bin' \
--output_filename='/home/user/Desktop/python_ws/track-with-confidence-v1.0/results/waymo/tmp/sub_val' \
--submission_filename='/home/user/Desktop/libs/waymo-od/waymo_open_dataset/metrics/tools/submission.txtpb'
'''
