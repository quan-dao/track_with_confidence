'''
Extract a subset of validation set ground truth
'''
import os
import numpy as np
import pickle
from tqdm import tqdm

from waymo_open_dataset.protos import metrics_pb2

data_root = './data/waymo'
context_names = [folder for folder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, folder))]
context_names.sort()

chosen_context_ids = np.random.randint(0, len(context_names), size=23).tolist()
chosen_contexts = [context_names[i] for i in chosen_context_ids]
with open(os.path.join(data_root, 'mini_val_contexts'), 'wb') as f:
    pickle.dump(chosen_contexts, f)

# load all ground truth
all_gt = metrics_pb2.Objects()
with open('/home/user/dataset/waymo-open/gt/validation_ground_truth_objects_gt.bin', 'rb') as f:
    all_gt.ParseFromString(f.read())

# extract objects in ground truth whose context name is in chosen_contexts
mini_val_objects = metrics_pb2.Objects()
for i in tqdm(range(len(all_gt.objects))):
    o = all_gt.objects[i]
    if o.context_name in chosen_contexts:
        mini_val_objects.objects.append(o)

with open(os.path.join(data_root, 'mini_val_ground_truth_objects.bin'), 'wb') as f:
    f.write(mini_val_objects.SerializeToString())
