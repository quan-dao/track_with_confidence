import json
from collections import namedtuple

from utils.data_classes import Bbox3D


NuScenesObject = namedtuple('NuScenesObject', 'stamp, obj_type, h, w, l, x, y, z, yaw, score, ref_frame')


def parse_detection_file(detection_file):
    """ Read a NuScenes detection file written in KITTI format

    Args:
        detection_file (str): path to detection file written in kitti format
    Returns:
        dict: {frame_index: list(NuScenesObject)}
    """
    read_objects = {}
    with open(detection_file, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            # first 2 entries are int
            line[:2] = [int(x) for x in line[:2]]
            # the rest are float
            line[2:] = [float(x) for x in line[2:]]

            o = NuScenesObject(*line)

            frame_idx = o.stamp
            try:
                read_objects[frame_idx].append(o)
            except KeyError:
                read_objects[frame_idx] = [o]

    return read_objects


def nuscenes_object_to_bbox3d(o):
    """Convert a NuScenesObject to a Bbox3D

    Args:
        o (NuScenesObject): a line in kitti-format detection file bundled by NuScenesObject
    Returns:
         Bbox3D
    """
    return Bbox3D(o.x, o.y, o.z, o.l, o.w, o.h, o.yaw, frame=o.ref_frame, obj_type=o.obj_type, stamp=o.stamp,
                  score=o.score)

