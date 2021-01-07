from collections import namedtuple
from utils.data_classes import Bbox3D


WaymoObject = namedtuple('WaymoObject', 'timestamp_micro, obj_type, h, w, l, x, y, z, yaw, score, ref_frame')


def parse_detection_file(detection_file):
    """ Read a Waymo detection file written in KITTI format

    Args:
        detection_file (str): path to detection file written in kitti format
    Returns:
        dict: {timestamp_micro: list(NuScenesObject)}
    """
    read_objects = {}
    with open(detection_file, 'r') as f:
        for line in f.readlines():
            line.rstrip()  # remove newline character
            line = line.split(',')
            line[0] = int(line[0])  # timestamp_micro
            line[2: -1] = [float(x) for x in line[2: -1]]  # from h to score

            o = WaymoObject(*line)
            try:
                read_objects[o.timestamp_micro].append(o)
            except KeyError:
                read_objects[o.timestamp_micro] = [o]

    return read_objects


def waymo_object_to_bbox3d(o, frame_index):
    """Convert a NuScenesObject to a Bbox3D

    Args:
        o (WaymoObject): a line in kitti-format detection file bundled by WaymoObject
        frame_index (int): index of frame this object appears (go from 0). NOT timestamp_micro
    Returns:
         Bbox3D
    """
    return Bbox3D(o.x, o.y, o.z, o.l, o.w, o.h, o.yaw, frame=o.ref_frame, obj_type=o.obj_type, stamp=frame_index,
                  score=o.score)

    