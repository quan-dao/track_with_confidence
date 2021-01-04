import numpy as np

from covariances import NuScenesCovariances as Cov
from global_config import GlobalConfig, get_nuscenes_name
from utils.data_classes import Bbox3D


# global constants (their value are set in global_config.py)
dim_z = GlobalConfig.dim_z  # [x, y, z, yaw]
yaw_idx = GlobalConfig.yaw_index
dataset = GlobalConfig.dataset


class Measurement(object):
    """For bounding boxes (Bbox3D) interface with tracking functionality"""
    def __init__(self, position, yaw, size, timestamp, obj_type, det_score, kitti_alpha=None):
        """
        Args:
             position (np.ndarray): position of box's center in global frame, shape (3, )
             yaw (float): orientation around vertical direction
             size (np.ndarray): (l, w, h), shape (3, )
             timestamp (int): index of the frame when this measurement is created
             obj_type (str): type of this measurement
             det_score (float): detection score
             kitti_alpha (float): camera observation angle
        """
        assert len(position.shape) == 1, 'position must be either an array'
        # measurement vector & covariance matrix
        self.z = np.append(position, yaw).reshape(dim_z, 1)
        nusc_name = get_nuscenes_name(obj_type, dataset)
        self.R = np.diag([Cov.R[nusc_name]['x'], Cov.R[nusc_name]['y'], Cov.R[nusc_name]['z'], Cov.R[nusc_name]['yaw']])
        # other fields
        self.size = size
        self.stamp = timestamp
        self.score = det_score
        self.obj_type = obj_type
        self.kitti_alpha = None
        if kitti_alpha is not None:
            self.kitti_alpha = kitti_alpha

    def __repr__(self):
        return 'Measurement| position: [{:.3f}, {:.3f}, {:.3f}],  yaw: {:.3f}, size: [{:.3f}, {:.3f}, {:.3f}],  ' \
               'obj_type: {},  score: {:.3f}'.format(*self.z[:3, 0].tolist(), self.z[yaw_idx, 0], *self.size.tolist(),
                                                     self.obj_type, self.score)


def cvt_bbox3d_to_measurement(box):
    """Convert a 3D Bbox to measurement

    Args:
        box (Bbox3D): a 3D bbox
    Returns:
        Measurement: a measurement
    """
    assert box.frame == 'world' or box.frame == 'c0', 'box must be in world (in case of Waymor or NuScenes) or ' \
                                                      '1st camera frame (in case of KITTI) for tracking to work'
    assert box.stamp is not None, 'box must have a timestamp'
    assert box.score is not None, 'box must have detection score'
    if dataset == 'kitti':
        return Measurement(box.center, box.yaw, np.array([box.l, box.w, box.h]), box.stamp, box.obj_type, box.score,
                           kitti_alpha=box.cam_obs_angle)
    else:
        return Measurement(box.center, box.yaw, np.array([box.l, box.w, box.h]), box.stamp, box.obj_type, box.score)
