import numpy as np
from copy import deepcopy

from utils.geometry import roty, rotz
from global_config import GlobalConfig


# global constants (their value are set in global_config.py)
dataset = GlobalConfig.dataset


class Bbox2D(object):
    """An axis-aligned bounding box on image"""
    def __init__(self, x_min, y_min, x_max, y_max, object_type):
        """
        Args:
             x_min (float): x-pixel-coordinate of top left corner
             y_min (float): y-pixel-coordinate of top left corner
             x_max (float): x-pixel-coordinate of bottom right corner
             y_max (float): y-pixel-coordinate of bottom right corner
             object_type (str): type of this box
        """
        self.obj_type = object_type
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def corners(self):
        """ Compute box's corners

        Returns
            np.ndarray: pixel-coordinate of box's corners, shape (4, 2)
        """
        return np.array([
            [self.x_min, self.y_min],
            [self.x_max, self.y_min],
            [self.x_max, self.y_max],
            [self.x_min, self.y_max]
        ])

    def clamp(self, img_width, img_height):
        """Clamp box's size with image size

        Args:
            img_width (int): image width
            img_height (int): image height

        Returns:
            Bbox2D: a new box with size clamped by image size
        """
        clamped_box = deepcopy(self)
        clamped_box.x_min = max(0.0, min(self.x_min, img_width))
        clamped_box.x_max = max(0.0, min(self.x_max, img_width))
        clamped_box.y_min = max(0.0, min(self.y_min, img_height))
        clamped_box.y_max = max(0.0, min(self.y_max, img_height))
        return clamped_box


class Bbox3D(object):
    """A 3D bounding box"""

    def __init__(self, center_x, center_y, center_z, length, width, height, yaw, **kwargs):
        """
        Args:
            center_x (float): x-coordinate of box's center
            center_y (float): y-coordinate of box's center
            center_z (float): z-coordinate of box's center
            length (float): dimension along box's local x-axis
            width (float): dimension along box's local y-axis
            height (float): dimension along box's local z-axis
            yaw (float):  rotation around up direction
        """
        self.center = np.array([center_x, center_y, center_z])
        self.l, self.w, self.h = length, width, height
        self.yaw = yaw
        # optional fields
        self.frame = None  # coordinate system where this box's pose is expressed
        self.obj_type = None
        self.cam_obs_angle = None  # applicable for KITTI dataset
        self.stamp = None
        self.score = None  # detection score
        self.id = None  # tracking id
        self.is_on_camera = None  # camera on which this box is visible, applicable for NuScenes and Waymo
        if 'frame' in kwargs.keys():
            self.frame = kwargs['frame']
        if 'obj_type' in kwargs.keys():
            self.obj_type = kwargs['obj_type']
        if 'alpha' in kwargs.keys():
            self.cam_obs_angle = kwargs['alpha']
        if 'stamp' in kwargs.keys():
            self.stamp = kwargs['stamp']
        if 'score' in kwargs.keys():
            self.score = kwargs['score']
        if 'id' in kwargs.keys():
            self.id = kwargs['id']

    def __repr__(self):
        re = 'Bbox3D| center:[{:.3f}, {:.3f}, {:.3f}],  size:[{:.3f}, {:.3f}, {:.3f}],  yaw:{:.3f}'.format(
            *self.center.tolist(), self.l, self.w, self.h, self.yaw)
        if self.frame:
            re += '  frame:{}'.format(self.frame)
        if self.obj_type:
            re += '  obj_type:{}'.format(self.obj_type)
        if self.cam_obs_angle:
            re += '  cam_obs_angle:{:.3f}'.format(self.cam_obs_angle)
        return re

    def corners(self):
        """ Compute box 's corners in box's local frame. Convention: \
            forward face is 0-1-2-3, backward face is 4-5-6-7, top face is 0-1-5-4, bottom face is 3-2-6-7
        """
        if dataset != 'kitti':
            # waymo & nuscenes convention: x-forward, y-left, z-up, origin of box local frame is box center
            x_corners = self.l / 2.0 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
            y_corners = self.w / 2.0 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
            z_corners = self.h / 2.0 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        else:
            # kitti convention: x-forward, y-down, origin of box local frame is center of bottom face
            x_corners = self.l / 2.0 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
            y_corners = self.h * np.array([-1, -1, 0, 0, -1, -1, 0, 0])
            z_corners = self.w / 2.0 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        return corners

    def transform_(self, current_to_dst, dst_frame):
        """Map box's coordinate from one frame to another

        Args:
            current_to_dst (np.ndarray): homogeneous transformation matrix, shape (4, 4)
            dst_frame (str): name of destination frame
        """
        # construct pose of box in its current frame of reference
        pose = np.eye(4)
        if dataset == 'kitti':
            # kitti convention: vertical direction is y-axis
            pose[:3, :3] = roty(self.yaw)
        else:
            # waymo & nuscenes: vertical direction is z-axis
            pose[:3, :3] = rotz(self.yaw)
        pose[:3, 3] = self.center
        # map pose to new frame
        pose = current_to_dst @ pose
        # update box's center & yaw accordingly
        self.center = pose[:3, 3]
        if dataset == 'kitti':
            # kitti convention: vertical direction is y-axis
            self.yaw = np.arctan2(pose[0, 2], pose[0, 0])
        else:
            # waymo & nuscenes: vertical direction is z-axis
            self.yaw = np.arctan2(pose[1, 0], pose[0, 0])
        self.frame = dst_frame

    def project_on_image(self, box_to_cam, cam_proj_mat):
        """ Project box's corners on image using camera projection matrix

        Args:
            box_to_cam (np.ndarray): transformation matrix from box frame to camera frame, shape (4, 4)
            cam_proj_mat (np.ndarray): camera projection matrix, (3, 4)

        Returns:
            np.ndarray: projected corners, (8, 2)
        """
        corners = self.corners()  # (3, 8)
        corners = np.vstack((corners, np.ones((1, 8))))  # (4, 8)
        proj_corners = cam_proj_mat @ box_to_cam @ corners  # (3, 8)
        # normalize
        proj_corners = proj_corners / proj_corners[2, :]
        return proj_corners[:2, :].T
