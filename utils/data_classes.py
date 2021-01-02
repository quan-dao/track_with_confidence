import numpy as np
from copy import deepcopy


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

    def __init__(self, center_x, center_y, center_z, length, width, height, yaw, dataset, **kwargs):
        """
        Args:
            center_x (float): x-coordinate of box's center
            center_y (float): y-coordinate of box's center
            center_z (float): z-coordinate of box's center
            length (float): dimension along box's local x-axis
            width (float): dimension along box's local y-axis
            height (float): dimension along box's local z-axis
            yaw (float):  rotation around up direction
            dataset (str): dataset of this box
        """
        assert dataset in ['kitti', 'waymo', 'nuscenes']
        self.center = np.array([center_x, center_y, center_z])
        self.l, self.w, self.h = length, width, height
        self.yaw = yaw
        self.dataset = dataset
        # optional fields
        self.frame = None
        self.obj_type = None
        self.cam_obs_angle = None  # applicable for KITTI dataset
        if 'frame' in kwargs.keys():
            self.frame = kwargs['frame']  # coordinate system where this box's pose is expressed
        if 'obj_type' in kwargs.keys():
            self.obj_type = kwargs['obj_type']
        if 'alpha' in kwargs.keys():
            self.cam_obs_angle = kwargs['alpha']

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
        if self.dataset != 'kitti':
            # waymo & nuscenes convention: x-forward, y-left, z-up, origin of box local frame is box center
            x_corners = self.l / 2.0 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
            y_corners = self.w / 2.0 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
            z_corners = self.h / 2.0 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        else:
            # kitti convention: z-forward, x-right, y-down, origin of box local frame is center of bottom face
            x_corners = self.l / 2.0 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
            y_corners = self.h * np.array([-1, -1, 0, 0, -1, -1, 0, 0])
            z_corners = self.w / 2.0 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        return corners

    def transform_(self, trans_mat, dst_frame):
        """TODO: map box's coordinate from one frame to another

        Args:
            trans_mat (np.ndarray): homogeneous transformation matrix, shape (4, 4)
            dst_frame (str): name of destination frame
        """
        pass

    def project_on_image(self, box_to_cam, cam_proj_mat):
        """ Project box's corners on image using camera projection matrix

        Args:
            box_to_cam (np.ndarray): transformation matrix from box frame to camera frame, shape (4, 4)
            cam_proj_mat (np.ndarray): camera projection matrix, (3, 4)

        ReturnsL
            np.ndarray: projected corners, (8, 2)
        """
        corners = self.corners()  # (3, 8)
        corners = np.vstack((corners, np.ones((1, 8))))  # (4, 8)
        proj_corners = cam_proj_mat @ box_to_cam @ corners  # (3, 8)
        # normalize
        proj_corners = proj_corners / proj_corners[2, :]
        return proj_corners[:2, :].T
