import numpy as np
import cv2
from copy import deepcopy

from utils.data_classes import Bbox3D
from utils.geometry import rotz
from .drawing_functions import draw_bbox3d


class WaymoCamera:
    # waymo defines camera orientation s.t x is forward, y is to the left, z is up
    # while OpenCV defines z is forward (same as optical axis), x to the right, y down
    # use following matrix to transform points in waymo camera frame to OpenCV camera frame
    ori_waymo_to_opencv = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]).astype(float)  # to orient waymo's camera frame into OpenCV convetion

    def __init__(self, name, intrinsic, extrinsic, im_width, im_height):
        """
        Args
            name (str): camera name
            intrinsic (list): f_u, f_v, c_u, c_v, k_{1, 2}, p_{1, 2}, k_3
            extrinsic (np.ndarray): transformation matrix from camera frame to ego_vehicle frame, shape (4, 4)
            im_width (int): image width
            im_height (int): image height
        """
        self.name = name
        self.K = np.array([
            [intrinsic[0], 0, intrinsic[2]],
            [0, intrinsic[1], intrinsic[3]],
            [0, 0, 1]
        ])
        self.distor_coef = np.array(intrinsic[4:])
        self.camera_to_vehicle = extrinsic  # [4, 4] transformation matrix
        self.vehicle_to_cam = np.linalg.inv(self.camera_to_vehicle)
        self.im_width = im_width
        self.im_height = im_height
        self.im = None  # to be updated every frame of a context

    def project_on_image(self, points):
        """Project a set of 3D points that are already in camera's frame onto camera's image

        Args:
            points (np.ndarray): a set of 3D points in camera frame, shape (3, n) (n is the number of points)
        Returns:
            np.ndarray: pixel coordinate of these points, shape (n, 2)
        """
        proj_pts, _ = cv2.projectPoints(points, np.zeros((3, 1)), np.zeros((3, 1)), self.K, self.distor_coef)
        return proj_pts.squeeze()  # shape (n, 2)


def compute_box_visibility(box, cam):
    """Compute visibility of a Bbox3D. Visibility is denoted by an int.
    -1: invisible, >=0 number of box's corners visible on an image

    Args:
        box (Bbox3D): before invoking this function, box is in 'world' frame, upon this function finishes, box is \
         put back into world frame
        cam (WaymoCamera): bundle of camera information
    Returns:
        int: -1 means invisible, >=0 number of box's corners visible on an image
    """
    assert box.frame == 'ego_vehicle', "This function is designed for box in 'ego_vehicle' frame, " \
                                       "while box.frame = {}".format(box.frame)
    box_in_cam = deepcopy(box)
    box_in_cam.transform_(cam.vehicle_to_cam, cam.name)
    if box_in_cam.center[0] < 0:
        # NOTE: in waymo's setup, camera frame's x-axis is the camera's optical axis
        # box is behind camera
        return -1
    else:
        # get box's corners in its local frame
        corners = box.corners()  # currently in box's local frame, shape (3, 8)
        corners = np.vstack((corners, np.ones((1, 8))))  # shape (4, 8)
        # construct box's pose in ego_vehicle frame
        box_to_ego = np.eye(4)
        box_to_ego[:3, :3] = rotz(box.yaw)
        box_to_ego[:3, 3] = box.center
        # map box's corners to camera frame
        corners = cam.ori_waymo_to_opencv @ cam.vehicle_to_cam @ box_to_ego @ corners
        # project on camera's image
        proj = cam.project_on_image(corners[:3, :])  # shape (8, 2)
        # count number of projected corners are inside the image
        valid_x = np.logical_and(proj[:, 0] > 1, proj[:, 0] < cam.im_width - 1)
        valid_y = np.logical_and(proj[:, 1] > 1, proj[:, 1] < cam.im_height - 1)
        visibility = np.sum(np.logical_and(valid_x, valid_y))
        return visibility


def find_camera(box, camera_infos):
    """Find camera on which a Bbox3D is visible on. The result is written into box.is_on_camera

    Args:
        box (Bbox3D): a Bbox3D in 'world' frame
        camera_infos (list[WaymoCamera]):
    """
    box_vis = [compute_box_visibility(box, cam) for cam in camera_infos]
    cam_idx = np.argmax(box_vis).item()
    if box_vis[cam_idx] > 3:
        box.is_on_camera = cam_idx


def draw_bbox3d_on_image(box, camera_infos, label=None):
    """Draw a Bbox3D whose visibility is known on its associating camera's image

    Args:
        box (Bbox3D): a Bbox3D in 'world' frame
        camera_infos (list[WaymoCamera]):
        label (str or int): box's label
    """
    if box.is_on_camera is not None:
        cam = camera_infos[box.is_on_camera]
        # get box's corners in its local frame
        corners = box.corners()  # currently in box's local frame, shape (3, 8)
        corners = np.vstack((corners, np.ones((1, 8))))  # shape (4, 8)
        # construct box's pose in ego_vehicle frame
        box_to_ego = np.eye(4)
        box_to_ego[:3, :3] = rotz(box.yaw)
        box_to_ego[:3, 3] = box.center
        # map box's corners to camera frame
        corners = cam.ori_waymo_to_opencv @ cam.vehicle_to_cam @ box_to_ego @ corners
        # project on camera's image
        proj = cam.project_on_image(corners[:3, :])  # shape (8, 2)
        draw_bbox3d(cam.im, proj, label)
