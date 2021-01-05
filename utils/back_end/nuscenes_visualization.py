import numpy as np
from collections import namedtuple
import cv2
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from utils.data_classes import Bbox3D
from utils.geometry import rotz
from .drawing_functions import draw_bbox3d


class NuCamera:
    def __init__(self, camera_token, nusc):
        """
        Args:
            camera_token (str): token of camera data
            nusc (NuScenes): for browsing NuScenes database
        """
        cam_data = nusc.get('sample_data', camera_token)
        calib_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        # construct camera projection matrix that map a point in 'world' frame into image coordinate
        cam_to_ego = np.eye(4)
        cam_to_ego[:3, :3] = Quaternion(calib_sensor['rotation']).rotation_matrix
        cam_to_ego[:3, 3] = np.array(calib_sensor['translation'])
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
        ego_to_world[:3, 3] = np.array(ego_pose['translation'])
        cam_to_world = ego_to_world @ cam_to_ego
        self.world_to_cam = np.linalg.inv(cam_to_world)  # cache this for checking box visibility
        K = np.array(calib_sensor['camera_intrinsic'])
        self.proj_matrix = K @ self.world_to_cam[:3, :]  # shape (3, 4)
        # other field
        self.name = cam_data['channel']
        self.imsize = (cam_data['height'], cam_data['width'])
        self.im = cv2.imread(nusc.get_sample_data_path(camera_token))


def get_box_to_world(box):
    """Get transformation from a 3D bbox (defined in nuscenes format) to 'world' frame

    Args:
        box (Bbox3D): 3D box

    Returns:
        np.ndarray: homogeneous transformation from box to camera
    """
    trans = np.eye(4)
    trans[:3, :3] = rotz(box.yaw)
    trans[:3, 3] = box.center
    return trans


def compute_box_visibility(box, cam):
    """Compute visibility of a Bbox3D. Visibility is denoted by an int.
    -1: invisible, >=0 number of box's corners visible on an image

    Args:
        box (Bbox3D): before invoking this function, box is in 'world' frame, upon this function finishes, box is \
         put back into world frame
        cam (NuCamera): bundle of camera information
    Returns:
        int: -1 means invisible, >=0 number of box's corners visible on an image
    """
    assert box.frame == 'world', "This function is designed for box in 'world' frame"
    box.transform_(cam.world_to_cam, cam.name)
    if box.center[2] < 0:
        # box is behind camera
        box.transform_(np.linalg.inv(cam.world_to_cam), 'world')  # map box back to world frame
        return -1
    else:
        # map box back to world frame
        box.transform_(np.linalg.inv(cam.world_to_cam), 'world')
        # project box on camera
        proj = box.project_on_image(get_box_to_world(box), cam.proj_matrix)
        # count number of projected corners are inside the image
        valid_x = np.logical_and(proj[:, 0] > 1, proj[:, 0] < cam.imsize[1] - 1)
        valid_y = np.logical_and(proj[:, 1] > 1, proj[:, 1] < cam.imsize[0] - 1)
        visibility = np.sum(np.logical_and(valid_x, valid_y))
        return visibility


def find_camera(box, camera_infos):
    """Find camera on which a Bbox3D is visible on. The result is written into box.is_on_camera

    Args:
        box (Bbox3D): a Bbox3D in 'world' frame
        camera_infos (tuple[NuCamera]):
    """
    box_vis = [compute_box_visibility(box, cam) for cam in camera_infos]
    cam_idx = np.argmax(box_vis).item()
    if box_vis[cam_idx] > 2:
        box.is_on_camera = cam_idx


def draw_bbox3d_on_image(box, camera_infos, label=None):
    """Draw a Bbox3D whose visibility is known on its associating camera's image

    Args:
        box (Bbox3D): a Bbox3D in 'world' frame
        camera_infos (tuple[NuCamera]):
        label (str or int): box's label
    """
    if box.is_on_camera is not None:
        cam = camera_infos[box.is_on_camera]
        proj = box.project_on_image(get_box_to_world(box), cam.proj_matrix)
        draw_bbox3d(cam.im, proj, label)

