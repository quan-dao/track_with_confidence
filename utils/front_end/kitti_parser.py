import numpy as np
from collections import namedtuple

from utils.data_classes import Bbox2D, Bbox3D


# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, imu_to_world')

# Bundle kitti-format object:
# x1, y1, x2, y2 define 2D bbox
# score is object detector confidence score
# h, w, l, x, y, z, rot_y define 3D bbox whose coordinates are in camera frame
# alpha is observation angle
KittiObject = namedtuple('KittiObject', 'frame, type, x1, y1, x2, y2, score, h, w, l, x, y, z, rot_y, alpha')


class Calibration(object):
    """Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2'].reshape((3, 4))
        # Rotation from reference camera coord to rect camera coord
        self.R0 = np.eye(4)
        self.R0[:3, :3] = calibs['R_rect'].reshape((3, 3))
        # projection matrix of rectified camera
        self.cam_proj_mat = self.P @ self.R0  # (3, 4)
        # Rigid transform from Velodyne coord to reference camera coord
        self.velo_to_cam = np.eye(4)
        self.velo_to_cam[:3, :] = calibs['Tr_velo_cam'].reshape((3, 4))
        self.cam_to_velo = np.linalg.inv(self.velo_to_cam)
        # rigid transform from velodyne to imu
        self.imu_to_velo = np.eye(4)
        self.imu_to_velo[:3, :] = calibs['Tr_imu_velo'].reshape((3, 4))
        self.velo_to_imu = np.linalg.inv(self.imu_to_velo)
        # rigid transform from cam to imu
        self.cam_to_imu = self.velo_to_imu @ self.cam_to_velo
        self.imu_to_cam = np.linalg.inv(self.cam_to_imu)


    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key_and_value = line.split()
                key = key_and_value[0]
                if ':' in key:
                    key = key[:-1]
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for i, x in enumerate(key_and_value) if i > 0])
                except ValueError:
                    pass

        return data


def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't]
    """
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transformation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.
       Poses are given in an East-North-Up coordinate system
       whose origin is the first GPS position.

    Args:
        oxts_files (list): list of oxts files
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                imu_to_world = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, imu_to_world))

    return oxts


def parse_detection_file(detection_file):
    """ Read a kitti-format detection file

    Args:
        detection_file (str): path to detection file written in kitti format

    Returns:
        dict: key is frame index, value is list of objects in this frame
    """
    read_objects = {}
    with open(detection_file, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            # first 2 entries are int
            line[:2] = [int(x) for x in line[:2]]
            # the rest are float
            line[2:] = [float(x) for x in line[2:]]

            o = KittiObject(*line)

            frame_idx = o.frame
            try:
                read_objects[frame_idx].append(o)
            except KeyError:
                read_objects[frame_idx] = [o]

    return read_objects


def kitti_obj_to_bbox2d(obj):
    """ Create a Bbox2D from a KittiObject"""
    return Bbox2D(obj.x1, obj.y1, obj.x2, obj.y2, obj.type)


def kitti_obj_to_bbox3d(obj):
    return Bbox3D(obj.x, obj.y, obj.z, obj.l, obj.w, obj.h, obj.rot_y, 'kitti', alpha=obj.alpha, frame='camera')


def get_box_to_cam_trans(box):
    """Get transformation from a 3D bbox (defined in kitti format) to camera

    Args:
        box (Bbox3D): 3D box

    Returns:
        np.ndarray: homogeneous transformation from box to camera
    """
    trans = np.eye(4)
    trans[:3, :3] = roty(box.yaw)
    trans[:3, 3] = box.center
    return trans
