import numpy as np

from global_config import GlobalConfig, get_nuscenes_name
from covariances import NuScenesCovariances as Cov
from utils.geometry import put_angle_in_range


# global constants (their value are set in global_config.py)
dim_z = GlobalConfig.dim_z  # [x, y, z, yaw]
yaw_idx = GlobalConfig.yaw_index
unmeasurable_state_cov = GlobalConfig.kf_unmeasurable__covariance


def is_null(val):
    """Check if a number is close to 0"""
    return np.abs(val) < 1e-5


class ConstantTurningRateVelocity(object):
    """ Motion model for Vehicle & Cyclist """
    dim_x = GlobalConfig.ctrv_dim_x
    num_velocities = dim_x - dim_z
    H = np.concatenate((np.eye(dim_z), np.zeros((dim_z, dim_x - dim_z))), axis=1).astype(np.float)

    def __init__(self, pose, obj_type, dataset):
        """
        Args:
            pose (np.ndarray): [4, 1] [x, y, z, theta] - pose of 3D box in global frame
            obj_type (str): type of box
            dataset (str): name of dataset
        """
        # state vector
        self.x = np.vstack((pose, np.zeros((self.num_velocities, 1))))  # [x, y, z, yaw, long_velo, z_dot, yaw_dot]
        # init value of state covariance matrix
        nusc_name = get_nuscenes_name(obj_type, dataset)
        self.P = np.diag([Cov.P[nusc_name]['x'], Cov.P[nusc_name]['y'], Cov.P[nusc_name]['z'], Cov.P[nusc_name]['yaw'],
                          unmeasurable_state_cov, unmeasurable_state_cov, unmeasurable_state_cov])
        # process noise (i.e motion model noise)
        self.Q = np.diag([Cov.Q[nusc_name]['x'], Cov.Q[nusc_name]['y'], Cov.Q[nusc_name]['z'], Cov.Q[nusc_name]['yaw'],
                          2*Cov.Q[nusc_name]['dx'], Cov.Q[nusc_name]['dz'], Cov.Q[nusc_name]['dyaw']])

    def predict(self):
        """ Perform prediction using Constant Turning Rate & Velocity model """
        theta, v, dz, dtheta = self.x[yaw_idx:, 0]

        #
        # compute new state
        #
        if not is_null(dtheta):
            delta_x = np.array([
                (v / dtheta) * (np.sin(theta + dtheta) - np.sin(theta)),
                (v / dtheta) * (-np.cos(theta + dtheta) + np.cos(theta)),
                dz,
                dtheta,
                0.,
                0.,
                0.
            ]).reshape(self.dim_x, 1)
        else:
            delta_x = np.array([
                v * np.cos(theta),
                v * np.sin(theta),
                dz,
                dtheta,
                0.,
                0.,
                0.
            ]).reshape(self.dim_x, 1)
        self.x += delta_x
        # normalize yaw angle
        self.x[yaw_idx, 0] = put_angle_in_range(self.x[yaw_idx, 0])

        #
        # compute new covariance matrix
        #
        F = np.eye(self.dim_x, dtype=np.float)  # linearized motion model
        if not is_null(dtheta):
            F[0, 3] = v * (-np.cos(theta) + np.cos(dtheta + theta)) / dtheta
            F[0, 4] = (-np.sin(theta) + np.sin(dtheta + theta)) / dtheta
            F[0, 6] = v * np.cos(dtheta + theta) / dtheta - v * (-np.sin(theta) + np.sin(dtheta + theta)) / dtheta ** 2

            F[1, 3] = v * (-np.sin(theta) + np.sin(dtheta + theta)) / dtheta
            F[1, 4] = (np.cos(theta) - np.cos(dtheta + theta)) / dtheta
            F[1, 6] = v * np.sin(dtheta + theta) / dtheta - v * (np.cos(theta) - np.cos(dtheta + theta)) / dtheta ** 2

            F[2, 5] = 1.0
            F[3, 6] = 1.0
        else:
            F[0, 3] = -v * np.sin(theta)
            F[0, 4] = np.cos(theta)

            F[1, 3] = v * np.cos(theta)
            F[1, 4] = np.sin(theta)

            F[2, 5] = 1.0
            F[2, 6] = 1.0

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, R):
        """ Perform KF update

        Args:
            z (np.ndarray): measurement vector, shape (dim_z, 1)
            R (np.ndarray): covariance of measurement vector, shape (dim_z, dim_z)
        """
        # kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        # update state vector
        self.x = self.x + K @ (z - self.H @ self.x)
        # normalize yaw angle
        self.x[yaw_idx, 0] = put_angle_in_range(self.x[yaw_idx, 0])
        # update covariance matrix
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P


class ConstantVelocity(object):
    """ Motion model for Pedestrian """
    dim_x = GlobalConfig.cv_dim_x
    num_velocities = dim_x - dim_z
    # motion model
    F = np.eye(dim_x, dtype=np.float)
    F[:num_velocities, -num_velocities:] = np.eye(num_velocities)
    # measurement model
    H = np.concatenate((np.eye(dim_z), np.zeros((dim_z, num_velocities))), axis=1).astype(np.float)

    def __init__(self, pose, obj_type, dataset):
        """
        Args:
            pose (np.ndarray): pose of 3D box in global frame (x, y, z, theta), shape (4, 1)
            obj_type (str): type of box
            dataset (str): name of dataset
        """
        # state vector
        self.x = np.vstack((pose, np.zeros((self.num_velocities, 1))))  # [x, y, z, yaw, x_dot, y_dot, z_dot, yaw_dot]
        # init value of state covariance matrix
        nusc_name = get_nuscenes_name(obj_type, dataset)
        self.P = np.diag([Cov.P[nusc_name]['x'], Cov.P[nusc_name]['y'], Cov.P[nusc_name]['z'], Cov.P[nusc_name]['yaw'],
                          unmeasurable_state_cov, unmeasurable_state_cov, unmeasurable_state_cov, unmeasurable_state_cov])
        # process noise (i.e motion model noise)
        self.Q = np.diag([Cov.Q[nusc_name]['x'], Cov.Q[nusc_name]['y'], Cov.Q[nusc_name]['z'], Cov.Q[nusc_name]['yaw'],
                          Cov.Q[nusc_name]['dx'], Cov.Q[nusc_name]['dy'], Cov.Q[nusc_name]['dz'], Cov.Q[nusc_name]['dyaw']])

    def predict(self):
        self.x = self.F @ self.x
        # normalize yaw angle
        self.x[yaw_idx, 0] = put_angle_in_range(self.x[yaw_idx, 0])
        # compute cov
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R):
        """ Perform KF update

        Args:
            z (np.ndarray): measurement vector, shape (dim_z, 1)
            R (np.ndarray): covariance of measurement vector, shape (dim_z, dim_z)
        """
        # kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        # correct state
        self.x = self.x + K @ (z - self.H @ self.x)
        # normalize yaw angle
        self.x[yaw_idx, 0] = put_angle_in_range(self.x[yaw_idx, 0])
        # correct cov
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P


class KalmanFilter(object):
    def __init__(self, pose, obj_type, dataset):
        """
        Args:
            pose (np.ndarray): pose of 3D box in global frame (x, y, z, theta), shape (4, 1)
            obj_type (str): type of box
            dataset (str): name of dataset
        """
        nusc_name = get_nuscenes_name(obj_type, dataset)
        if nusc_name == 'pedestrian':
            self._state = ConstantVelocity(pose, obj_type, dataset)
            self._model = 'cv'
        else:
            self._state = ConstantTurningRateVelocity(pose, obj_type, dataset)
            self._model = 'ctrv'

    def predict(self):
        """ Perform prediction according motion model """
        self._state.predict()

    def update(self, z: np.ndarray, R: np.ndarray):
        """ Perform KF update

        Args:
            z (np.ndarray): measurement vector, shape (dim_z, 1)
            R (np.ndarray): covariance of measurement vector, shape (dim_z, dim_z)
        """
        self._state.update(z, R)

    def set_yaw_(self, new_yaw: float):
        """ Set KF state vector's yaw angle """
        self._state.x[yaw_idx, 0] = put_angle_in_range(new_yaw)

    def set_velocity_(self, new_velocity: np.ndarray):
        """ Set KF state vector's velocity

        :param new_velocity: [num_velocities] is an array (not a vector)
        """
        self._state.x[-self.num_velocities:, 0] = new_velocity  # new_velocity is an array (not a vector)

    @property
    def x(self) -> np.ndarray:
        return self._state.x

    @property
    def P(self) -> np.ndarray:
        return self._state.P

    @property
    def model(self) -> str:
        return self._model

    @property
    def num_velocities(self) -> int:
        return self._state.num_velocities

    @property
    def H(self) -> np.ndarray:
        """ Get measurement model """
        return self._state.H
