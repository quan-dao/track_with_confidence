import numpy as np
from copy import deepcopy

from kalman_filter import KalmanFilter
from global_config import GlobalConfig
from measurement import Measurement
from utils.geometry import put_angle_in_range
from utils.stats import pseudo_log_likelihood
from utils.data_classes import Bbox3D


# global constants (their value are set in global_config.py)
yaw_idx = GlobalConfig.yaw_index


class State(object):
    """ Represent state of an object at a point of time """

    def __init__(self, meas):
        """
        Args:
            meas (Measurement): a measurement whose measurement vector is expressed in global frame
        """
        self._kf = KalmanFilter(meas.z, meas.obj_type)
        self.size = meas.size  # [3] (length, width, height)
        self.stamp = meas.stamp

    @property
    def velocity(self) -> np.ndarray:
        return self._kf.x[-self._kf.num_velocities:, 0]

    @property
    def measurement_model(self) -> np.ndarray:
        return self._kf.H

    @property
    def x(self) -> np.ndarray:
        return self._kf.x

    @property
    def P(self) -> np.ndarray:
        return self._kf.P

    def predict(self, num_step, velocity=np.array([]), forward=True):
        """ Perform prediction step of KF

        Args:
            num_step (int): number of time step to predict
            velocity (np.ndarray): assigned velocity for prediction
            forward (bool): true if the prediction is forward in time
        Returns:
            State: predicted state
        """
        state = deepcopy(self)
        # assign timestamp for predicted state
        if forward:
            state.stamp += num_step
        else:
            state.stamp -= num_step
        # set velocity (if necessary)
        if velocity.size > 0:
            state._kf.set_velocity_(velocity)
        # predict
        for i in range(num_step):
            state._kf.predict()
        return state

    def __update_size_(self, measured_size, avg_previous_sizes, num_previous_sizes):
        """ Update size by averaging measurement's size with previous states' size

        Args:
            measured_size (np.ndarray): size of the detected 3D box, shape (3, )
            avg_previous_sizes (np.ndarray): average size of previous states, shape (3, )
            num_previous_sizes (int): number of previous sizes used to calculate avg_previous_sizes
        """
        self.size = (measured_size + num_previous_sizes * avg_previous_sizes) / (1 + num_previous_sizes)

    def __update_state_(self, meas):
        """ Perform prediction step (this time predict mutates state vector), then update step of KF

        Args:
            meas (Measurement): a measurement
        """
        time_gap = meas.stamp - self.stamp
        assert time_gap >= 0, 'Incompatible time. measurement have to be at least as recent as associated state'
        for i in range(time_gap):
            self._kf.predict()

        # angle correction
        angle_diff = np.abs(meas.z[yaw_idx, 0] - self._kf.x[yaw_idx, 0])
        if np.pi / 2.0 <= angle_diff <= 3.0 * np.pi / 2.0:
            # flip prediction angle
            self._kf.set_yaw_(self._kf.x[yaw_idx, 0] + np.pi)
        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if np.abs(meas.z[yaw_idx, 0] - self._kf.x[yaw_idx, 0]) >= 3 * np.pi / 2.0:
            if meas.z[yaw_idx, 0] > 0:
                self._kf.set_yaw_(self._kf.x[yaw_idx, 0] + np.pi * 2)
            else:
                self._kf.set_yaw_(self._kf.x[yaw_idx, 0] - np.pi * 2)

        # update KF state
        self._kf.update(meas.z, meas.R)

    def update_(self, meas, avg_previous_sizes, num_previous_sizes):
        """ Update this state with a measurement

        Args:
            meas (Measurement): a measurement
            avg_previous_sizes (np.ndarray): average size of previous states, shape (3, )
            num_previous_sizes (int): number of previous sizes used to calculate avg_previous_sizes
        """
        self.__update_size_(meas.size, avg_previous_sizes, num_previous_sizes)
        self.__update_state_(meas)
        # update state's stamp
        self.stamp = meas.stamp

    def log_likelihood(self, meas):
        """ Compute log likelihood of a measurement conditioned on a state

        Args:
            meas (Measurement): a measurement
        Returns:
            float: log likelihood
        """
        z_expected = self._kf.H @ self._kf.x
        S = self._kf.H @ self._kf.P @ self._kf.H.T + meas.R
        S = (S + S.T) / 2.0

        # angle correction
        angle_diff = np.abs(meas.z[yaw_idx, 0] - z_expected[yaw_idx, 0])
        if np.pi / 2.0 <= angle_diff <= 3.0 * np.pi / 2.0:
            # flip prediction angle
            z_expected[yaw_idx, 0] = put_angle_in_range(z_expected[yaw_idx, 0] + np.pi)

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if np.abs(meas.z[yaw_idx, 0] - z_expected[yaw_idx, 0]) >= 3 * np.pi / 2.0:
            if meas.z[yaw_idx, 0] > 0:
                z_expected[yaw_idx, 0] += np.pi * 2
            else:
                z_expected[yaw_idx, 0] -= np.pi * 2

        return pseudo_log_likelihood(meas.z, mean=z_expected, cov=S)


def cvt_state_to_bbox3d(state, box_id=None, score=None, alpha=None):
    """Convert a state to a 3D bounding box

    Args:
        state (State): a state
        box_id (int or str): id of tracklet where state is from
        score (float): detection score (to format tracking result)
        alpha (float): camear observation angle
    Returns:
        Bbox3D: a 3D bounding box
    """
    box = Bbox3D(state.x[0, 0], state.x[1, 0], state.x[2, 0], state.size[0], state.size[1], state.size[2], state.x[3, 0])
    if box_id is not None:
        if not isinstance(box_id, str):
            box_id = str(box_id)
        box.id = box_id
    if score is not None:
        box.score = score
    if alpha is not None:
        box.cam_obs_angle = alpha
    return box

