import numpy as np

from global_config import GlobalConfig
from measurement import Measurement
from state import State
from utils.stats import pseudo_log_likelihood


# global constants (their value are set in global_config.py)
inf = GlobalConfig.inf
log_aff_thres = GlobalConfig.tracklet_tuning_log_likelihood_threshold
conf_thres = GlobalConfig.tracklet_confidence_threshold
dim_z = GlobalConfig.dim_z
num_prev_sizes = GlobalConfig.tracklet_num_previous_sizes
dataset = GlobalConfig.dataset


class Tracklet(object):
    """ Represent a tracklet """
    count = 1 if dataset == 'kitti' else 0  # to count number of instances of this class, kitti start counting from 1
    beta = GlobalConfig.tracklet_beta

    def __init__(self, meas):
        """
        Args:
            meas (Measurement): the measurement that init this tracklet
        """
        # get tracklet ID
        self.id = Tracklet.count
        Tracklet.count += 1
        # init tracklet's history
        head = State(meas)  # first state of this tracklet
        self.history = [head]
        # init tracklet confidence
        self.conf = 0.9 * conf_thres
        self.sum_affinity = 0  # to compute tracklet confidence
        self.is_terminated = False  # True is tracklet is terminated
        self.obj_type = meas.obj_type
        self.most_recent_meas_score = meas.score  # to create tracking result for NuScenes
        self.kitti_meas_alpha = meas.kitti_alpha  # to create tracking result for KITTI
        self.just_born = True  # to prevent new born tracklet from being reported

    def __repr__(self):
        return 'Tracklet | id: {},\t conf: {:.4f},\t just_born: {}, \t terminated: {},\t obj_type: {}'.format(
            self.id, self.conf, self.just_born, self.is_terminated, self.obj_type
        )

    @property
    def head(self) -> State:
        """ Return the first state of this tracklet """
        return self.history[0]

    @property
    def tail(self) -> State:
        """ Return the last state of this tracklet """
        return self.history[-1]

    def compute_confidence_(self, current_timestamp: int) -> None:
        """Compute tracklet's confidence and assign it to 'conf' attribute
        Important: This function is called after extend_ (and merge_ in case of merging with a low confidence tracklet)
        """
        num_missing = current_timestamp - self.head.stamp - len(self.history) + 1
        self.conf = (self.sum_affinity / len(self.history)) * np.exp(-Tracklet.beta * num_missing / len(self.history))

    def compute_avg_previous_sizes(self) -> np.ndarray:
        """ Compute the average size of num_previous_sizes states in this tracklet"""
        avg_sizes = np.zeros(3, dtype=np.float)
        if len(self.history) < num_prev_sizes:
            for state in self.history:
                avg_sizes += state.size
            return avg_sizes / len(self.history)
        else:
            for i in range(-1, -num_prev_sizes - 1, -1):
                avg_sizes += self.history[i].size
            return avg_sizes / num_prev_sizes

    def extend_(self, meas: Measurement) -> None:
        """ Extend tracklet's history by updating its tail with a measurement

        :param meas: the measurement associated with this tracklet
        """
        assert meas.obj_type == self.obj_type, 'association has to be made within the same class'
        time_gap = meas.stamp - self.tail.stamp
        assert time_gap > 0, 'Incompatible time stamp, measurement has to be more recent the tail of a tracklet'

        # update tracklet's sum affinity
        meas_affinity = Tracklet.compute_affinity(self, meas)
        assert meas_affinity > -inf, 'an invalid association is used to update tracklet'

        # update tracklet's sum affinity for computing tracklet's confidence
        self.sum_affinity += np.exp(meas_affinity)

        # update tracklet's history with new state resulted from associated measurement
        predicted_tail = self.tail.predict(time_gap)
        predicted_tail.update_(meas, self.compute_avg_previous_sizes(), min([len(self.history), num_prev_sizes]))
        self.history.append(predicted_tail)

        self.most_recent_meas_score = meas.score  # to create tracking result
        self.kitti_meas_alpha = meas.kitti_alpha  # to create tracking result for KITTI
        # to switch off just_born flag
        if self.just_born:
            self.just_born = False

    def is_time_compatible(self, other):
        """ Check if other tracklet is time compatible for concatenating with this tracklet

        Args:
            other (Tracklet): another tracklet
        Returns:
            (bool, bool): (is_compatible, this_ended_first)
        """
        is_compatible = self.tail.stamp <= other.head.stamp or other.tail.stamp <= self.head.stamp
        if is_compatible:
            this_ended_first = self.tail.stamp <= other.head.stamp
            return is_compatible, this_ended_first
        else:
            return False, False

    def merge_(self, other):
        """ Merge this tracklet (high confidence) with a low confident tracklet

        Args:
            other (Tracklet): a low confident tracklet
        """
        assert self.conf > conf_thres >= other.conf, 'merge_ can only be called for a pair of high conf ' \
                                                     'and low conf tracklets'
        # check time compatibility
        is_compatible, this_ended_first = self.is_time_compatible(other)
        if not is_compatible:
            # two tracklets are time incompatible, no merging
            return

        # assert is_compatible, 'other tracklet is not time compatible for concatenating'
        if this_ended_first:
            # concatenate history
            self.history += other.history
        else:
            # other ended first
            # concatenate history
            self.history = other.history + self.history
            # take ID from older tracklet
            self.id = other.id
        # update tracklet's sum affinity
        self.sum_affinity += other.sum_affinity

    def compute_avg_velocity(self):
        """ Compute average velocity of every state in this tracklet's history

        Returns:
            np.ndarray: trackelt's average velocity, shape (3, )
        """
        assert self.history, 'Tracklet does not have any state in its history'
        avg_velocity = np.zeros(self.head.velocity.size)
        for i, state in enumerate(self.history):
            if i == 0:
                # skip the head, since this state's velocity is unreliable (high covariance, cuz don't know)
                continue
            avg_velocity += state.velocity
        # dimension check
        assert len(avg_velocity.shape) == 1, 'avg_velocity must be an array not a column vector'
        return avg_velocity / (len(self.history) - 1) if len(self.history) > 1 else avg_velocity

    def propagate_forward(self, num_step):
        """ Propagate this tracklet's tail forward in time

        Args:
            num_step (int): duration of the propagation
        Returns:
            list[State]: list of predicted states, ordered according to timestamp ascending
        """
        if num_step == 0:
            # no propagation
            return [self.tail]

        avg_velocity = self.compute_avg_velocity()
        predicted_states = []
        state = self.tail
        for i in range(num_step):
            predicted_states.append(state.predict(num_step=1, velocity=avg_velocity))
            state = predicted_states[-1]
        return predicted_states

    def propagate_backward(self, num_step: int):
        """ Propagate this tracklet's head backward in time

        Args:
            num_step (int): duration of the propagation
        Returns:
            list[State]: list of predicted states, ordered according to timestamp ascending
        """
        if num_step == 0:
            # no propagation
            return [self.head]

        avg_velocity_backward = -self.compute_avg_velocity()
        predicted_states = []
        state = self.head
        for i in range(num_step):
            predicted_states.append(state.predict(num_step=1, velocity=avg_velocity_backward, forward=False))
            state = predicted_states[-1]
        # reverse predicted_states to put them in timestamp ascending order
        predicted_states.reverse()
        return predicted_states

    @staticmethod
    def __compute_motion_affinity_ordered(end_first, end_later):
        """  Compute motion affinity for 2 tracklets that are time compatible based on log likelihood of position

        Args:
            end_first (Tracklet): tracklet ends first
            end_later (Tracklet): tracklet ends later
        Returns:
            float: motion affinity
        """
        time_gap = end_later.head.stamp - end_first.tail.stamp
        assert time_gap >= 0, 'Time incompatible, check is_time_compatible function of class Tracklet'
        # propagate end_first forward in time
        forward = end_first.propagate_forward(time_gap)
        # propagate end_later backward in time
        backward = end_later.propagate_backward(time_gap)

        # compute forward log likelihood
        assert forward[-1].stamp == end_later.head.stamp, 'Timestamp mismatch'
        H = end_later.head.measurement_model
        forward_cov = H @ end_later.head.P @ H.T
        forward_likelihood = pseudo_log_likelihood(forward[-1].x[:dim_z], mean=end_later.head.x[:dim_z],
                                                   cov=forward_cov)

        # compute backward log likelihood
        assert backward[0].stamp == end_first.tail.stamp, 'Timestamp mismatch'
        backward_cov = H @ end_first.tail.P @ H.T
        backward_likelihood = pseudo_log_likelihood(backward[0].x[:dim_z], mean=end_first.tail.x[:dim_z],
                                                    cov=backward_cov)

        motion_aff = forward_likelihood + backward_likelihood
        return motion_aff

    @staticmethod
    def compute_motion_affinity(tracklet, other):
        """Compute motion affinity between a tracklet and another tracklet or a measurement

        Args:
            tracklet (Tracklet): tracklet of concern
            other (Tracklet or Measurement): object to compute motion affinity with tracklet of concern
        Returns:
            float: motion affinity
        """
        if type(other) == Tracklet:
            is_compatible, tracklet_ends_first = tracklet.is_time_compatible(other)
            if not is_compatible:
                return -inf

            if tracklet_ends_first:
                return Tracklet.__compute_motion_affinity_ordered(tracklet, other)
            else:
                return Tracklet.__compute_motion_affinity_ordered(other, tracklet)
        else:
            assert type(other) == Measurement, '2nd input must be either Tracklet or Measurement'
            time_gap = other.stamp - tracklet.tail.stamp
            assert time_gap > 0, 'Incompatible time stamp, measurement has to be more recent the tail of a tracklet'
            # propagate tail to the timestamp of measurement
            predicted_tail = tracklet.tail.predict(time_gap)
            assert predicted_tail.stamp == other.stamp, 'Timestamp mismatch'
            # compute measurement likelihood
            meas_loglikelihood = predicted_tail.log_likelihood(other)
            return meas_loglikelihood

    @staticmethod
    def compute_size_affinity(tracklet, other):
        """Compute size affinity between a tracklet and another tracklet or a measurement

        Args:
            tracklet (Tracklet): tracklet of concern
            other (Tracklet or Measurement): object to compute motion affinity with tracklet of concern
        Returns:
            float: motion affinity
        """
        if type(other) == Tracklet:
            is_compatible, tracklet_ends_first = tracklet.is_time_compatible(other)
            if not is_compatible:
                return -inf

            if tracklet_ends_first:
                diff = np.abs(tracklet.tail.size - other.head.size)
                return -np.sum(diff / (tracklet.tail.size + other.head.size))
            else:
                diff = np.abs(other.tail.size - tracklet.head.size)
                return -np.sum(diff / (other.tail.size + tracklet.head.size))
        else:
            assert type(other) == Measurement, '2nd input must be either Tracklet or Measurement'
            diff = np.abs(tracklet.tail.size - other.size)
            return -np.sum(diff / (tracklet.tail.size + other.size))

    @staticmethod
    def compute_affinity(tracklet, other) -> float:
        """Compute affinity between a tracklet and another tracklet or a measurement

        Args:
            tracklet (Tracklet): tracklet of concern
            other (Tracklet or Measurement): object to compute motion affinity with tracklet of concern
        Returns:
            float: motion affinity
        """
        assert tracklet.obj_type == other.obj_type, 'Only consider same class association'
        motion_aff = Tracklet.compute_motion_affinity(tracklet, other)
        size_aff = Tracklet.compute_size_affinity(tracklet, other)
        aff = motion_aff + size_aff
        return aff if aff > log_aff_thres else -inf
