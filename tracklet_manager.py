import numpy as np

from tracklet import Tracklet
from measurement import Measurement
from global_config import GlobalConfig
from utils.data_association import greedy_matching as lap_solver


# global constants (their value are set in global_config.py)
inf = GlobalConfig.inf
conf_thres = GlobalConfig.tracklet_confidence_threshold
log_aff_thres = GlobalConfig.tracklet_tuning_log_likelihood_threshold
termination_const = GlobalConfig.tracklet_tuning_global_assoc_termination_constance


class TrackletManager(object):
    """ To carry out 2-step data association """

    def __init__(self, obj_type: str):
        self.all_tracklets = []
        self._high_conf_tracklets_ids = []  # index of high conf tracklets
        self._low_conf_tracklets_ids = []  # index of low conf tracklets
        self.obj_type = obj_type

    def __repr__(self):
        return 'TrackletManager | num_tracklets: {}, tracklets: {}'.format(
            len(self.all_tracklets), self.all_tracklets
        )

    def separate_by_confidence_(self):
        """ Separate all tracklets into set of high confidence tracklets & low confidence tracklets """
        self._high_conf_tracklets_ids, self._low_conf_tracklets_ids = [], []
        for i, tracklet in enumerate(self.all_tracklets):
            if tracklet.conf > conf_thres:
                self._high_conf_tracklets_ids.append(i)
            else:
                self._low_conf_tracklets_ids.append(i)

    def local_association_(self, all_measurements):
        """ Perform local association for tracklets with high confidence,
        once associated tracklets are updated accordingly

        Args:
            all_measurements (list[Measurement]): all measurements at this time step
        Returns:
            list[int]: indexes of measurements that are unassociated
        """
        if not self._high_conf_tracklets_ids:
            # there are now high confidence tracklet, go straight to global association
            unassoc_measurements = [i for i in range(len(all_measurements))]
            return unassoc_measurements

        # extract all confidence tracklet
        ids_conf_tracklets = self._high_conf_tracklets_ids

        # construct cost matrix
        cost_matrix = np.zeros((len(ids_conf_tracklets), len(all_measurements)))
        for i in range(len(ids_conf_tracklets)):
            tracklet = self.all_tracklets[ids_conf_tracklets[i]]
            for j in range(len(all_measurements)):
                cost_matrix[i, j] = -Tracklet.compute_affinity(tracklet, all_measurements[j])

        # solve linear assignment
        assoc_pairs, unassoc_ids_conf, unassoc_measurements = lap_solver(cost_matrix, -log_aff_thres)
        unpacked_assoc_pairs = [(ids_conf_tracklets[i], j) for i, j in assoc_pairs]

        # update for high conf tracklets which associated with a measurement
        for ids_high_conf, j in unpacked_assoc_pairs:
            self.all_tracklets[ids_high_conf].extend_(all_measurements[j])

        return unassoc_measurements

    def global_association_(self, unassoc_measurements):
        """ Perform global association for tracklet with low confidence. Once associated low conf tracklets are updated
        or merge with high confidence tracklets accordingly.

        Args:
            unassoc_measurements (list[Measurement]): measurements left over after local association
            list[int]: indexes of measurements that are unassociated, these are used to init new tracklet
        """
        num_low, num_high = len(self._low_conf_tracklets_ids), len(self._high_conf_tracklets_ids)
        num_meas = len(unassoc_measurements)

        # create cost matrix for event A: low conf tracklets are associated with high confidence tracklets
        mat_A = np.zeros((num_low, num_high))
        for i in range(num_low):
            tracklet_low = self.all_tracklets[self._low_conf_tracklets_ids[i]]
            for j in range(num_high):
                tracklet_high = self.all_tracklets[self._high_conf_tracklets_ids[j]]
                mat_A[i, j] = -Tracklet.compute_affinity(tracklet_low, tracklet_high)

        # create cost matrix for event B: low conf tracklets are terminated
        mat_B = np.zeros((num_low, num_low)) + inf
        for i in range(num_low):
            tracklet_low = self.all_tracklets[self._low_conf_tracklets_ids[i]]
            mat_B[i, i] = -np.log(max(termination_const - tracklet_low.conf, 1e-20))

        # create cost matrix for event C: measurements are associate with low conf tracklets
        mat_C = np.zeros((num_meas, num_low))
        for i in range(num_meas):
            for j in range(num_low):
                tracklet_low = self.all_tracklets[self._low_conf_tracklets_ids[j]]
                mat_C[i, j] = -Tracklet.compute_affinity(tracklet_low, unassoc_measurements[i])

        # create dummy block
        mat_dummy = np.zeros((num_meas, num_high)) + inf

        # assemble all into global cost matrix
        global_cost = np.zeros((num_low + num_meas, num_high + num_low))
        global_cost[:num_low, :num_high] = mat_A
        global_cost[:num_low, num_high:] = mat_B
        global_cost[num_low:, :num_high] = mat_dummy
        global_cost[num_low:, num_high:] = mat_C

        # solve LAP & interpret result
        assoc_pairs, unassoc_rows, unassoc_cols = lap_solver(global_cost, -log_aff_thres)

        for i, j in assoc_pairs:
            if i < num_low:
                # i points to low confidence tracklet
                tracklet_low = self.all_tracklets[self._low_conf_tracklets_ids[i]]
                if j < num_high:
                    # event A: tracklet low assoc with a tracklet high
                    tracklet_high = self.all_tracklets[self._high_conf_tracklets_ids[j]]
                    tracklet_high.merge_(tracklet_low)
                    # terminate low confidence tracklet after it is merged with high confidence one
                    tracklet_low.is_terminated = True
                else:
                    if j - num_high == i:
                        # event B: termination of a low confidence tracklet
                        # if tracklet_low.conf < 1e-2:
                        tracklet_low.is_terminated = True
            else:
                # i points to a measurement
                if j < num_high:
                    pass
                    # if TrackletManager.debug_mode:
                    #     print('[WARN] @TrackletManager/ global_association_ | in mat_dummy, a nonsense association')
                else:
                    # event C: measurement is associated with a low conf tracklet
                    meas = unassoc_measurements[i - num_low]
                    tracklet_low = self.all_tracklets[self._low_conf_tracklets_ids[j - num_high]]
                    tracklet_low.extend_(meas)

        # collect unassociated measurements to initiate new tracks
        leftover_measurements_ids = [i - num_low for i in unassoc_rows if i >= num_low]
        return leftover_measurements_ids

    def remove_terminated_tracklets_(self):
        to_remove_ids = [i for i, tracklet in enumerate(self.all_tracklets) if tracklet.is_terminated]
        for i in reversed(to_remove_ids):
            del self.all_tracklets[i]

    def update_tracklet_confidence_(self, current_timestamp):
        """ Recalculate confidence for every tracklet after 2-step data association

        Args:
            current_timestamp (int)
        """
        for tracklet in self.all_tracklets:
            assert not tracklet.is_terminated, 'tracklet {} is terminated and need to be removed'.format(tracklet.id)
            tracklet.compute_confidence_(current_timestamp)

    def run_(self, all_measurements, timestamp):
        """ Execute 1 iteration of track-with-confidence

        Args:
            all_measurements (list[Measurement]): all measurements at this time step
            timestamp (int): timestamp of current time step
        """
        if not self.all_tracklets:
            for meas in all_measurements:
                self.all_tracklets.append(Tracklet(meas))
        else:
            # tracklets had been init
            # category tracklets to high confidence & low confidence
            self.separate_by_confidence_()

            # local association for high confidence tracklet
            unassoc_meas_indicies = self.local_association_(all_measurements)

            # global association for low confidence tracklet & unassoc measurements
            unassoc_measurements = [all_measurements[j] for j in unassoc_meas_indicies]
            leftover_meas_indicies = self.global_association_(unassoc_measurements)

            # remove terminated tracklets
            self.remove_terminated_tracklets_()

            # recalculate tracklet confidence
            self.update_tracklet_confidence_(timestamp)

            # create new tracklets from measurements left over after global association
            for j in leftover_meas_indicies:
                new_tracklet = Tracklet(unassoc_measurements[j])
                self.all_tracklets.append(new_tracklet)
