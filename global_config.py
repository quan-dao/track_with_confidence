
kitti_object_name = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
kitti_to_nuscenes = {'Pedestrian': 'pedestrian', 'Car': 'car', 'Cyclist': 'bicycle'}
nuscenes_tracking_names = ('bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck')
waymo_to_nuscenes = {'VEHICLE': 'car', 'PEDESTRIAN': 'pedestrian', 'CYCLIST': 'bicycle'}


def get_nuscenes_name(name, dataset):
    """Convert object name from a dataset to NuScenes

    Args:
        name (str): object name
        dataset (str): dataset name

    Returns:
        str: NuScenes name
    """
    if dataset == 'kitti':
        return kitti_to_nuscenes[name]
    elif dataset == 'waymo':
        return waymo_to_nuscenes[name]
    elif dataset == 'nuscenes':
        return name
    else:
        raise ValueError("dataset has to in ['kitti', 'waymo', 'nuscenes']")


class GlobalConfig:
    """Store all hyper parameters of track-with-confidence """
    inf = 1e5  # a very big number to simulate infinity
    dataset = 'nuscenes'

    '''
    Dataset Path
    '''
    kitti_tracking_root = '/home/user/dataset/kitti/tracking'
    nuscenes_root_mini = '/home/user/dataset/nuscenes/v1.0-mini'
    nuscenes_root_trainval = '/home/user/dataset/nuscenes/v1.0-trainval'
    nuscenes_unpack_detection_dir = './data/nuscenes/megvii_detection'


    '''
    Kalman Filter Parameters
    '''
    dim_z = 4  # [x, y, z, yaw]
    yaw_index = 3  # index of yaw in z and state vector
    kf_unmeasurable__covariance = 1e1  # initial covariance of states that are unmeasurable (e.g. velocity)
    ctrv_dim_x = 7  # [x, y, z, yaw, longitudal_velocity, z_dot, yaw_dot]
    cv_dim_x = 8  # [x, y, z, yaw, x_dot, y_dot, z_dot, yaw_dot]

    '''
    Tracklet Parameters
    '''
    if dataset == 'nuscenes':
        tracklet_num_previous_sizes = 5  # for State.__update_size
        tracklet_confidence_threshold = 0.45  # to determine a tracklet is high or low confident
        tracklet_beta = 1.35  # for computing tracklet confidence (best: 5.35)
        # tuning
        tracklet_tuning_log_likelihood_threshold = -4.5  # (best: -6.5)
        tracklet_tuning_global_assoc_termination_constance = 0.05  # (best: 0.5)
        nuscenes_tracklet_report_conf_threshold = {
            'bicycle': 0.225,
            'bus': 0.,
            'car': 0.1,
            'motorcycle': 0.,
            'pedestrian': 0.1,
            'trailer': 0.,
            'truck': 0.1
        }

    elif dataset == 'kitti':
        tracklet_num_previous_sizes = 15  # for State.__update_size
        tracklet_confidence_threshold = 0.45  # to determine a tracklet is high or low confident
        tracklet_beta = 5.35  # for computing tracklet confidence (best: 5.35)
        # tuning
        tracklet_tuning_log_likelihood_threshold = -6.5  # (best: -6.5)
        tracklet_tuning_global_assoc_termination_constance = 0.6  # (best: 0.5)

    elif dataset == 'waymo':
        tracklet_num_previous_sizes = 5  # for State.__update_size
        tracklet_confidence_threshold = 0.45  # to determine a tracklet is high or low confident
        tracklet_beta = 5.35  # for computing tracklet confidence (best: 5.35)
        # tuning
        tracklet_tuning_log_likelihood_threshold = -1.5
        tracklet_tuning_global_assoc_termination_constance = 1.0
        tracklet_report_conf_threshold = {
            'VEHICLE': 0.,
            'PEDESTRIAN': 0.,
            'CYCLIST': 0.0
        }
        nbr_terminals = 4
    else:
        raise ValueError("{} is not supported".format(dataset))
