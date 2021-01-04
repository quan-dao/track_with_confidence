
kitti_object_name = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
kitti_to_nuscenes = {'Pedestrian': 'pedestrian', 'Car': 'car', 'Cyclist': 'bicycle'}


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
        raise ValueError('waymo is not supported yet')
    else:
        raise ValueError("dataset has to in ['kitti', 'waymo']")


class GlobalConfig:
    """Store all hyper parameters of track-with-confidence """
    inf = 1e5  # a very big number to simulate infinity
    dataset = 'kitti'

    '''
    Kalman Filter Parameters
    '''
    dim_z = 4  # [x, y, z, yaw]
    yaw_index = 3  # index of yaw in z and state vector
    kf_unmeasurable__covariance = 1e2  # initial covariance of states that are unmeasurable (e.g. velocity)
    ctrv_dim_x = 7  # [x, y, z, yaw, longitudal_velocity, z_dot, yaw_dot]
    cv_dim_x = 8  # [x, y, z, yaw, x_dot, y_dot, z_dot, yaw_dot]

    '''
    Tracklet Parameters
    '''
    tracklet_num_previous_sizes = 15  # for State.__update_size
    tracklet_confidence_threshold = 0.45  # to determine a tracklet is high or low confident
    tracklet_beta = 5.35  # for computing tracklet confidence (best: 5.35)
    # tuning
    tracklet_tuning_log_likelihood_threshold = -6.5  # (best: -6.5)
    tracklet_tuning_global_assoc_termination_constance = 0.6  # (best: 0.5)
