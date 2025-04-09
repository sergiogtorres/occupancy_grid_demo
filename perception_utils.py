import numpy as np
import warnings
from scipy.special import logit

DETECTION_MODE = 0
INVERSE_MEASUREMENT_MODE = 1

def check_within_range_bearing(arr_range, arr_bearing,
                               current_range, current_bearing, dr, dphi,
                               ground_truth_map=None,
                               mode=DETECTION_MODE):
    """
    if ground_truth_map is given, range_down_up, bearing_down_up are teh upper and lower ranges
    if ground_truth_map is None:
        checked ranges:  [0, current detection range+dr]
        checked bearings: [current bearing-dphi, current bearing+dphi]

    TODO: this function does two quite different things:
        1. simulate the LiDAR detection if mode=DETECTION_MODE
        2. helper function for inverse_measurement_mode
        --> separate these into two functions for clarity

    mask_ranges_and_bearings: contains True for any position within the ranges range_down_up, bearing_down_up
    :param arr_range:
    :param arr_bearing:
    :param range_down_up:
    :param bearing_down_up:
    :param ground_truth_map:
    :return:
        mask_obstacles: contains True for any position where an obstacle is detected (at the end of the ray)
        mask_no_obstacles: contains True for any position where no obstacle is detected (along the ray)
                           None if ground_truth_map is given (i.e., working in detection mode)
        mask_no_info: contains True if no info obtained for each point point
    """
    range_down_up = [current_range-dr/2, current_range+dr/2]
    bearing_down_up = [current_bearing-dphi/2, current_bearing+dphi/2]

    mask_bearings = bearing_down_up[0] <= arr_bearing
    mask_bearings *= arr_bearing <= bearing_down_up[1]

    mask_ranges = range_down_up[0] <= arr_range
    mask_ranges *= arr_range <= range_down_up[1]
    mask_ranges_and_bearings = mask_ranges * mask_bearings
                                                # now, masks simply contains points within
                                                # the annular portion defined by the ranges

    # if in measurement mode, check if, in those indices, there is an obstacle in the gt map
    if ground_truth_map is not None:
        mask_obstacles = mask_ranges_and_bearings * ground_truth_map
        mask_no_obstacles = None
        mask_no_info = None

    else:                                           # no gt map here.

        mask_obstacles = mask_ranges_and_bearings   # we are considering any point within the ranges as a detection

        mask_no_obstacles = (np.bitwise_not(mask_obstacles)) \
                            & (arr_range < range_down_up[1]) \
                            & mask_bearings                     # not an obstacle,
                                                                # closer than the measured distance,
                                                                # and within the bearing range
        mask_no_info = np.bitwise_not(mask_obstacles) & np.bitwise_not(mask_no_obstacles)

    if (mode == DETECTION_MODE and ground_truth_map is None) \
        or (mode == INVERSE_MEASUREMENT_MODE and ground_truth_map is not None):
        warnings.warn("Wrong combination of mode and ground_truth_map in check_within_range_bearing!")

    return mask_obstacles, mask_no_obstacles, mask_no_info
def inverse_measurement_model(ran: float, bearing: float, car) -> float:

    """
    returns p(m^i|y_t)
    TODO: implement with Bresenham's line algorithm for Ray Tracing
    :param ran:
    :param bearing:
    :param car:
    :return:
    """
    mask_obstacles, mask_no_obstacles, mask_no_info \
        = check_within_range_bearing(car.ground_truth_map_ran, car.ground_truth_map_bearing,
                                     ran, bearing, car.lidar_dr, car.lidar_dphi,
                                     ground_truth_map=None,
                                     mode = INVERSE_MEASUREMENT_MODE)

    m_t = mask_obstacles*1 + mask_no_obstacles*0 + mask_no_info*0.5
    return m_t

def update_belief_map(m_t, l_prev, l_0=0.5):

    l_t = logit(m_t) + l_prev - l_0

    return l_t