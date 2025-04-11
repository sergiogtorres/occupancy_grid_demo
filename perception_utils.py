import numpy as np
import warnings
from scipy.special import logit
import cv2
import utils

DETECTION_MODE = 0
INVERSE_MEASUREMENT_MODE = 1

def draw_perception_line(im, start, range, bearing, dphi, dr, pixels_to_a_meter):
    """
    TODO: adjust to detected range?
    TODO: check correct usage of (x, y) and not (y, x)
    :param im:
    :param start:
    :param range:
    :param bearing:
    :param dphi:
    :return:
    """
    range_in_pixels = range*pixels_to_a_meter
    dr_in_pixels = dr*pixels_to_a_meter
    pt1 = tuple(start)
    bear_plus = -(bearing+dphi)  # negative sign to correct for computer vision convention of y positive downwards
    bear_minus = -(bearing-dphi)
    delta_a = range_in_pixels*np.array([np.cos(bear_plus), np.sin(bear_plus)])
    delta_b = range_in_pixels*np.array([np.cos(bear_minus), np.sin(bear_minus)])

    delta_a_3 = dr_in_pixels*np.array([np.cos(bear_plus), np.sin(bear_plus)])
    delta_b_3 = dr_in_pixels*np.array([np.cos(bear_minus), np.sin(bear_minus)])

    x2_a, y2_a = (start + delta_a).astype(np.uint32)
    x2_b, y2_b = (start + delta_b).astype(np.uint32)

    x3_a, y3_a = (start + delta_a + delta_a_3).astype(np.uint32)
    x3_b, y3_b = (start + delta_b + delta_b_3).astype(np.uint32)

    pt2_a = x2_a, y2_a
    pt2_b = x2_b, y2_b

    pt3_a = x3_a, y3_a
    pt3_b = x3_b, y3_b


    #print(f"drawing line from {pt1} to {pt2_a} and {pt2_b}")
    cv2.line(im, pt1, pt2_a, (0,255,0), 1)
    cv2.line(im, pt1, pt2_b, (0,255,0), 1)

    cv2.line(im, pt2_a, pt3_a, (0,0,255), 1)
    cv2.line(im, pt2_b, pt3_b, (0,0,255), 1)

    cv2.putText(im, f"bearing:{np.round(bearing, 2)}, "
                    f"range:{np.round(range, 2)} m, {np.round(range_in_pixels, 0)} px"
                    f"start:{start}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)

    cv2.putText(im, f"delta_a:{np.round(delta_a, 2)}, ",
                (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(im, f"delta_b:{np.round(delta_b, 2)}, ",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(im, f"pt1:{pt1}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)
    cv2.putText(im, f"pt2_a:{pt2_a}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)
    cv2.putText(im, f"pt2_b:{pt2_b}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)
def check_within_range_bearing(arr_range, arr_bearing,
                               current_range, current_bearing, dr, dphi,
                               ground_truth_map=None,
                               mode=DETECTION_MODE,
                               frame_to_debug = None):
    """
    if ground_truth_map is given, range_down_up, bearing_down_up are teh upper and lower ranges
    if ground_truth_map is None:
        checked ranges:  [0, current detection range+dr]
        checked bearings: [current bearing-dphi, current bearing+dphi]

    This function takes in range in meters, since it works with the arr_range array.
    TODO: this function does two quite different things:
        1. simulate the LiDAR detection if mode=DETECTION_MODE
        2. helper function for inverse_measurement_mode
        --> separate these into two functions for clarity


    mask_ranges_and_bearings: contains True for any position within the ranges range_down_up, bearing_down_up
    :param arr_range: relative range of each point in meters
    :param arr_bearing: relative bearing of each point in radians
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
    #bearing_down_up = [current_bearing-dphi/2, current_bearing+dphi/2]

    mask_bearings = utils.angles_in_range(arr_bearing, current_bearing, dphi/2)
    #print(f"calling angles_in_range(arr, {np.round(current_bearing, 2)}, {np.round(dphi/2, 2)}")
    #mask_bearings = bearing_down_up[0] <= arr_bearing
    #mask_bearings *= arr_bearing <= bearing_down_up[1]

    # TODO: mask_ranges seems to have some holes (small but annoying). Why? Fix.
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

    #if np.any(mask_obstacles):
        #if mode == DETECTION_MODE:
            #print(f"obstacle detected @ {np.where(mask_obstacles)}")
    if frame_to_debug is not None:
        frame_to_debug[mask_ranges] += np.array([0, 0, 255]).astype(np.uint8)
        frame_to_debug[mask_bearings] += np.array([255, 0, 0]).astype(np.uint8)
        frame_to_debug[mask_obstacles] += np.array([0, 255, 0]).astype(np.uint8)
        #cv2.imshow("frame_to_debug", frame_to_debug)
        #cv2.waitKey(1)

    return mask_obstacles, mask_no_obstacles, mask_no_info
def inverse_measurement_model(ran: float, bearing: float, car) -> float:

    """
    returns p(m^i|y_t)
    :param ran:
    :param bearing:
    :param car:
    :return:
    """
    # TODO: implement with Bresenham's line algorithm for Ray Tracing
    # TODO: refine implementation. All possible points are assigned 1... do a gradient instead ?

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