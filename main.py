import numpy as np
import numpy.typing as npt
import cv2
from scipy.special import logit, expit
import time

import car_lib
import perception_utils

# TODO: check all instances of use of position -> x, y correct order? cv2 is x, y!

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    target_fps = 30
    target_frame_time = 1/target_fps
    ### Loading obstacle_map ###
    # Load as grayscale
    img = cv2.imread('map_always_detect.bmp', cv2.IMREAD_GRAYSCALE)
    print(img.shape)  # e.g., (height, width)
    obstacle_map = img == 0
    lidar_rps = 1/10
    #############################
    max_x_meters, max_y_meters = 20, 20 #in meters

    pixels_to_a_meter_x = obstacle_map.shape[1]/max_x_meters
    pixels_to_a_meter_y = obstacle_map.shape[0]/max_y_meters

    norm_difference_ratios = (pixels_to_a_meter_x - pixels_to_a_meter_y)/(pixels_to_a_meter_x + pixels_to_a_meter_y)

    assert norm_difference_ratios < 0.02
    pixels_to_a_meter = (pixels_to_a_meter_x+pixels_to_a_meter_y)/2

    map_center_x_px, map_center_y_px = [obstacle_map.shape[1]//2, obstacle_map.shape[0]//2]
    map_min_x_meters = (0-map_center_x_px)/pixels_to_a_meter
    map_max_x_meters = (obstacle_map.shape[1]-map_center_x_px)/pixels_to_a_meter
    map_min_y_meters = (0-map_center_y_px)/pixels_to_a_meter
    map_max_y_meters = (obstacle_map.shape[0]-map_center_x_px)/pixels_to_a_meter

    map_x_range_meters = np.linspace(map_min_x_meters, map_max_x_meters, obstacle_map.shape[1])
    map_y_range_meters = np.linspace(map_max_y_meters, map_min_y_meters, obstacle_map.shape[0])

    car = car_lib.Car([map_center_x_px, map_center_y_px],
                      map_x_range_meters, map_y_range_meters,
                      lidar_rps, pixels_to_a_meter,
                      map_center_x_px, map_center_y_px)
    running = True
    l_0 = logit(0.5)
    l_t = l_prev = np.full(obstacle_map.shape, l_0)
    dt = target_frame_time
    while running:
        start_time = time.time()
        # 1. get measurement for bearing
        car.update_relative_range_bearing(map_x_range_meters, map_y_range_meters)

        ran = car.sense(ground_truth_map = obstacle_map, noise = car.NOISE_OFF)
        print(f"dt:{dt}, "
              f"bearing:{np.round(car.lidar_bearing, 2)}, "
              f"range:{np.round(ran, 2)}")

        # 2. get p(mi| yt)
        m_t = perception_utils.inverse_measurement_model(ran, car.lidar_bearing, car)

        # 3. update belief map
        #print(l_t)
        l_prev = np.copy(l_t)
        l_t = perception_utils.update_belief_map(m_t, l_prev, l_0)
        grid_frame = cv2.cvtColor((expit(l_t)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        #grid_frame += np.random.random(grid_frame.shape)
        perception_utils.draw_perception_line(grid_frame, car.pos_px, ran, car.lidar_bearing, car.lidar_dphi, car.lidar_dr, pixels_to_a_meter)
        gt_map_draw = np.copy(img)
        gt_map_draw = cv2.cvtColor((gt_map_draw).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        perception_utils.draw_perception_line(gt_map_draw, car.pos_px, ran, car.lidar_bearing, car.lidar_dphi, car.lidar_dr, pixels_to_a_meter)

        cv2.imshow("grid_frame", grid_frame)
        cv2.imshow("gt_map", gt_map_draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Waits 1ms, breaks on 'q' key
            break
        car.update_state(dt) # for now, only rotates the lidar bearing

        ranges = car.ground_truth_map_ran
        bearings = car.ground_truth_map_bearing
        ground_truth_map_xx_yy_meters = car.ground_truth_map_xx_yy_meters
        delta = car.delta

        ###### after everything else is finished, calculate required sleep time ######
        dt = time.time() - start_time
        sleep_time = target_frame_time - dt
        if sleep_time > 0:
            time.sleep(sleep_time)
            dt = target_frame_time




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
