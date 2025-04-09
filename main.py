import numpy as np
import numpy.typing as npt
import cv2
from scipy.special import logit, expit
import time

import car_lib
import perception_utils

# TODO: check all instances of use of position -> x, y correct order? cv2 is x, y!
def draw_perception_line(im, start, range, bearing, dphi):
    """
    TODO: adjust to detected range?
    :param im:
    :param start:
    :param range:
    :param bearing:
    :param dphi:
    :return:
    """
    pt1 = tuple(start)
    delta_a = range*np.array([np.cos(bearing+dphi), np.sin(bearing+dphi)])
    delta_b = range*np.array([np.cos(bearing-dphi), np.sin(bearing-dphi)])
    x2_a, y2_a = (start + delta_a).astype(np.uint32)
    x2_b, y2_b = (start + delta_b).astype(np.uint32)
    pt2_a = x2_a, y2_a
    pt2_b = x2_b, y2_b
    #print(f"drawing line from {pt1} to {pt2_a} and {pt2_b}")
    cv2.line(im, pt1, pt2_a, (0,255,0), 1)
    cv2.line(im, pt1, pt2_b, (0,255,0), 1)

    cv2.putText(im, f"bearing:{np.round(bearing, 2)}, "
                    f"range:{np.round(range, 2), }"
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
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    target_fps = 30
    target_frame_time = 1/target_fps
    ### Loading obstacle_map ###
    # Load as grayscale
    img = cv2.imread('map_big.bmp', cv2.IMREAD_GRAYSCALE)
    print(img.shape)  # e.g., (height, width)
    obstacle_map = img == 0
    lidar_rps = 0.5
    #############################
    max_x, max_y = 20, 20 #in meters
    map_x_range = np.linspace(0, max_x, obstacle_map.shape[1])
    map_y_range = np.linspace(0, max_y, obstacle_map.shape[0])
    car = car_lib.Car([obstacle_map.shape[1]//2, obstacle_map.shape[0]//2],
                      map_x_range,map_y_range, lidar_rps)
    running = True
    l_0 = logit(0.5)
    l_t = l_prev = np.full(obstacle_map.shape, l_0)
    dt = target_frame_time
    while running:
        print(f"dt:{dt}, bearing:{np.round(car.lidar_bearing, 2)}")
        start_time = time.time()
        # 1. get measurement for bearing
        car.update_relative_range_bearing(map_x_range, map_y_range)

        ran = car.sense(ground_truth_map = obstacle_map, noise = car.NOISE_OFF)

        # 2. get p(mi| yt)
        m_t = perception_utils.inverse_measurement_model(ran, car.lidar_bearing, car)

        # 3. update belief map
        print(l_t)
        l_prev = np.copy(l_t)
        l_t = perception_utils.update_belief_map(m_t, l_prev, l_0)
        grid_frame = cv2.cvtColor((expit(l_t)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        #grid_frame += np.random.random(grid_frame.shape)
        draw_perception_line(grid_frame, car.pos, ran, car.lidar_bearing, car.lidar_dphi)
        cv2.imshow("grid_frame", grid_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Waits 1ms, breaks on 'q' key
            break
        car.update_state(dt) # for now, only rotates the lidar bearing

        ###### after everything else is finished, calculate required sleep time ######
        dt = time.time() - start_time
        sleep_time = target_frame_time - dt
        if sleep_time > 0:
            time.sleep(sleep_time)
            dt = target_frame_time




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
