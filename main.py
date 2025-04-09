import numpy as np
import numpy.typing as npt
import car_lib
from scipy.special import logit

def get_meas_bearing(bearing: float) -> float:
    
    return ran 


def update_belief_map(m_t: npt.NDArray[np.float64], l_prev: npt.NDArray[np.float64], l_0: npt.NDArray[np.float64]) -> float:

    pass
    #return l_t

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    max_x, max_y =
    map_x_range = np.linspace(0, max_x, obstacle_map.shape[1])
    map_y_range = np.linspace(0, max_y, obstacle_map.shape[0])
    car = car_lib.Car()
    running = True
    while running:
        # 1. get measurement for bearing
        car.update_relative_range_bearing(map_x_range, map_y_range)

        # bearing = ...
        # ran = get_meas(bearing)

        # 2. get p(mi| yt)

        # mi = inverse_model(ran, bearing)

        # 3. update belief map

        # l_t = update_belief_map(m_t, l_prev, l_0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
