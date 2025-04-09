import numpy as np


def inverse_measurement_model(ran: float, bearing: float, car) -> float:

    m_t = car.check_within_range_bearing(car.ground_truth_map_ran, car.ground_truth_map_bearing,
                                         [ran-car.lidar_dr/2, ran+car.lidar_dr/2],
                                         [bearing-car.lidar_dphi/2, bearing+car.lidar_dphi/2],
                                         ground_truth_map=None)

    return m_t
