import numpy as np
import math
import perception_utils
import utils
class Car():

    NOISE_OFF = 0
    NOISE_ON = 1
    def __init__(self, starting_pos, map_x_range, map_y_range, lidar_rps, pixels_to_a_meter, map_center_x_px, map_center_y_px):
        self.pos_px = np.array(starting_pos) #position of the car, in [x, y] format, in px
        self.max_range = 10
        self.lidar_dr = 0.10
        self.lidar_dphi = 2*np.pi/100
        self.lidar_bearing = 0
        self.lidar_bearing_angular_speed = lidar_rps*2*np.pi
        self.pixels_to_a_meter = pixels_to_a_meter

        self.map_center_x_px = map_center_x_px
        self.map_center_y_px = map_center_y_px
        self.map_center_coords_px = np.array([self.map_center_x_px, self.map_center_y_px])
        self.neg_y = np.array([0, -1])

        ground_truth_map_xx_meters, ground_truth_map_yy_meters = np.meshgrid(map_x_range, map_y_range)
        self.ground_truth_map_xx_yy_meters = np.dstack([ground_truth_map_xx_meters, ground_truth_map_yy_meters])

        self.ground_truth_map_ran = self.ground_truth_map_bearing = None
        self.update_relative_range_bearing(map_x_range, map_y_range)
        self.delta = None

    def get_world_pos(self):
        return (self.pos_px - self.map_center_coords_px) * self.neg_y / self.pixels_to_a_meter
    def update_state(self, dt):

        self.lidar_bearing = utils.wrap_to_pi(self.lidar_bearing + dt*self.lidar_bearing_angular_speed)
    def update_relative_range_bearing(self, map_x_range, map_y_range):
        """
        Helper function to get the relative range and bearing meshgrids
        :return:
        """
        pos_meters = self.get_world_pos()
        self.delta = self.ground_truth_map_xx_yy_meters - pos_meters / self.pixels_to_a_meter
        ground_truth_map_ran = np.linalg.norm(self.delta, axis = 2)
        ground_truth_map_bearing = np.atan2(self.delta[:,:,1], self.delta[:,:,0])

        self.ground_truth_map_ran = ground_truth_map_ran
        self.ground_truth_map_bearing = ground_truth_map_bearing



    def sense(self, ground_truth_map, noise = NOISE_OFF, frame_to_debug = None):
        """
        Uses the ground truth car position and ground truth obstacle map to generate a measurement.
        TODO: include noisy measurements (introduce measurement into bearing -> measure range -> introduce noise to range.

        Could actually obtain the measurement mi directly as the boolean array detections, but for realism, this
        function only returns the range where detections contains at least one detection is present
        :param bearing: desired angle to measure at
        :param ground_truth_map: the ground truth obstacle map
        :return: ran, the measured range r(φ).
        """

        # convert the entire ground_truth_map coordinates into relative range and bearing
        # 1. loop through the ray to check if obstacle
        for r in np.linspace(0, self.max_range, num = math.ceil(self.max_range/self.lidar_dr)):
            # TODO: before looping, filter the arrays so I only consider the relevant bearings -> index transformations
            #range_down_up = [r-self.lidar_dr, r+self.lidar_dr]
            #bearing_down_up = [self.lidar_bearing-self.lidar_dphi, self.lidar_bearing+self.lidar_dphi]
            detections, _, _ = \
                perception_utils.check_within_range_bearing(self.ground_truth_map_ran, self.ground_truth_map_bearing,
                                                            r, self.lidar_bearing, self.lidar_dr, self.lidar_bearing,
                                                            ground_truth_map,
                                                            mode=perception_utils.DETECTION_MODE,
                                                            frame_to_debug = frame_to_debug)

            if np.any(detections) or np.sum(detections)>0:
                print(f"detection @ r:{np.round(r, 2)}, bearing:{np.round(self.lidar_bearing, 2)}, ")
                break



        # 1.a if obstacle, stop, and we have the range


        return r
