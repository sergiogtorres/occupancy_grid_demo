import numpy as np
import math
import perception_utils
import utils
import cv2
class Car():

    NOISE_OFF = 0
    NOISE_ON = 1
    def __init__(self, starting_pos, map_x_range, map_y_range, lidar_rps, pixels_to_a_meter, map_center_x_px, map_center_y_px):
        self.pos_px = np.array(starting_pos).astype(float) #position of the car, in [x, y] format, in px. Float.
        self.max_range = 10
        self.lidar_dr = 0.10
        self.lidar_dphi = 2*np.pi/100
        self.lidar_bearing = 0
        self.lidar_bearing_angular_speed = lidar_rps*2*np.pi
        self.pixels_to_a_meter = pixels_to_a_meter
        self.velocity = np.array([0, 0])
        self.base_speed = 10 #px/s
        self.current_speed = self.base_speed

        self.map_center_x_px = map_center_x_px
        self.map_center_y_px = map_center_y_px
        self.map_center_coords_px = np.array([self.map_center_x_px, self.map_center_y_px])
        self.neg_y = np.array([1, -1])

        ground_truth_map_xx_meters, ground_truth_map_yy_meters = np.meshgrid(map_x_range, map_y_range)
        self.ground_truth_map_xx_yy_meters = np.dstack([ground_truth_map_xx_meters, ground_truth_map_yy_meters])

        self.ground_truth_map_ran = self.ground_truth_map_bearing = None
        self.update_relative_range_bearing(map_x_range, map_y_range)
        self.delta = None

    def get_world_pos(self):
        return (self.pos_px - self.map_center_coords_px) * self.neg_y / self.pixels_to_a_meter
    def update_state(self, dt):

        self.lidar_bearing = utils.wrap_to_pi(self.lidar_bearing + dt*self.lidar_bearing_angular_speed)
        self.pos_px += self.velocity * dt
        self.velocity *= 0 # reset velocity after every iteration. This is just for debugging, so it's kept simple
    def update_relative_range_bearing(self, map_x_range, map_y_range):
        """
        Helper function to get the relative range and bearing meshgrids
        :return:
        """
        pos_meters = self.get_world_pos()
        print(f"pos_meters:{pos_meters}")
        self.delta = self.ground_truth_map_xx_yy_meters - pos_meters# / self.pixels_to_a_meter
        ground_truth_map_ran = np.linalg.norm(self.delta, axis = 2)
        ground_truth_map_bearing = np.atan2(self.delta[:,:,1], self.delta[:,:,0])

        self.ground_truth_map_ran = ground_truth_map_ran
        self.ground_truth_map_bearing = ground_truth_map_bearing



    def sense(self, ground_truth_map, noise = NOISE_OFF, frame_to_debug = None):
        """
        Uses the ground truth car position and ground truth obstacle map to generate a measurement.
        TODO: include noisy measurements (introduce measurement into bearing -> measure range -> introduce noise to range.

        Could actually obtain the measurement "mi" directly as the boolean array detections, but for realism, this
        function only returns the range where detections contains at least one detection is present
        :param bearing: desired angle to measure at
        :param ground_truth_map: the ground truth obstacle map
        :return: ran, the measured range r(φ).
        """
        mask_bearings = utils.angles_in_range(self.ground_truth_map_bearing, self.lidar_bearing, self.lidar_dphi / 2)

        r = np.min(self.ground_truth_map_ran[ground_truth_map & mask_bearings])
        r = min(r, self.max_range)

        print(0)
        #vvvv debug vvvv
        debug_r_lower = r - self.lidar_dr
        debug_r_upper = r + self.lidar_dr
        debug_ranges_mask = (self.ground_truth_map_ran >= debug_r_lower) & (self.ground_truth_map_ran <= debug_r_upper)


        frame_to_debug[debug_ranges_mask & mask_bearings] += np.array([0, 255, 0]).astype(np.uint8) # plotting obstacles
        #frame_to_debug += (np.random.random(frame_to_debug.shape)*255).astype(np.uint8)
        #frame_to_debug[mask_bearings] += np.array([255, 0, 0]).astype(np.uint8)



        #if frame_to_debug is not None:
        #    frame_to_debug[:,:,2] = 0
        #    frame_to_debug[:,:,0] = 0



        return r, self.ground_truth_map_ran[mask_bearings], ground_truth_map[mask_bearings], mask_bearings

    def move_up(self):
        self.velocity[1] = -self.current_speed
    def move_down(self):
        self.velocity[1]= self.current_speed

    def move_left(self):
        self.velocity[0]= -self.current_speed

    def move_right(self):
        self.velocity[0]= self.current_speed

    def speed_modifier(self, more_speed):
        if more_speed:
            self.current_speed = self.base_speed * 3
        else:
            self.current_speed = self.base_speed
