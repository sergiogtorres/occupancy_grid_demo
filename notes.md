
11 Apr 2025
Plotting the current bearing and distance ranges being checked helped me realize a silly bug: I was calling check_within_range_bearing() with self.lidar_bearing instead of self.lidar_dphi in Car.sense().
