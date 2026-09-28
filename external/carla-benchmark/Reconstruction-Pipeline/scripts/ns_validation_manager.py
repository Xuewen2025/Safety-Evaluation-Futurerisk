import carla
import math

DISTANCE_THRESHOLD = 10.0
TIME_THRESHOLD = 10.0

class ValidationManager:
    def __init__(self, crash_location):
        """
        Args:
            crash_location (tuple): (x, y) of the expected crash center.
            distance_threshold (float): Radius (meters) to consider "arrived" at crash zone.
            time_threshold (float): Max allowed time difference (seconds) between arrivals.
        """
        self.crash_x, self.crash_y = crash_location
        
        # State tracking
        self.arrival_time_v1 = None
        self.arrival_time_v2 = None
        self.crash_event_detected = False
        
        # Results
        self.time_difference = None
        self.impact_angle_v1 = None # Clock point for V1
        self.impact_angle_v2 = None # Clock point for V2
        self.simulation_valid = False

    def _get_distance(self, location):
        return math.sqrt((location.x - self.crash_x)**2 + (location.y - self.crash_y)**2)

    def _normalize_angle(self, angle):
        """Normalize angle to [0, 360)"""
        while angle < 0: angle += 360
        while angle >= 360: angle -= 360
        return angle

    def _calculate_clock_point(self, actor_transform, other_actor_loc):
        """
        Calculates 'Clock Point' (1-12) using local vector transformation.
        Robust against global coordinate system differences.
        """
        fwd_vec = actor_transform.get_forward_vector()
        right_vec = actor_transform.get_right_vector()
        
        dx = other_actor_loc.x - actor_transform.location.x
        dy = other_actor_loc.y - actor_transform.location.y

        local_x = (dx * fwd_vec.x) + (dy * fwd_vec.y)
        local_y = (dx * right_vec.x) + (dy * right_vec.y)
        
        # 4. Calculate Angle in Local Space
        # atan2(y, x) gives angle from Forward axis towards Right axis
        # In this local space: 
        #   0 deg = Forward (12 o'clock)
        #   90 deg = Right (3 o'clock)
        #   -90 deg = Left (9 o'clock)
        #   180 deg = Back (6 o'clock)
        angle = math.degrees(math.atan2(local_y, local_x))
        
        # 5. Map to Clock
        # Since +90 is 3 o'clock (Right), we can map directly:
        # Angle 0 -> 12
        # Angle 90 -> 3
        # Angle 180 -> 6
        # Angle -90 -> 9
        
        # Formula: Clock = (Angle / 30)
        # If angle is 0, we want 12.
        # If angle is 90, we want 3. 
        # Let's adjust angle to "Clockwise from 12" style first.
        
        # Current: 0=Fwd, 90=Right. This IS Clockwise!
        # So we just need to handle the offset.
        
        clock_val = angle / 30.0
        
        if clock_val <= 0:
            # Handle negative angles (Left side, e.g. -90 -> -3)
            # -3 should be 9. 12 + (-3) = 9.
            # 0 should be 12.
            final_clock = 12 + clock_val
        else:
            # Positive angles (Right side, e.g. 90 -> 3)
            final_clock = clock_val
            
        final_clock = round(final_clock)
        
        # Handle edge case of rounding 11.9 to 12 or 0.1 to 0 -> 12
        if final_clock == 0: return 12
        return int(final_clock)

    def _is_angle_valid(self, calculated, expected):
        """Helper to check if calculated clock matches expected clock (+/- 1 hour tolerance)"""
     
        diff = abs(calculated - expected)
        
        if diff <= 2 or diff >= 10:
            return True
            
        return False
    
    def _normalize_angle_diff_rad(self, angle):
        """Normalize angle difference (radians) to range [-pi, pi]"""
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def _analyze_trajectory_geometry(self, route):
        """
        Analyzes a list of (x, y) points to determine the maneuver direction
        using specific radian thresholds.
        """
        if not route or len(route) < 2:
            return "Unknown"

        start_idx = 0
        end_start_sample = min(5, len(route)-1)
        
        dx_start = route[end_start_sample][0] - route[start_idx][0]
        dy_start = route[end_start_sample][1] - route[start_idx][1]
        heading_start = math.atan2(-dy_start, dx_start) # y-axis flipped because of carla to opendrive tranformation

        end_idx = len(route) - 1
        start_end_sample = max(0, len(route) - 6)
        
        dx_end = route[end_idx][0] - route[start_end_sample][0]
        dy_end = route[end_idx][1] - route[start_end_sample][1]
        heading_end = math.atan2(-dy_end, dx_end)  # y-axis flipped because of carla to opendrive tranformation

        angle_diff = self._normalize_angle_diff_rad(heading_end - heading_start)
        
        STRAIGHT_THRESHOLD = 0.45  # ~23 degrees
        UTURN_THRESHOLD = 2.1     # ~120 degrees

        if abs(angle_diff) > UTURN_THRESHOLD:
            return "U-Turn"
        elif angle_diff > STRAIGHT_THRESHOLD:
            return "Turning Left"
        elif angle_diff < -STRAIGHT_THRESHOLD:
            return "Turning Right"
        else:
            return "Going Straight"
        
    def update_arrival_times(self, current_sim_time, veh1_loc, veh2_loc):
        """
        Call this every simulation tick to check if vehicles entered the crash zone.
        """
        if self.arrival_time_v1 is None:
            if self._get_distance(veh1_loc) < DISTANCE_THRESHOLD:
                self.arrival_time_v1 = current_sim_time
                # print(f"[Validation] Vehicle 1 arrived at crash zone at {current_sim_time:.2f}s")

        if self.arrival_time_v2 is None:
            if self._get_distance(veh2_loc) < DISTANCE_THRESHOLD:
                self.arrival_time_v2 = current_sim_time
                # print(f"[Validation] Vehicle 2 arrived at crash zone at {current_sim_time:.2f}s")

    def register_crash(self, vehicle1, vehicle2, veh1_route, veh2_route, veh1_direction, veh2_direction, expected_clock_v1=None, expected_clock_v2=None):
        """
        Call this ONLY when the collision sensor triggers.
        Calculates angles and validates the scenario.
        """
        if self.crash_event_detected:
            return
        
        self.crash_event_detected = True
        
        valid_time = False
        if self.arrival_time_v1 is not None and self.arrival_time_v2 is not None:
            self.time_difference = abs(self.arrival_time_v1 - self.arrival_time_v2)
            if self.time_difference <= TIME_THRESHOLD:
                valid_time = True
        else:
            valid_time = False

        # if valid_time:
        if expected_clock_v1 == -1 or expected_clock_v2 == -1:
            valid_angle_v1 = True
            valid_angle_v2 = True

        else:
            t1 = vehicle1.get_transform()
            t2 = vehicle2.get_transform()
            
            calc_clock_v1 = self._calculate_clock_point(t1, t2.location)
            
            calc_clock_v2 = self._calculate_clock_point(t2, t1.location)

            self.impact_angle_v1 = calc_clock_v1
            self.impact_angle_v2 = calc_clock_v2

            valid_angle_v1 = self._is_angle_valid(calc_clock_v1, expected_clock_v1)

            valid_angle_v2 = self._is_angle_valid(calc_clock_v2, expected_clock_v2)
            
        detected_traj_v1 = self._analyze_trajectory_geometry(veh1_route)
        detected_traj_v2 = self._analyze_trajectory_geometry(veh2_route)

        def check_traj_match(detected, report):
            if report is None:
                return True
            
            d = detected.lower()
            r = report.lower()
            
            if "u-turn" in d or "uturn" in d:
                if "u-turn" in r or "uturn" in r: return True
                
            if "straight" in d and "straight" in r: return True
            if "left" in d and "left" in r: return True
            if "right" in d and "right" in r: return True
            
            return False

        valid_trajectory_v1 = check_traj_match(detected_traj_v1, veh1_direction)
        
        valid_trajectory_v2 = check_traj_match(detected_traj_v2, veh2_direction)
        

        # Final Validation Log
        print("========= Validation Log ==========")
        pass_or_fail_time = "N/A"
        pass_or_fail_angle = "N/A"
        pass_or_fail_trajectory = "N/A"
        if valid_time:
            pass_or_fail_time = "PASSED"
        else:
            pass_or_fail_time = "FAILED"
        if valid_angle_v1 and valid_angle_v2:
            pass_or_fail_angle = "PASSED"
        else:
            pass_or_fail_angle = "FAILED"
        
        if valid_trajectory_v1 and valid_trajectory_v2:
            pass_or_fail_trajectory = "PASSED"
        else:
            pass_or_fail_trajectory = "FAILED"
        
        
        print(f">>> 1. Crashed at the crash location: {pass_or_fail_time} <<<")
        print(f">>> 2. Impact Point Match: {pass_or_fail_angle} <<<")
        print(f">>> 3. Trajectory Match: {pass_or_fail_trajectory} <<<")

        