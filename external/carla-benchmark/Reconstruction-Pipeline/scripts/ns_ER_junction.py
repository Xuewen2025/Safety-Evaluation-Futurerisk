import math
import os
import json
from ns_roadextraction_utils import RoadExtraction


class RoadExtractionJunction(RoadExtraction):

    def _get_road_by_id(self, road_id):
        for road in self.road_data:
            if road['id'] == road_id: 
                return road
        return None

    def _get_road_id_by_name(self, road_name):
        """
        Finds the OpenDRIVE road ID corresponding to the given road name.
        Assumes the road dictionary contains a 'name' field.
        """
        for road in self.road_data:
            if road['name'].lower() in road_name.lower() or road_name.lower() in road['name'].lower(): 
                return road['id']
        return None
    
    def _get_lane_boundaries(self, road):
        right_lane_width = 0.0
        left_lane_width = 0.0
        left_lane = False
        lane_section = road['lane_section'][0] 
        
        for lane in lane_section.get('right_lanes', []):
            if(lane['type'] == "driving"):
                right_lane_width += lane['width_a']

        for lane in lane_section.get('left_lanes', []):
            left_lane = True
            if(lane['type'] == "driving"):

                left_lane_width += lane['width_a']
        
        return right_lane_width, left_lane_width, left_lane

    def _get_terminal_point(self, road, junction_id):
        if not road or not road.get('geometry'):
            return None, None
        
        predecessor_type = None
        predecessor_id = None
        successor_type = None
        successor_id = None
        is_start_connected = False
        is_end_connected = False

        # Extract connection data safely
        if len(road['predecessor']) != 0:
            predecessor_type, predecessor_id = road['predecessor']

        if len(road['successor']) != 0:
            successor_type, successor_id = road['successor']

        if predecessor_type and predecessor_id:
            is_start_connected = (predecessor_type == 'junction' and predecessor_id == junction_id)

        if is_start_connected:
            first_geom = road['geometry'][0]
            X = first_geom['x']
            Y = first_geom['y']
            hdg = first_geom['hdg']
            return X, Y, hdg

        if successor_type and successor_id:
            is_end_connected = (successor_type == 'junction' and successor_id == junction_id)

        if is_end_connected:
            last_geom = road['geometry'][-1]
            
            hdg = last_geom['hdg']
            
            if last_geom['type'] == 'line':
                X = last_geom['x'] + last_geom['length'] * math.cos(last_geom['hdg'])
                Y = last_geom['y'] + last_geom['length'] * math.sin(last_geom['hdg'])
                
            elif last_geom['type'] == 'paramPoly3':
                (X, Y), _ = self._get_paramPoly3_point_and_tangent(last_geom, 1.0)

            if X is not None and Y is not None:
                return X, Y, hdg

        return None, None, None

    def _calculate_junction_quadrants(self, junction_id):
        
        junction_data = next((j for j in self.junction_data if j['id'] == junction_id), None)
        
        if junction_data is None:
            return None

        terminal_points = []

        junction_corner_candidates = []
        processed_road_ids = set()
        
        for connection in junction_data['connections']:
            
            # --- Check Incoming Road ---
            incoming_id = connection['incomingRoad']

            if incoming_id not in processed_road_ids:
                road = self._get_road_by_id(incoming_id)
                
                if road and road['junction'] == -1: 
                    X, Y, hdg = self._get_terminal_point(road, junction_id)
                    
                    if X is not None:
                        terminal_points.append((X, Y))
                        processed_road_ids.add(incoming_id)

                        # offset, left_lane = self._get_lane_boundaries(road)
                        right_lane_width, left_lane_width, left_lane = self._get_lane_boundaries(road)
                        
                        if not left_lane:
                            H_perp2 = hdg - math.pi / 2
                            # Corner Candidate 1 (Center + Offset 1)
                            C2_X = X + right_lane_width * math.cos(H_perp2)
                            C2_Y = Y + right_lane_width * math.sin(H_perp2)
                            junction_corner_candidates.append((C2_X, C2_Y))

                        else:
                            H_perp1 = hdg + math.pi / 2
                            H_perp2 = hdg - math.pi / 2
                            
                            # Corner Candidate 1 (Center + Offset 1)
                            C1_X = X + left_lane_width * math.cos(H_perp1)
                            C1_Y = Y + left_lane_width * math.sin(H_perp1)
                            
                            # Corner Candidate 2 (Center + Offset 2)
                            C2_X = X + right_lane_width * math.cos(H_perp2)
                            C2_Y = Y + right_lane_width * math.sin(H_perp2)

                            junction_corner_candidates.append((C1_X, C1_Y))
                            junction_corner_candidates.append((C2_X, C2_Y))

        if not terminal_points:
            return None # Cannot calculate extents

        X_coords = [p[0] for p in junction_corner_candidates]
        Y_coords = [p[1] for p in junction_corner_candidates]
        
        X_min, X_max = min(X_coords), max(X_coords)
        Y_min, Y_max = min(Y_coords), max(Y_coords)

        X_center = (X_min + X_max) / 2
        Y_center = (Y_min + Y_max) / 2
        
        midpoints = {
            'NE': ((X_center + X_max) / 2, (Y_center + Y_max) / 2),
            'NW': ((X_min + X_center) / 2, (Y_center + Y_max) / 2),
            'SW': ((X_min + X_center) / 2, (Y_min + Y_center) / 2),
            'SE': ((X_center + X_max) / 2, (Y_min + Y_center) / 2)
        }

        quadrant_data = {
            'junction_id': junction_id,
            'center': {'x': X_center, 'y': Y_center},
            'extents': {
                'X_min': X_min,
                'X_max': X_max,
                'Y_min': Y_min,
                'Y_max': Y_max
            },
            'midpoints': midpoints
        }
        
        return quadrant_data

    def _get_geometry_corners(self, geom, right_lane_width, left_lane_width, left, right):
        
        P_start_corner = None
        hdg_start = None
        P_end_corner = None
        hdg_end = None

        if geom['type'] == 'line':
            P_start_corner = (geom['x'], geom['y'])
            hdg_start = geom['hdg']
            P_end_corner = (
                geom['x'] + geom['length'] * math.cos(hdg_start),
                geom['y'] + geom['length'] * math.sin(hdg_start)
            )
            hdg_end = hdg_start

        elif geom['type'] == 'paramPoly3':
            # Start Point (u=0)
            P_start_corner, (Tx_start, Ty_start) = self._get_paramPoly3_point_and_tangent(geom, 0.0)
            hdg_start = math.atan2(Ty_start, Tx_start)
            
            # End Point (u=1)
            P_end_corner, (Tx_end, Ty_end) = self._get_paramPoly3_point_and_tangent(geom, 1.0)
            hdg_end = math.atan2(Ty_end, Tx_end)
        
        if P_start_corner is None:
            return None

        Nx_start = -math.sin(hdg_start)
        Ny_start = math.cos(hdg_start)
        
        C1_od = (round(P_start_corner[0] + left_lane_width * Nx_start, 2), round(P_start_corner[1] + left_lane_width * Ny_start, 2)) # Start, Left
        C2_od = (round(P_start_corner[0] - right_lane_width * Nx_start, 2), round(P_start_corner[1] - right_lane_width * Ny_start, 2)) # Start, Right
        
        Nx_end = -math.sin(hdg_end)
        Ny_end = math.cos(hdg_end)

        C3_od = (round(P_end_corner[0] - right_lane_width * Nx_end, 2), round(P_end_corner[1] - right_lane_width * Ny_end, 2)) # End, Right
        C4_od = (round(P_end_corner[0] + left_lane_width * Nx_end, 2), round(P_end_corner[1] + left_lane_width * Ny_end, 2)) # End, Left

        if right:
            right_segment = [
                (round(P_start_corner[0], 2), -round(P_start_corner[1], 2)), 
                (C2_od[0], -C2_od[1]),
                (C3_od[0], -C3_od[1]),
                (round(P_end_corner[0], 2), -round(P_end_corner[1], 2))
            ]

            return right_segment

        if left:
            left_segment = [
                (C1_od[0], -C1_od[1]), 
                (round(P_start_corner[0], 2), -round(P_start_corner[1], 2)),
                (round(P_end_corner[0], 2), -round(P_end_corner[1], 2)),
                (C4_od[0], -C4_od[1])
            ]
            return left_segment

        return None
    
    def _find_neighbour_segments(self, connecting_road_segment, distance_each_side, left, right):
        road_id = int(connecting_road_segment['predecessor_road_id'])
        s_start_point = connecting_road_segment['first_segment']['s']

        road = next((r for r in self.road_data if r['id'] == road_id), None)
        if road is None:
            return []

        right_lane_width, left_lane_width, _ = self._get_lane_boundaries(road)

        geometries = road['geometry']
        
        current_geom_index = -1
        for i, geom in enumerate(geometries):
            if geom['s'] <= s_start_point < (geom['s'] + geom['length']):
                current_geom_index = i
                break
        
        if current_geom_index == -1:
            return []

        current_segments = [geometries[current_geom_index]]
        
        previous_segments = []
        s_required_prev = s_start_point - distance_each_side
        i = current_geom_index - 1
        
        while i >= 0:
            geom = geometries[i]
            
            if (geom['s'] + geom['length']) >= s_required_prev:
                previous_segments.insert(0, geom)
                
                if geom['s'] <= s_required_prev:
                    break
            i -= 1

        next_segments = []
        s_required_next = s_start_point + distance_each_side
        i = current_geom_index + 1
        
        while i < len(geometries):
            geom = geometries[i]
            
            if geom['s'] <= s_required_next:
                next_segments.append(geom)
                
                if (geom['s'] + geom['length']) >= s_required_next:
                    break
            i += 1 

        all_geometries = previous_segments + current_segments + next_segments
        all_segments_corners = []

        for geom in all_geometries:
            segment = self._get_geometry_corners(geom, right_lane_width, left_lane_width, left, right)
            if segment:
                all_segments_corners.append((segment))
                
        # The final list structure is: [(left_segment, right_segment), (left_segment, right_segment), (left_segment, right_segment), ...]
        #left segment structure: [(corner1_x, corner1_y), (corner2_x, corner2_y), ... corner4]
        return all_segments_corners

    def _find_closest_junction(self, road_id, current_s):
        """
        Finds the junction connected to the given road.
        If the road connects to two junctions (one at start, one at end),
        it uses 'current_s' to determine which one is physically closer.
        """
        road = self._get_road_by_id(road_id)
        if not road:
            print(f"Error: Road {road_id} not found.")
            return None

        pred_type, pred_id = road['predecessor'] if road['predecessor'] else (None, None)
        succ_type, succ_id = road['successor'] if road['successor'] else (None, None)

        is_pred_junction = (pred_type == 'junction')
        is_succ_junction = (succ_type == 'junction')

        if is_pred_junction and not is_succ_junction:
            return pred_id

        if is_succ_junction and not is_pred_junction:
            return succ_id

        if is_pred_junction and is_succ_junction:
            if current_s is not None:
                dist_to_start = current_s              
                dist_to_end = float(road['length']) - current_s 
                
                if dist_to_start < dist_to_end:
                    return pred_id
                else:
                    return succ_id

        return None
    
    def _find_junction_roads_exiting_to(self, junction_id, external_road_id):
        """
        Finds all internal junction road segments that flow out 
        (have the target external road as their successor).
        """
        best_match = {
            'type': None,
            'road_id': None,
            'junction_id': None,
            's': None,
            'hdg': None,
            'hdg_at_CP': None,
            'segment_corners': None 
        }

        all_junction_roads = []
        junction_id = junction_id

        for road in self.road_data:
            successor = road['successor']
            if road['junction'] == junction_id and successor[0] == 'road' and successor[1] == external_road_id:
                    right_lane_width, left_lane_width, _ = self._get_lane_boundaries(road)

                    for geom in road['geometry']:

                        P_ref = None
                        t_norm = None
                        
                        if geom['type'] == 'line':
                            P_start = (geom['x'], geom['y'])
                            hdg_at_end = geom['hdg']
                            length = geom['length']

                            P_end = (
                                geom['x'] + geom['length'] * math.cos(hdg_at_end),
                                geom['y'] + geom['length'] * math.sin(hdg_at_end)
                            )

                            roadtype = 'line'
                            
                        elif geom['type'] == 'paramPoly3':
                            t_norm = 1 #at the end of this pareampoly
                            
                            (_, _), (Tx_at_ref, Ty_at_ref) = self._get_paramPoly3_point_and_tangent(geom, t_norm)

                            hdg_at_end = math.atan2(Ty_at_ref, Tx_at_ref)
                            length = geom['length'] 

                            roadtype = 'parampoly'

                        if 0 <= t_norm <= 1:
                            left = False
                            right = True
                            corners = self._get_geometry_corners(geom, right_lane_width, left_lane_width, left, right)
                            
                            best_match['type'] = roadtype
                            best_match['road_id'] = road['id']
                            best_match['junction_id'] = junction_id
                            best_match['s'] = geom['s']
                            best_match['hdg'] = geom['hdg']
                            best_match['hdg_at_CP'] = hdg_at_end
                            best_match['segment_corners'] = corners #in carla coordinate

                        all_junction_roads.append(best_match.copy())

        return all_junction_roads
    
    def _normalize_angle(self, angle):
        """Normalizes an angle to the range [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _determine_maneuver_direction(self, road_id):
        """
        Calculates the maneuver (Straight, Left, Right) for a junction road
        assuming the road consists of a SINGLE geometry element.
        """
        road = self._get_road_by_id(road_id)
        if not road or not road['geometry']:
            return "Unknown"

        geom = road['geometry'][0]
        
        hdg_start = geom['hdg']
        hdg_end = hdg_start
        
        if geom['type'] == 'line':
            hdg_end = geom['hdg']
            
        elif geom['type'] == 'paramPoly3':
            (_, _), (Tx, Ty) = self._get_paramPoly3_point_and_tangent(geom, 1.0)
            hdg_end = math.atan2(Ty, Tx)

        angle_diff = self._normalize_angle(hdg_end - hdg_start)

        STRAIGHT_THRESHOLD = 0.4  # ~23 degrees
        UTURN_THRESHOLD = 2.1     # ~120 degrees (Anything sharper than this is a U-turn)

        # print("road id: ", road_id)
        # print("angle diff: ", angle_diff)
        if abs(angle_diff) > UTURN_THRESHOLD:
            return "U-Turn"
        elif angle_diff > STRAIGHT_THRESHOLD:
            return "Turning Left"
        elif angle_diff < -STRAIGHT_THRESHOLD:
            return "Turning Right"
        else:
            return "Going Straight"

    def find_junction(self, P):

        best_match = {
            'distance': float('inf'),
            'type': None,
            'road_id': None,
            'junction_id': None,
            's': None,
            's_total': None,
            't': None,
            'P_ref_x': None,
            'P_ref_y': None,
            'hdg': None,
            'hdg_at_CP': None,
            'segment_corners': None 
        }

        P_x, P_y = P 
        junction = False

        for road in self.road_data:
            road_id = road['id']
            junction_id = road['junction']
            min_distance = float('inf')
            
            right_lane_width, left_lane_width, _ = self._get_lane_boundaries(road) 
            
            for geom in road['geometry']:   

                P_ref = None
                t_norm = None
                
                if geom['type'] == 'line':
                    P_start = (geom['x'], geom['y'])
                    hdg = geom['hdg']
                    length = geom['length']

                    P_end = (
                        geom['x'] + geom['length'] * math.cos(hdg),
                        geom['y'] + geom['length'] * math.sin(hdg)
                    )

                    P_ref, t_norm = self._calculate_closest_point_on_line(P, P_start, P_end)

                    roadtype = 'line'
                    
                elif geom['type'] == 'paramPoly3':
                    P_ref, t_norm = self._calculate_closest_point_on_paramPoly3(P, geom)
                    
                    (_, _), (Tx_at_ref, Ty_at_ref) = self._get_paramPoly3_point_and_tangent(geom, t_norm)
                    
                    hdg = math.atan2(Ty_at_ref, Tx_at_ref)
                    length = geom['length'] # Used for s calculation

                    roadtype = 'parampoly'

                distance = math.sqrt((P_x - P_ref[0])**2 + (P_y - P_ref[1])**2)

                if distance < min_distance:
                    min_distance = distance

                if distance <= right_lane_width and distance < best_match['distance']:

                    s_current_geom = t_norm * length
                    s_total = geom['s'] + s_current_geom 
                    
                    Tx = math.cos(hdg) # Tangent vector X
                    Ty = math.sin(hdg) # Tangent vector Y
                    Vx = P[0] - P_ref[0]       # Vector from P_ref to P (Lateral X)
                    Vy = P[1] - P_ref[1]       # Vector from P_ref to P (Lateral Y)
                          
                    signed_distance = distance * math.copysign(1, -(Tx * Vy - Ty * Vx))
                    
                    if 0 <= t_norm <= 1:
                        left = False
                        right = True
                        corners = self._get_geometry_corners(geom, right_lane_width, left_lane_width, left, right)
                        
                        best_match['distance'] = distance
                        best_match['type'] = roadtype
                        best_match['road_id'] = road_id
                        best_match['junction_id'] = junction_id
                        best_match['s'] = geom['s']
                        best_match['s_total'] = s_total
                        best_match['t'] = signed_distance
                        best_match['hdg'] = geom['hdg']
                        best_match['hdg_at_CP'] = hdg
                        best_match['segment_corners'] = corners #in carla coordinate

                        if junction_id > 0:
                            junction = True
                            return junction, best_match
           
        return junction, best_match

    def move_CP_to_closest_junction(self, P, quadrant_data):

        P_x, P_y = P
        
        closest_midpoint = None
        min_distance_sq = float('inf') # Use squared distance for efficiency

        midpoints = quadrant_data['midpoints']

        for direction, M in midpoints.items():
            M_x, M_y = M

            distance_sq = (P_x - M_x)**2 + (P_y - M_y)**2
            
            # print(f"  - {direction}: ({M_x:.2f}, {M_y:.2f}), Distance Squared: {distance_sq:.2f}")

            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_midpoint = M
        
        return closest_midpoint

    def find_valid_junctions_from_CP(self, P):
        best_match = {
            'distance': float('inf'),
            'type': None,
            'road_id': None,
            'junction_id': None,
            's': None,
            't': None,
            'P_ref_x': None,
            'P_ref_y': None,
            'hdg': None,
            'hdg_at_CP': None,
            'segment_corners': None 
        }

        P_x, P_y = P # Unpack query point
        all_junctions = []
        candidate_junctions = []

        for road in self.road_data:
            road_id = road['id']
            junction_id = road['junction']

            right_lane_width, left_lane_width, _ = self._get_lane_boundaries(road) 
            
            for geom in road['geometry']:   

                P_ref = None
                t_norm = None
                
                if geom['type'] == 'line':
                    P_start = (geom['x'], geom['y'])
                    hdg = geom['hdg']
                    length = geom['length']

                    P_end = (
                        geom['x'] + geom['length'] * math.cos(hdg),
                        geom['y'] + geom['length'] * math.sin(hdg)
                    )

                    P_ref, t_norm = self._calculate_closest_point_on_line(P, P_start, P_end)

                    roadtype = 'line'
                    
                elif geom['type'] == 'paramPoly3':
                    P_ref, t_norm = self._calculate_closest_point_on_paramPoly3(P, geom)
                    
                    (_, _), (Tx_at_ref, Ty_at_ref) = self._get_paramPoly3_point_and_tangent(geom, t_norm)
                    
                    hdg = math.atan2(Ty_at_ref, Tx_at_ref)
                    length = geom['length']

                    roadtype = 'parampoly'

                distance = math.sqrt((P_x - P_ref[0])**2 + (P_y - P_ref[1])**2)

                if distance <= right_lane_width and 0 <= t_norm <= 1 and junction_id > 0:
                    
                    s_current_geom = t_norm * length
                    s_total = geom['s'] + s_current_geom 
                    
                    Tx = math.cos(hdg) # Tangent vector X
                    Ty = math.sin(hdg) # Tangent vector Y
                    Vx = P[0] - P_ref[0]       # Vector from P_ref to P (Lateral X)
                    Vy = P[1] - P_ref[1]       # Vector from P_ref to P (Lateral Y)
                          
                    signed_distance = distance * math.copysign(1, -(Tx * Vy - Ty * Vx))
                    # print("signed distance: ", signed_distance)
                    
                    left = False
                    right = True
                    corners = self._get_geometry_corners(geom, right_lane_width, left_lane_width, left, right)
                    
                    best_match['distance'] = distance
                    best_match['type'] = roadtype
                    best_match['road_id'] = road_id
                    best_match['junction_id'] = junction_id
                    best_match['s'] = geom['s']
                    best_match['t'] = signed_distance
                    best_match['hdg'] = geom['hdg']
                    best_match['hdg_at_CP'] = hdg
                    best_match['segment_corners'] = corners #in carla coordinate

                    if best_match['distance'] <= right_lane_width and best_match['t'] > 0:
                        
                        all_junctions.append(best_match.copy())
                    elif best_match['distance'] <= 2 and best_match['t'] < 0:
                        candidate_junctions.append(best_match.copy())
        
        if len(all_junctions) < 2 and candidate_junctions:
            print("in")
            candidate_junctions.sort(key=lambda x: x['distance'])
        
            for candidate in candidate_junctions:
                if len(all_junctions) >= 2:
                    break
                all_junctions.append(candidate)

        return all_junctions
     
    def find_connecting_roads(self, all_junctions, distance_each_side):

        connecting_road_first_segment = {
            'junction_road_id': None,
            'predecessor_road_id': None,
            'first_segment': None
        }

        all_connected_road_segments = []

        for junction_road in all_junctions:
            left = False
            right = False

            junc_road_id = junction_road['road_id']
            junction_id = junction_road['junction_id']
            junc_road = self._get_road_by_id(junc_road_id)
            predecessor = junc_road['predecessor']
            
            if predecessor and predecessor[0] == 'road':
                predecessor_id = predecessor[1]

                pred_road = self._get_road_by_id(predecessor_id)
                if pred_road and pred_road['junction'] == -1:
                    first_segment = None
                    pred_geometries = pred_road['geometry']

                    # 1. Successor Case: R_pred connects via its SUCCESSOR link (s=L)
                    # This means R_pred flows toward the junction. We want the segment at s=L.
                    pred_successor = pred_road['successor']
                    pred_predecessor = pred_road['predecessor']

                    if len(pred_successor) != 0:
                        if (pred_successor[0] == 'junction' and pred_successor[1] == junction_id):
                            first_segment = pred_geometries[-1]

                    # 2. Predeccessor Case: R_pred connects via its PREDECESSOR link (s=0)
                    # This means R_pred starts at the junction. We want the segment at s=0.
                    if first_segment is None and len(pred_predecessor) != 0:
                        if (pred_predecessor[0] == 'junction' and pred_predecessor[1] == junction_id):
                            first_segment = pred_geometries[0]
                            # print(first_segment)

                    if first_segment:
                        # print("first seg: ", first_segment)
                        connecting_road_first_segment = {
                            'junction_road_id': junc_road_id,
                            'predecessor_road_id': predecessor_id,
                            'first_segment': first_segment
                        }

            if first_segment['hdg'] == junction_road['hdg']:
                left = False
                right = True
            else: 
                left = True
                right = False

            segments = self._find_neighbour_segments(connecting_road_first_segment, distance_each_side, left, right)
            connected_road_segments = {
                'junction_id': junction_id,
                'junction_road_id': junc_road_id,
                'junction_road_segment': junction_road['segment_corners'],
                'connected_road_id': predecessor_id,
                'connected_road_segments': segments,
            }

            all_connected_road_segments.append(connected_road_segments)
             
        return all_connected_road_segments

    def filter_junction_candidates(self, junction_candidates, crash_description):
        """
        Filters the 'junction_candidates' list.
        
        Args:
            junction_candidates: The list returned by find_valid_junctions_from_CP
            crash_description: String (e.g., "Going Straight", "Turning Left")
            
        Returns:
            A filtered list containing only the junction candidates matching the description.
        """
        filtered_results = []
        target_maneuver = crash_description.lower()
        for candidate in junction_candidates:
            road_id = candidate['road_id']    
            detected_maneuver = self._determine_maneuver_direction(road_id)
            # print("road id: ", road_id)
            # print(detected_maneuver)
            match = False
            if "u-turn" in target_maneuver or "uturn" in target_maneuver:
                if detected_maneuver == "U-Turn":
                    match = True
            elif "left" in target_maneuver and  "left" in detected_maneuver.lower():
                    match = True   
            elif "right" in target_maneuver and "right" in detected_maneuver.lower():
                match = True
            elif "straight" in target_maneuver and "straight" in detected_maneuver.lower():
                match = True
            
            if match:
                candidate['maneuver_type'] = detected_maneuver
                filtered_results.append(candidate)
        
        return filtered_results

    def check_point_on_road(self, P_x, P_y, distance_each_side, crash_description):
        P_opendrive = -P_y
        P = (P_x, P_opendrive)
        
        junction, best_match = self.find_junction(P)

        if junction:
            # """ This function calculates the center of the junction 
            # and the 4 middles points of each segment of that junction """
            # quadrant_data = self._calculate_junction_quadrants(best_match['junction_id'])
            # # print("quadrant data: ", quadrant_data)

            # """ This function will move the crash point to the closest segment's 
            # middle point to simplify things. """
            # P_new = self.move_CP_to_closest_junction(P, quadrant_data)
            # # print("P new: ", P_new)

            """ This function finds all the junction roads the crash point falls in. """
            all_junctions = self.find_valid_junctions_from_CP(P)
            # print("printing all junctions...", len(all_junctions))
            # print(all_junctions)

        else:
            """ If the crash point is not on any junction rather it's on a road but the crash involved the junction """
            junction_id = self._find_closest_junction(best_match['road_id'], best_match['s_total'])

            if junction_id:
                external_road_id = best_match['road_id']
                all_junctions = self._find_junction_roads_exiting_to(junction_id, external_road_id)
                # print(f"Found {len(all_junctions)} junction roads exiting to {external_road_id}:")
                # print("all junctions: ", all_junctions)

        """ This function filters the valid junction according to the vehicles pre-crash description. """
        filtered_junctions = self.filter_junction_candidates(all_junctions, crash_description)
        # print("Filtered junction: ", filtered_junctions)
        """ This function will find the connecting road for each junction 
        and the segments of that road """
        all_connected_road_segments = self.find_connecting_roads(filtered_junctions, distance_each_side)
        
        return all_connected_road_segments
    