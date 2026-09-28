import math
import os
from ns_roadextraction_utils import RoadExtraction


class RoadExtractionNonJunction(RoadExtraction):

    def _check_lanes(self, road):
        right = False
        left = False
        lane_section = road['lane_section'][0]

        if lane_section['right_lanes']:
            right = True
        if lane_section['left_lanes']:
            left = True
        return left, right

    def _get_geometry_corners(self, geom, lane_width, left, right):
        
        P_start_corner = None
        hdg_start = None
        P_end_corner = None
        hdg_end = None
        left_segment = None
        right_segment = None

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
        
        C1_od = (round(P_start_corner[0] + lane_width * Nx_start, 2), round(P_start_corner[1] + lane_width * Ny_start, 2)) # Start, Left
        C2_od = (round(P_start_corner[0] - lane_width * Nx_start, 2), round(P_start_corner[1] - lane_width * Ny_start, 2)) # Start, Right
        
        Nx_end = -math.sin(hdg_end)
        Ny_end = math.cos(hdg_end)

        C3_od = (round(P_end_corner[0] - lane_width * Nx_end, 2), round(P_end_corner[1] - lane_width * Ny_end, 2)) # End, Right
        C4_od = (round(P_end_corner[0] + lane_width * Nx_end, 2), round(P_end_corner[1] + lane_width * Ny_end, 2)) # End, Left

        if left:
            left_segment = [
                (C1_od[0], -C1_od[1]), 
                (round(P_start_corner[0], 2), -round(P_start_corner[1], 2)),
                (round(P_end_corner[0], 2), -round(P_end_corner[1], 2)),
                (C4_od[0], -C4_od[1])
            ]
        if right:
            right_segment = [
                (round(P_start_corner[0], 2), -round(P_start_corner[1], 2)), 
                (C2_od[0], -C2_od[1]),
                (C3_od[0], -C3_od[1]),
                (round(P_end_corner[0], 2), -round(P_end_corner[1], 2))
            ]
        
        return left_segment, right_segment

    def find_neighbour_segments(self, best_match, distance_each_side):
        road_id = best_match['road_id']
        s_start_point = best_match['s']
        
        road = next((r for r in self.road_data if r['id'] == road_id), None)
        if road is None:
            return []

        left, right = self._check_lanes(road)

        geometries = road['geometry']
        lane_width = self._get_lane_boundaries(road)
        
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
            left_segment, right_segment = self._get_geometry_corners(geom, lane_width, left, right)
            # if left_segment and right_segment:
            all_segments_corners.append((left_segment, right_segment))
                
        # The final list structure is: [(left_segment, right_segment), (left_segment, right_segment), (left_segment, right_segment), ...]
        #left segment structure: [(corner1_x, corner1_y), (corner2_x, corner2_y), ... corner4]
        road_segments = {
            'road_id': best_match['road_id'],
            'road_segments': all_segments_corners
        }
        
        return road_segments

    def find_closest_road(self, P):
        best_match = {
            'distance': float('inf'),
            'type': None,
            'road_id': None,
            'junction': None,
            's': None,
            't': None,
            'P_ref_x': None,
            'P_ref_y': None,
            'hdg': None,
            'segment_corners': None 
        }

        P_x, P_y = P

        for road in self.road_data:
            road_id = road['id']
            junction = road['junction']
            
            # Get the half-width magnitude of the road (from your existing _get_road_boundaries)
            # This width is used to define the lateral extent of the bounding box.
            lane_width = self._get_lane_boundaries(road) 

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
                    
                    # We must recalculate the true heading and tangent at the closest point P_ref
                    # t_norm here is the parameter 'u'
                    (_, _), (Tx_at_ref, Ty_at_ref) = self._get_paramPoly3_point_and_tangent(geom, t_norm)
                    
                    # Recalculate the heading at P_ref for the final result
                    hdg = math.atan2(Ty_at_ref, Tx_at_ref)
                    length = geom['length'] # Used for s calculation

                    roadtype = 'parampoly'

                distance = math.sqrt((P_x - P_ref[0])**2 + (P_y - P_ref[1])**2)
                # print("road id: ", road_id)
                # print("distance: ", distance)

                if distance <= best_match['distance']:
                    # print("found a road")

                    s_current_geom = t_norm * length
                    s_total = geom['s'] + s_current_geom 
                    
                    Tx = math.cos(hdg) # Tangent vector X
                    Ty = math.sin(hdg) # Tangent vector Y
                    Vx = P[0] - P_ref[0]       # Vector from P_ref to P (Lateral X)
                    Vy = P[1] - P_ref[1]       # Vector from P_ref to P (Lateral Y)
                          
                    signed_distance = distance * math.copysign(1, -(Tx * Vy - Ty * Vx))
                    # print("signed distance: ", signed_distance)
                    
                    # Update best match only if the projection is longitudinally valid
                    if 0 <= t_norm <= 1:
                        road = next((r for r in self.road_data if r['id'] == road_id), None)
                        if road is None:
                            return []

                        left, right = self._check_lanes(road)
                        corners = self._get_geometry_corners(geom, lane_width, left, right)
                        
                        best_match['type'] = roadtype
                        best_match['distance'] = distance
                        best_match['road_id'] = road_id
                        best_match['junction'] = junction
                        best_match['s'] = s_total
                        best_match['t'] = signed_distance
                        best_match['P_ref_x'] = P_ref[0]    #in opendrive coordinate
                        best_match['P_ref_y'] = P_ref[1]    #in opendrive coordinate
                        if signed_distance < 0:
                            best_match['hdg'] = hdg + math.pi
                        else:
                            best_match['hdg'] = hdg
                        best_match['segment_corners'] = corners #in carla coordinate
                    
        return best_match
    
    def check_point_on_road(self, P_x, P_y, distance_each_side):
        P_opendrive = -P_y
        P = (P_x, P_opendrive)
        
        best_match = self.find_closest_road(P)

        road_segments = self.find_neighbour_segments(best_match, distance_each_side)

        return road_segments
    
