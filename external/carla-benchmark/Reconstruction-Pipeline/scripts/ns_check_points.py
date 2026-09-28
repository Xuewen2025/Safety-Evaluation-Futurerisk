import math
import xml.etree.ElementTree as ET


class OpenDRIVEChecker:

    def __init__(self, xodr_data):
        self.map_data = self._parse_opendrive_data(xodr_data)

    def _parse_geometry(self, geom_element):
            geom_type_element = list(geom_element)[0]
            
            base_data = {
                's': float(geom_element.get('s')),
                'x': float(geom_element.get('x')),
                'y': float(geom_element.get('y')),
                'hdg': float(geom_element.get('hdg')),
                'length': float(geom_element.get('length')),
                'type': geom_type_element.tag
            }

            if geom_type_element.tag == 'line':
                return base_data

            elif geom_type_element.tag == 'paramPoly3':
                poly_data = {
                    'aU': float(geom_type_element.get('aU')),
                    'bU': float(geom_type_element.get('bU')),
                    'cU': float(geom_type_element.get('cU')),
                    'dU': float(geom_type_element.get('dU')),
                    'aV': float(geom_type_element.get('aV')),
                    'bV': float(geom_type_element.get('bV')),
                    'cV': float(geom_type_element.get('cV')),
                    'dV': float(geom_type_element.get('dV')),
                    'pRange': geom_type_element.get('pRange', 'arcLength')
                }
                return {**base_data, **poly_data}

            return None

    def _parse_lanes(self, lane_element):
        width_element = lane_element.find('width')
        # We only extract width 'a' (constant width) for simplicity
        width_a = float(width_element.get('a')) if width_element is not None else 0.0
        
        return {
            'id': int(lane_element.get('id')),
            'type': lane_element.get('type'),
            'width_a': width_a
        }

    def _parse_opendrive_data(self, xodr_xml_string):
        parsed_roads = []
        try:
            root = ET.fromstring(xodr_xml_string)
            
            for road_element in root.findall('road'):
                road_data = {
                    'id': int(road_element.get('id')),
                    'name': road_element.get('name'),
                    'geometry': [],
                    'lane_section': []
                }
                
                # 1. Parse Geometry (planView)
                for geom_element in road_element.findall('planView/geometry'):
                    geom = self._parse_geometry(geom_element)
                    if geom:
                        road_data['geometry'].append(geom)

                # 2. Parse Lane Sections
                for section_element in road_element.findall('lanes/laneSection'):
                    section_data = {'s_offset': float(section_element.get('s')), 'right_lanes': [], 'left_lanes': []}

                    # Parse Right Lanes (Negative IDs)
                    for lane_element in section_element.findall('right/lane'):
                        section_data['right_lanes'].append(self._parse_lanes(lane_element))

                    # Parse Left Lanes (Positive IDs)
                    for lane_element in section_element.findall('left/lane'):
                        section_data['left_lanes'].append(self._parse_lanes(lane_element))

                    road_data['lane_section'].append(section_data)

                parsed_roads.append(road_data)
                
            
            # print(f"SUCCESS: Parsed {len(parsed_roads)} roads from XODR data.")
            return parsed_roads

        except ET.ParseError as e:
            print(f"ERROR: Failed to parse XML string: {e}")
            return []

    def _calculate_closest_point_on_line(self, P, P1, P2):
        V = (P2[0] - P1[0], P2[1] - P1[1])  # Segment vector
        W = (P[0] - P1[0], P[1] - P1[1])  # Vector from P1 to P

        # Calculate the projection parameter 'c1' (Dot product W . V)
        c1 = W[0] * V[0] + W[1] * V[1] 
        c2 = V[0] * V[0] + V[1] * V[1]  # Squared length of segment (V . V)
        

        if c1 <= 0:
            # Projection falls before P1, so P1 is the closest point.
            return P1, 0.0

        t = c1 / c2

        if c2 == 0:
            # P1 and P2 are the same point (zero length), return P1
            return P1, t

        if c1 >= c2:
            # Projection falls after P2, so P2 is the closest point.
            return P2, t

        # Projection falls within P1-P2 segment
        Px = P1[0] + t * V[0]
        Py = P1[1] + t * V[1]
        
        return (Px, Py), t

    def _get_paramPoly3_point_and_tangent(self, geom, u):
        """Calculates (x, y) coordinates, and tangent components (dx/du, dy/du) 
           for a given parameter u on the paramPoly3 geometry."""
        
        # Polynomials for the U and V components of the curve in the local system (u, v)
        # U(u) = aU + bU*u + cU*u^2 + dU*u^3
        U = geom['aU'] + geom['bU']*u + geom['cU']*(u**2) + geom['dU']*(u**3)
        V = geom['aV'] + geom['bV']*u + geom['cV']*(u**2) + geom['dV']*(u**3)
        
        # Derivatives for the local tangent vector (dU/du, dV/du)
        # U'(u) = bU + 2*cU*u + 3*dU*u^2
        dU_du = geom['bU'] + 2*geom['cU']*u + 3*geom['dU']*(u**2)
        # V'(u) = bV + 2*cV*u + 3*dV*u^2
        dV_du = geom['bV'] + 2*geom['cV']*u + 3*geom['dV']*(u**2)
        
        # Rotation matrix (from local u/v to global x/y)
        hdg = geom['hdg']
        cos_hdg = math.cos(hdg)
        sin_hdg = math.sin(hdg)
        
        # Global coordinates (X, Y)
        X = geom['x'] + U * cos_hdg - V * sin_hdg
        Y = geom['y'] + U * sin_hdg + V * cos_hdg
        
        # Global Tangent (T_x, T_y) from local derivatives
        Tx = dU_du * cos_hdg - dV_du * sin_hdg
        Ty = dU_du * sin_hdg + dV_du * cos_hdg

        return (X, Y), (Tx, Ty)

    def _calculate_closest_point_on_paramPoly3(self, P, geom, max_iterations=10, tolerance=1e-5):
        
        P_x, P_y = P
        
        # Initial guess for the parameter u (midpoint of the segment)
        # OpenDRIVE pRange="normalized" means u is between 0 and 1.
        u = 0.5 
        
        for _ in range(max_iterations):
            # 1. Get current point (X, Y) and tangent (Tx, Ty) on the curve at u
            (X, Y), (Tx, Ty) = self._get_paramPoly3_point_and_tangent(geom, u)
            
            # 2. Vector from curve point to query point (V_x, V_y)
            V_x = X - P_x
            V_y = Y - P_y
            
            # 3. Objective function (distance squared derivative w.r.t u): F(u) = (X-Px)*Tx + (Y-Py)*Ty = 0
            # This is the dot product of the vector V (from P to X, Y) and the tangent T.
            F_u = V_x * Tx + V_y * Ty
            
            # 4. Check for convergence
            if abs(F_u) < tolerance:
                # Closest point found, normalize the distance along the length
                s_norm = u # For normalized pRange, s_norm is u
                return (X, Y), s_norm 
            
            # 5. Calculate derivative of F(u): F'(u) (This requires the second derivatives of X and Y w.r.t u)
            # F'(u) = Tx^2 + Ty^2 + (X-Px)*Tx' + (Y-Py)*Ty'
            
            # Since calculating Tx' and Ty' from the rotation is very complex, 
            # we use a simpler approach: the magnitude of the tangent vector squared (||T||^2) as the denominator 
            # for a simple update, often used for path following:
            F_prime_approx = Tx**2 + Ty**2
            
            if F_prime_approx < tolerance: # Flat spot or zero length
                break 

            # 6. Newton's step: u_next = u - F(u) / F'(u)
            delta_u = F_u / F_prime_approx
            u_next = u - delta_u
            
            # 7. Clamp u to the segment boundaries [0, 1]
            u = max(0.0, min(1.0, u_next))
            
            # Check for convergence in u
            if abs(delta_u) < tolerance:
                (X, Y), _ = self._get_paramPoly3_point_and_tangent(geom, u)
                s_norm = u
                return (X, Y), s_norm
        
        # If max iterations reached, return the last calculated point/parameter
        (X, Y), _ = self._get_paramPoly3_point_and_tangent(geom, u)
        s_norm = u
        return (X, Y), s_norm

    def _get_lane_width(self, road):
        lane_width = 0.0

        # In this simplified model, we only use the laneSection at s=0
        lane_section = road['lane_section'][0] 
        
        for lane in lane_section.get('right_lanes', []):
            if(lane['type'] == "driving"):
                lane_width += lane['width_a']

        return lane_width

    def _is_direction_correct(self, spawn_P, crash_P, lane_hdg_odr):
        """
        Checks if the lane heading aligns with the vector from Spawn to Crash.
        """
        # 1. Vector from Spawn to Crash (in CARLA coordinates)
        vec_crash_x = crash_P[0] - spawn_P[0]
        vec_crash_y = crash_P[1] - spawn_P[1]

        # 2. Vector of Lane Heading (converted to CARLA coordinates)
        # OpenDRIVE +Y is CARLA -Y. So a heading of theta in ODR is (cos(theta), -sin(theta)) in CARLA.
        vec_lane_x = math.cos(lane_hdg_odr)
        vec_lane_y = -math.sin(lane_hdg_odr)

        # 3. Dot Product
        dot_product = (vec_crash_x * vec_lane_x) + (vec_crash_y * vec_lane_y)

        # Positive means aligned (< 90 deg diff). Negative means opposed (> 90 deg diff).
        return dot_product > 0

    def _mirror_point_to_opposite_lane(self, spawn_P, best_match):
        """
        Flips the point across the road's center reference line.
        Used when the car is in the oncoming traffic lane.
        """
        # Current Point in OpenDRIVE space (flip Y)
        P_curr_x = spawn_P[0]
        P_curr_y = -spawn_P[1]

        # Reference Point (Center line) in OpenDRIVE space
        P_ref_x = best_match['P_ref_x']
        P_ref_y = best_match['P_ref_y']

        # MATH: Mirror P_curr across P_ref
        # Vector Center->Car = P_curr - P_ref
        # New Vector Center->NewCar = -(P_curr - P_ref)
        # NewCar = P_ref - (P_curr - P_ref) = 2*P_ref - P_curr
        P_new_x = 2 * P_ref_x - P_curr_x
        P_new_y = 2 * P_ref_y - P_curr_y

        # Convert back to CARLA space (flip Y back)
        carla_new_x = P_new_x
        carla_new_y = -P_new_y

        # Flip Heading 180 degrees (Pi radians)
        new_hdg = best_match['hdg'] + math.pi
        
        # Normalize heading to -pi to pi
        if new_hdg > math.pi: new_hdg -= 2 * math.pi

        return carla_new_x, carla_new_y, new_hdg
    
    def find_closest_road(self, P):
        best_match = {
            'distance': float('inf'),
            'type': None,
            'road_id': None,
            's': None,
            't': None,
            'P_ref_x': None,
            'P_ref_y': None,
            'hdg': None,
            'both_lanes': None,
            'segment_corners': None 
        }

        P_x, P_y = P # Unpack query point

        for road in self.map_data:
            road_id = road['id']
            
            # Get the half-width magnitude of the road (from your existing _get_road_boundaries)
            # This width is used to define the lateral extent of the bounding box.
            lane_width = self._get_lane_width(road) 

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
                    lanes = road['lane_section']
                    both = -1
                    if lanes[0]['right_lanes'] and lanes[0]['left_lanes']:
                        both = 1
                    else:
                        both = 0

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
                        
                        best_match['type'] = roadtype
                        best_match['distance'] = distance
                        best_match['road_id'] = road_id
                        best_match['s'] = s_total
                        best_match['t'] = signed_distance
                        best_match['P_ref_x'] = P_ref[0]    #in opendrive coordinate
                        best_match['P_ref_y'] = P_ref[1]    #in opendrive coordinate
                        best_match['both_lanes'] = both
                        if signed_distance < 0:
                            best_match['hdg'] = hdg + math.pi
                        else:
                            best_match['hdg'] = hdg
                        
        return best_match

    def move_point_to_road(self, P, best_match):
        
        if best_match['road_id'] is not None:
            road = next(r for r in self.map_data if r['id'] == best_match['road_id'])
            lane_width = self._get_lane_width(road)
        
        P_x = P[0]
        P_y = P[1]
        distance_snapped = lane_width/2.0
        
        P_ref_x = best_match['P_ref_x']
        P_ref_y = best_match['P_ref_y']
        hdg_ref = best_match['hdg']
        
        P_snapped_x = P_x + distance_snapped * (-math.sin(hdg_ref))
        P_snapped_y = P_y + distance_snapped * (math.cos(hdg_ref))
        
        carla_snapped_x = P_snapped_x
        carla_snapped_y = -P_snapped_y # Apply the Y-flip back to CARLA standard

        return carla_snapped_x, carla_snapped_y, hdg_ref
    
    def move_point_from_road(self, best_match):

        # Clamp the current lateral offset 't' to the valid road boundaries
        distance_snapped = best_match['distance'] - 2
        
        # 3. Calculate the snapped coordinates (P_snapped) in OpenDRIVE space
        P_ref_x = best_match['P_ref_x']
        P_ref_y = best_match['P_ref_y']
        hdg_ref = best_match['hdg']
        # print("hdg_ref: ", hdg_ref)
        # The direction perpendicular to the heading (Normal Vector) points left (positive t)
        # Normal_x = -sin(hdg_ref)
        # Normal_y = cos(hdg_ref)

        # Snapped point coordinates (OpenDRIVE space)
        P_snapped_x = P_ref_x + distance_snapped * (-math.sin(hdg_ref))
        P_snapped_y = P_ref_y + distance_snapped * (math.cos(hdg_ref))
        
        # 4. Convert back to CARLA coordinates
        carla_snapped_x = P_snapped_x
        carla_snapped_y = -P_snapped_y # Apply the Y-flip back to CARLA standard

        return carla_snapped_x, carla_snapped_y, hdg_ref

    def check_point_on_road(self, spawn_P, crash_P, snap):
        P_x = spawn_P[0]
        P_y = spawn_P[1]
        P_opendrive = -P_y
        P = (P_x, P_opendrive)
        
        best_match = self.find_closest_road(P)
        if best_match['road_id'] is None:
            return False, None, None, None, None

        road = next(r for r in self.map_data if r['id'] == best_match['road_id'])
        lane_width = self._get_lane_width(road)

        final_x, final_y, final_hdg = P_x, P_y, best_match['hdg']
        
        if snap:
            if 1 < best_match['distance'] < lane_width - 1:
                final_x, final_y, final_hdg = P_x, P_y, best_match['hdg']
            
            elif best_match['distance'] < 1:
                # print("distance:", best_match['distance'])
                print("Too close to the center boundary. New point calculating")
                final_x, final_y, final_hdg = self.move_point_from_road(best_match)
                is_modified = True
            
            elif best_match['distance'] > lane_width - 1:
                # print("distance:", best_match['distance'])
                print("Not on road. New Point Calculating...")
                final_x, final_y, final_hdg = self.move_point_to_road(P, best_match)
            
            if crash_P is not None:
                
                if not self._is_direction_correct((final_x, final_y), crash_P, final_hdg):
                    print(f"Wrong Direction Detected!")
                    if best_match['both_lanes'] == 0:
                        final_hdg += math.pi

                        final_hdg = (final_hdg + math.pi) % (2 * math.pi) - math.pi
                    
                    else:
                        final_x, final_y, final_hdg = self._mirror_point_to_opposite_lane((final_x, final_y), best_match)

            return True, final_x, final_y, best_match['distance'], final_hdg
        
        else:
            if best_match['distance'] < lane_width:
                return True, final_x, final_y, best_match['distance'], final_hdg
            else:
                return False, None, None, best_match['distance'], None


def check_and_get_direction(spawn_P, crash_P, xodrPath, snap):
    with open(xodrPath, 'r') as file:
        xodr_content = file.read()

    checker = OpenDRIVEChecker(xodr_content)

    is_on, P_x, P_y, distance, hdg = checker.check_point_on_road(spawn_P, crash_P, snap)

    return is_on, P_x, P_y, distance, hdg  


    
