import math
import xml.etree.ElementTree as ET

class RoadExtraction:

    def __init__(self, xodr_data):
        self.road_data, self.junction_data = self._parse_opendrive_data(xodr_data)

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
        parsed_junctions = []

        try:
            root = ET.fromstring(xodr_xml_string)
            
            for road_element in root.findall('road'):
                road_data = {
                    'id': int(road_element.get('id')),
                    'name': road_element.get('name'),
                    'length': road_element.get('length'),
                    'junction': int(road_element.get('junction')),
                    'geometry': [],
                    'lane_section': [],
                    'predecessor': (),
                    'successor': ()
                }

                for connection in road_element.findall('link/predecessor'):
                    predecessor_type = connection.get('elementType')
                    predecessor_id = int(connection.get('elementId'))
                    pred = (predecessor_type, predecessor_id)
                    road_data['predecessor'] = pred

                for connection in road_element.findall('link/successor'):
                    successor_type = connection.get('elementType')
                    successor_id = int(connection.get('elementId'))
                    succ = (successor_type, successor_id)
                    road_data['successor'] = succ
                
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

            for junction_element in root.findall('junction'):
                junction_data = {
                    'id': int(junction_element.get('id')),
                    'name': junction_element.get('name'),
                    'connections': []
                }
                
                for connection_element in junction_element.findall('connection'):
                    connection_data = {
                        'id': int(connection_element.get('id')),
                        'incomingRoad': int(connection_element.get('incomingRoad')),
                        'connectingRoad': int(connection_element.get('connectingRoad')),
                        'contactPoint': connection_element.get('contactPoint'),
                        'lane_links': []
                    }
                    
                    for lane_link_element in connection_element.findall('laneLink'):
                        connection_data['lane_links'].append({
                            'from': int(lane_link_element.get('from')),
                            'to': int(lane_link_element.get('to'))
                        })
                    
                    junction_data['connections'].append(connection_data)
                
                parsed_junctions.append(junction_data)    
            
            return parsed_roads, parsed_junctions

        except ET.ParseError as e:
            print(f"ERROR: Failed to parse XML string: {e}")
            return []

    def _calculate_closest_point_on_line(self, P, P1, P2):
        V = (P2[0] - P1[0], P2[1] - P1[1])  # Segment vector
        W = (P[0] - P1[0], P[1] - P1[1])  # Vector from P1 to P

        # Calculate the projection parameter 'c1' (Dot product W . V)
        c1 = W[0] * V[0] + W[1] * V[1] 

        if c1 <= 0:
            # Projection falls before P1, so P1 is the closest point.
            return P1, 0.0

        c2 = V[0] * V[0] + V[1] * V[1]  # Squared length of segment (V . V)
        
        if c2 == 0:
            # P1 and P2 are the same point (zero length), return P1
            return P1, 0.0

        if c1 >= c2:
            # Projection falls after P2, so P2 is the closest point.
            return P2, 1.0

        # Projection falls within P1-P2 segment
        t = c1 / c2
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

    def _get_lane_boundaries(self, road):
        lane_width = 0.0

        lane_section = road['lane_section'][0] 
        
        for lane in lane_section.get('right_lanes', []):
            if(lane['type'] == "driving"):
                lane_width += lane['width_a']

        return lane_width


