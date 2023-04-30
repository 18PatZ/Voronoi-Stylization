import bisect
import math
import heapq
import itertools

import numpy as np

class KeyList(object):
    # bisect doesn't accept a key function before 3.10,
    # so we build the key into our sequence.
    def __init__(self, l, key):
        self.l = l
        self.key = key
    def __len__(self):
        return len(self.l)
    def __getitem__(self, index):
        return self.key(self.l[index])


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


class Point:
    x = 0.0
    y = 0.0

    def __init__(self, x, y):
       self.x = x
       self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"


class Arc:
    focus = None
    id = -1

    site_id = None
    
    prev_arc = None
    next_arc = None
    
    prev_endpoint = None
    next_endpoint = None

    def __init__(self, focus, site_id=None, id=-1, prev_arc=None, next_arc=None, prev_endpoint=None, next_endpoint=None):
        self.focus = focus
        self.site_id = site_id
        self.id = id
        self.prev_arc = prev_arc
        self.next_arc = next_arc
        self.prev_endpoint = prev_endpoint
        self.next_endpoint = next_endpoint


def order2(a, b):
    if a <= b:
        return (a, b)
    return (b, a)

def order3(a, b, c):
    if a <= b:
        if a <= c: # a < b c
            return (a,) + order2(b, c)
        else: # c < a < b
            return (c, a, b)
    else:
        if b <= c: # b < a c
            return (b,) + order2(a, c)
        else: # c < b < a
            return (c, b, a)


class Face:
    site = None
    id = None
    centroid = None
    
    edges = []

    def __init__(self, site=None, id=None, edges=None):
        self.site = site
        self.id = id
        self.edges = edges
        
        if len(self.edges) > 0:
            self.centroid = np.array([0., 0.])
            for edge in self.edges:
                self.centroid += edge.start + edge.end
            self.centroid /= (2 * len(self.edges))
        else:
            self.centroid = None
    

class Triangle:
    sites = []
    vertices = []
    inner = True

    def __init__(self, sites, vertices, inner=True):
        self.sites = sites
        # self.sites.sort()
        # self.sites = tuple(self.sites)
        self.inner = inner

        self.vertices = vertices


class Edge:
    start = None
    end = None

    vec = None

    boundary_start = None
    boundary_end = None
    ending_sweep = None

    site1_id = None
    site2_id = None
    site_ids = []

    def __init__(self, start, vec=None, end=None, site1_id=None, site2_id=None):
        assert(site1_id is not None and site2_id is not None)
        self.start = start
        self.vec = vec
        self.end = end
        if vec is None and end is not None:
            self.vec = normalize(end - start)
        self.site1_id = site1_id
        self.site2_id = site2_id
        self.site_ids = order2(self.site1_id, self.site2_id)

    def __str__(self):
        return f"<S {self.start} | V {self.vec}>"

    def copy(self, end_value=None):
        clone = Edge(start=self.start, vec=self.vec, site1_id=self.site1_id, site2_id=self.site2_id)
        clone.site_ids = self.site_ids
        clone.end = self.end if self.end is not None else end_value
        clone.boundary_start = self.boundary_start
        clone.boundary_end = self.boundary_end
        clone.ending_sweep = clone.ending_sweep
        return clone

    def np_copy(self, end_value=None):
        clone = self.copy(end_value)
        if self.start is not None:
            clone.start = npa(self.start)
        if self.end is not None:
            clone.end = npa(self.end)
        return clone

    def same_ids(self, other_edge):
        return self.site_ids[0] == other_edge.site_ids[0] and self.site_ids[1] == other_edge.site_ids[1]


def arcY(arc, sweepY, x):
    f = arc.focus
    
    # if f.y == sweepY:
    #     f = focus2 # otherwise 0 division

    assert f.y != sweepY
    
    return 1.0 / (2 * (f.y - sweepY)) * (x - f.x)**2 + (f.y + sweepY) / 2 # formula for parabola with focus (f.x, f.y) and directrix sweepY


def arcIntersect(arc1, arc2, sweepY):
    focus1 = arc1.focus
    focus2 = arc2.focus

    if focus1.y == focus2.y:
        px = (focus1.x + focus2.x) / 2.0 # on same level - in middle
    elif focus2.y == sweepY:
        px = focus2.x
    elif focus1.y == sweepY:
        px = focus1.x
    else:
        # use quadratic formula
        z0 = 2.0 * (focus1.y - sweepY)
        z1 = 2.0 * (focus2.y - sweepY)

        a = 1.0/z0 - 1.0/z1
        b = -2.0 * (focus1.x/z0 - focus2.x/z1)
        c = 1.0 * (focus1.x**2 + focus1.y**2 - sweepY**2) / z0 - 1.0 * (focus2.x**2 + focus2.y**2 - sweepY**2) / z1

        if b*b - 4*a*c < 0:
            print("ERROR: domain error", focus1, focus2, sweepY, b, b*b,4*a*c)
            assert b*b - 4*a*c >= 0

        px = 1.0 / (2*a) * (-b+math.sqrt(b*b - 4*a*c))


        # 1.0 / (2 * (f1.y - sweepY)) * (x - f1.x)**2 + (f1.y + sweepY) / 2 - 1.0 / (2 * (f2.y - sweepY)) * (x - f2.x)**2 - (f2.y + sweepY) / 2 = 0
        # 1/z0 * (x - f1.x)**2 - 1/z1 * (x - f2.x)**2 + (f1.y + sweepY) / 2 - (f2.y + sweepY) / 2 = 0
        # 1/z0 * (x^2 - 2 x f1.x + f1.x^2) - 1/z1 * (x^2 - 2 f2.x + f2.x^2) + (f1.y + sweepY) / 2 - (f2.y + sweepY) / 2
        
    
    # py = 1.0 * (p.y**2 + (p.x-px)**2 - sweepY**2) / (2*p.y-2*sweepY)
    # y = 1.0 / (2(yf - yd) * (x-xf)^2 + (yf+yd)/2)
    # y = 1.0 / 2(yf - yd) * (x^2-2xf + xf^2) + (yf+yd)/2)
    # y = 1.0 / 2(yf - yd) * ((x-xf)^2 + yf^2-yd^2))

    # py = 1.0 / (2 * (f.y - sweepY)) * (px - f.x)**2 + (f.y + sweepY) / 2 # formula for parabola with focus (f.x, f.y) and directrix sweepY

    if arc1.focus.y == sweepY:
        py = None
    else:
        py = arcY(arc1, sweepY, px)
        
    point = Point(px, py)

    return point

def rrIntersect(o1, d1, o2, d2):
    ###
    dx = o2[0] - o1[0]
    dy = o2[1] - o1[1]

    det = d2[0] * d1[1] - d2[1] * d1[0]

    if det == 0:
        return -1, -1, None

    u = (dy * d2[0] - dx * d2[1]) / det
    v = (dy * d1[0] - dx * d1[1]) / det
    ###

    if u < 0 or v < 0:
        return u, v, None

    point = o1 + u * d1
    return u, v, point


def edgeIntersect(edge1, edge2):
    o1 = npa(edge1.start)
    d1 = edge1.vec

    o2 = npa(edge2.start)
    d2 = edge2.vec

    _, _, point = rrIntersect(o1, d1, o2, d2)
    return point



def edgeIntersectLineSegment(edge1, line):
    _, _, point = rayIntersectLineSegment(npa(edge1.start), edge1.vec, line)
    return point

def rayIntersectLineSegment(origin, direction, line):
    o1 = origin
    d1 = direction

    o2 = line[0]
    d2 = normalize(line[1])
    
    line_length = np.linalg.norm(line[1])

    u, v, point = rrIntersect(o1, d1, o2, d2)
    if v > line_length:
        return u, v, None

    return u, v, point


def edgeIntersectBoundingBox(edge, bounding_box, only_closest=False, max_ray_length=None):
    return rayIntersectBoundingBox(npa(edge.start), edge.vec, bounding_box, only_closest, max_ray_length)

def rayIntersectBoundingBox(origin, direction, bounding_box, only_closest=False, max_ray_length=None):

    closest_intersect = (None, None)
    closest_u = None

    for line_index in range(len(bounding_box)):
        line = bounding_box[line_index]
        u, v, intersection = rayIntersectLineSegment(origin, direction, line)

        if intersection is not None and (max_ray_length is None or u < max_ray_length):
            if closest_u is None or u < closest_u:
                closest_u = u
                closest_intersect = (intersection, (line, line_index))

                if not only_closest:
                    return closest_intersect
    
    return closest_intersect

def isPointInPolygon(point, polygon): # assumes vertices are ordered

    p = npa(point)
    sign = None

    for line in polygon:
        point_to_start = line[0] - p
        cross = np.cross([point_to_start[0], point_to_start[1], 0], [line[1][0], line[1][1], 0])
        s = np.sign(cross[2]) # Z direction
        if sign is None:
            sign = s
        elif s != sign: #flipped signs, that means we're not in polygon
            return False

    return True

def isPointInFace(point, face): # assumes vertices are ordered

    p = point
    sign = None

    for edge in face.edges:
        point_to_start = edge.start - p
        disp = edge.end - edge.start
        cross = np.cross([point_to_start[0], point_to_start[1], 0], [disp[0], disp[1], 0])
        s = np.sign(cross[2]) # Z direction
        if sign is None:
            sign = s
        elif s != sign: #flipped signs, that means we're not in polygon
            return False

    return True


def get_image_bounding_box(img):
    w = img.shape[1]
    h = img.shape[0]
    
    tl = np.array([0, 0])
    tr = np.array([w, 0])
    br = np.array([w, -h])
    bl = np.array([0, -h])

    # =====>||
    # /\    ||
    # ||    \/
    # ||<=====
    # return [
    #     (tl, tr-tl),
    #     (tr, br-tr),
    #     (br, bl-br),
    #     (bl, tl-bl),
    # ]

    # we want counterclockwise actually
    # ||<=====
    # ||    /\
    # \/    ||
    # =====>||
    return [
        (tl, bl-tl),
        (bl, br-bl),
        (br, tr-br),
        (tr, tl-tr),
    ]
    



class Endpoint:
    edge = None
    left_arc = None
    right_arc = None

    def __init__(self, edge, left_arc=None, right_arc=None):
        self.edge = edge
        self.left_arc = left_arc
        self.right_arc = right_arc

    def calculateX(self, sweepY, outputY=False):
        # start = self.edge.start
        # vec = self.edge.vec
        if self.left_arc is not None and self.right_arc is not None:
            intersection = arcIntersect(self.left_arc, self.right_arc, sweepY)

            if not outputY:
                return intersection.x
            else:
                return intersection.x, intersection.y
        else:
            return -1000 if not outputY else (-1000, 0)




class Beachline:
    endpoints = []
    
    arc_counter = 0
    
    def _insert(self, endpoint):
        self.endpoints.append(endpoint)

    def _insert_at(self, index, endpoint):
        self.endpoints.insert(index, endpoint)
    
    def _search(self, sweep, value):
        key_list = KeyList(self.endpoints, key=lambda endpoint: endpoint.calculateX(sweep))
        beach_index = bisect.bisect_right(key_list, value)
        return beach_index

    
    def nextElement(self, endpoint):
        ind = self.endpoints.index(endpoint)
        if ind < 0 or ind >= len(self.endpoints):
            return None
        return self.endpoints[ind + 1]

    
    def remove(self, arc, intersection):
        endpoint_left = arc.prev_endpoint
        endpoint_right = arc.next_endpoint

        edge_left = endpoint_left.edge
        edge_right = endpoint_right.edge

        arc_left = endpoint_left.left_arc
        arc_right = endpoint_right.right_arc

        endpoint_left.right_arc = arc_right

        arc_left.next_arc = arc_right
        arc_right.prev_arc = arc_left

        arc_right.prev_endpoint = endpoint_left

        self.endpoints.remove(endpoint_right)

        # new edge
        edge_start = (intersection.x, intersection.y)

        connecting_vec = npa(arc_left.focus) - npa(arc_right.focus) # right to left
        
        tangent = normalize(connecting_vec)
        orthogonal = np.cross(np.array([tangent[0], tangent[1], 0]), np.array([0, 0, 1]))
        
        up_vec = normalize(orthogonal[:2])
        down_vec = - up_vec

        new_edge = Edge(pt(edge_start), down_vec, site1_id=arc_left.site_id, site2_id=arc_right.site_id)
        endpoint_left.edge = new_edge

        # complete old edges
        edge_left.end = intersection
        edge_right.end = intersection

        return ([arc_left, arc_right], [edge_left, edge_right])


    def insert(self, site, site_id):
        sweepY = site.y
        
        new_arc = Arc(focus=site, site_id=site_id, id=self.arc_counter)
        self.arc_counter += 1

        if len(self.endpoints) == 0: # this is first arc
            self._insert(Endpoint(edge=None, left_arc=None, right_arc=new_arc)) # we indicate a single arc with None for left arc

            return [new_arc]

        else:
            beach_index = 0

            intersect_x = site.x

            if len(self.endpoints) > 1:
                beach_index = self._search(sweep = sweepY, value = intersect_x) - 1
                if beach_index < 0:
                    beach_index = 0
            
            endpoint = self.endpoints[beach_index]
            existing_arc = endpoint.right_arc

            # split arc into two pieces - we're adding two endpoints. Also creating two half edges, perpendicular to vector connecting sites
            connecting_vec = npa(existing_arc.focus) - npa(new_arc.focus) # new to old
            # edge_center = npa(new_arc.focus) + connecting_vec / 2.0

            colinear = False

            if existing_arc.focus.y == sweepY: # both are on the same y-value!
                intersect_y = 100000
                intersect_x = (existing_arc.focus.x + new_arc.focus.x) / 2.0 # bisector is in middle
                colinear = True
            else:
                intersect_y = arcY(existing_arc, sweepY, x = site.x)

            edge_start = (intersect_x, intersect_y)

            tangent = normalize(connecting_vec)
            up_vec = np.array([0, 0, 1])
            orthogonal = np.cross(np.array([tangent[0], tangent[1], 0]), up_vec)
            
            right_vec = normalize(orthogonal[:2])
            left_vec = - right_vec

            edge_left = Edge(pt(edge_start), left_vec, site1_id=existing_arc.site_id, site2_id=new_arc.site_id)
            edge_right = Edge(pt(edge_start), right_vec, site1_id=new_arc.site_id, site2_id=existing_arc.site_id)

            if not colinear:
                arc_left = Arc(focus=existing_arc.focus, site_id=existing_arc.site_id, id=self.arc_counter, prev_arc=existing_arc.prev_arc, next_arc=new_arc)
                self.arc_counter += 1

                existing_arc.prev_arc = new_arc # make existing arc the new right arc split, so we don't have to modify existing's right endpoint
                
                new_endpoint_left = Endpoint(edge=edge_left, left_arc=arc_left, right_arc=new_arc)
                new_endpoint_right = Endpoint(edge=edge_right, left_arc=new_arc, right_arc=existing_arc)

                arc_left.prev_endpoint = existing_arc.prev_endpoint
                arc_left.next_endpoint = new_endpoint_left

                new_arc.prev_endpoint = new_endpoint_left
                new_arc.next_endpoint = new_endpoint_right

                existing_arc.prev_endpoint = new_endpoint_right
                

                endpoint.right_arc = arc_left
                self._insert_at(beach_index + 1, new_endpoint_left)
                self._insert_at(beach_index + 2, new_endpoint_right)

                return [arc_left, new_arc, existing_arc]

            else: #if colinear, only splits into two, not three - new arc is also the rightmost
                existing_arc.prev_arc = new_arc # keep existing arc the left one, since there is no right one

                # edges are vertical for colinear
                edge_down = edge_left if (edge_left.vec[1] < 0) else edge_right
                
                new_endpoint = Endpoint(edge=edge_down, left_arc=existing_arc, right_arc=new_arc)
                
                existing_arc.next_endpoint = new_endpoint

                new_arc.prev_endpoint = new_endpoint
                new_arc.next_endpoint = None

                self._insert_at(beach_index + 1, new_endpoint)

                return [existing_arc, new_arc]

        


     

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def push(self, value, item, type):
        # check for duplicate
        if item in self.entry_finder: return
        count = next(self.counter)
        # use flipped y-coordinate as a primary key (heapq in python is min-heap)
        combined = (value, item, type)
        entry = [-value, count, combined]
        self.entry_finder[combined] = entry
        heapq.heappush(self.pq, entry)

    def remove_entry(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = 'Removed'

    def pop(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item != 'Removed':
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def top(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item != 'Removed':
                del self.entry_finder[item]
                self.push(item)
                return item
        raise KeyError('top from an empty priority queue')

    def empty(self):
        return not self.pq
            

def pt(arr):
    return Point(arr[0], arr[1])

def npa(pt):
    return np.array([pt.x, pt.y])

def npt(tup):
    return np.array([tup[0], tup[1]])

def arrToCvTup(a):
    return (int(a[0]), int(a[1]))

def flipY(pt):
    return np.array([pt[0], -pt[1]])

def conv2d3d(vec2d):
    return np.array([vec2d[0], vec2d[1], 0])

def vecAngle(vec):
    v = normalize(vec)
    c = np.array([1, 0])
    # c = normalize(compare)

    dot = np.dot(v, c) # dot
    det = v[0] * c[1] - c[0] * v[1]      # determinant
    angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    
    angle = -angle * 180 / math.pi
    # if angle < 0:
    #     angle += 360

    return angle

def cosWithHorizontal(vec):
        dot = normalize(vec)[0] # dot with (1, 0) is just x component
        return dot # for sorting by angle, we just need cos(angle), since cos is monotonic from 0 to pi


def vectorPortionInRegion(start, end, bounding_box):    
    start_inside = isPointInPolygon(pt(start), bounding_box)
    end_inside = isPointInPolygon(pt(end), bounding_box)

    if start_inside and end_inside:
        return start, end

    displacement = end-start
    

    if not start_inside:
        intersection, _ = rayIntersectBoundingBox(start, displacement, bounding_box, only_closest=True, max_ray_length=1)
        if intersection is None:
            return None, None

            start = intersection
    if not end_inside:
        nudged_start = start + normalize(displacement) * 0.01 # since start already intersects box, needs a little nudge
        intersection, _ = rayIntersectBoundingBox(nudged_start, displacement, bounding_box, only_closest=True, max_ray_length=1)
        
        if intersection is not None:
            end = intersection
        
    return start, end

if __name__ == '__main__':
    test_array = [(1,2),(3,4),(5.2,6),(5.2,7000),(5.3,8),(9,10)]
    # test_array = []
    k = KeyList(test_array, key=lambda x: x[0])
    print(bisect.bisect_right(k, 9))

    print(rayIntersectLineSegment(np.array([10, 0]), np.array([0, -10]), (np.array([100, 0]), np.array([-100, 0]))))

    # print(vecAngle(np.array([1, 0]), np.array([1, 0])))
    # print(vecAngle(np.array([1, 1]), np.array([1, 0])))
    # print(vecAngle(np.array([-1, 1]), np.array([1, 0])))
    # print(vecAngle(np.array([-1, -1]), np.array([1, 0])))
    # print(vecAngle(np.array([1, -1]), np.array([1, 0])))

    # test_array.append((2,0))
    # print(bisect.bisect_right(k, 6))
    # a1 = Arc(pt((0, 1)))
    # a2 = Arc(pt((2, -1)))
    # end = Endpoint(None, a1, a2)
    # print(end.calculateX(-1))