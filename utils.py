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



class Edge:
    start = None
    end = None

    vec = None

    boundary = None
    ending_sweep = None

    site1_id = None
    site2_id = None

    def __init__(self, start, vec, site1_id=None, site2_id=None):
        self.start = start
        self.vec = vec
        self.site1_id = site1_id
        self.site2_id = site2_id

    def __str__(self):
        return f"<S {self.start} | V {self.vec}>"


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


def edgeIntersectBoundingBox(edge, bounding_box):
    return rayIntersectBoundingBox(npa(edge.start), edge.vec, bounding_box)

def rayIntersectBoundingBox(origin, direction, bounding_box, only_closest=False, max_ray_length=None):

    closest_intersect = (None, None)
    closest_u = None

    for line in bounding_box:
        u, v, intersection = rayIntersectLineSegment(origin, direction, line)

        if intersection is not None and (max_ray_length is None or u < max_ray_length):
            if closest_u is None or u < closest_u:
                closest_u = u
                closest_intersect = (intersection, line)

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

    return [
        (tl, tr-tl),
        (tr, br-tr),
        (br, bl-br),
        (bl, tl-bl),
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


if __name__ == '__main__':
    test_array = [(1,2),(3,4),(5.2,6),(5.2,7000),(5.3,8),(9,10)]
    # test_array = []
    k = KeyList(test_array, key=lambda x: x[0])
    print(bisect.bisect_right(k, 9))


    # test_array.append((2,0))
    # print(bisect.bisect_right(k, 6))
    # a1 = Arc(pt((0, 1)))
    # a2 = Arc(pt((2, -1)))
    # end = Endpoint(None, a1, a2)
    # print(end.calculateX(-1))