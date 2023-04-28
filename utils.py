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
    
    prev_arc = None
    next_arc = None
    
    prev_edge = None
    next_edge = None

    def __init__(self, focus, prev_arc=None, next_arc=None):
        self.focus = focus
        self.prev_arc = prev_arc
        self.next_arc = next_arc
        # self.prev_edge = None
        # self.next_edge = None



class Edge:
    start = None
    end = None

    vec = None

    def __init__(self, start, vec):
        self.start = start
        self.vec = vec

    def __str__(self):
        return f"<S {self.start} | V {self.vec}>"


def arcY(arc, sweepY, x):
    f = arc.focus
    
    # if f.y == sweepY:
    #     f = focus2 # otherwise 0 division
    
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

        px = 1.0 / (2*a) * (-b+math.sqrt(b*b - 4*a*c))


        # 1.0 / (2 * (f1.y - sweepY)) * (x - f1.x)**2 + (f1.y + sweepY) / 2 - 1.0 / (2 * (f2.y - sweepY)) * (x - f2.x)**2 - (f2.y + sweepY) / 2 = 0
        # 1/z0 * (x - f1.x)**2 - 1/z1 * (x - f2.x)**2 + (f1.y + sweepY) / 2 - (f2.y + sweepY) / 2 = 0
        # 1/z0 * (x^2 - 2 x f1.x + f1.x^2) - 1/z1 * (x^2 - 2 f2.x + f2.x^2) + (f1.y + sweepY) / 2 - (f2.y + sweepY) / 2
        
    
    # py = 1.0 * (p.y**2 + (p.x-px)**2 - sweepY**2) / (2*p.y-2*sweepY)
    # y = 1.0 / (2(yf - yd) * (x-xf)^2 + (yf+yd)/2)
    # y = 1.0 / 2(yf - yd) * (x^2-2xf + xf^2) + (yf+yd)/2)
    # y = 1.0 / 2(yf - yd) * ((x-xf)^2 + yf^2-yd^2))

    # py = 1.0 / (2 * (f.y - sweepY)) * (px - f.x)**2 + (f.y + sweepY) / 2 # formula for parabola with focus (f.x, f.y) and directrix sweepY
    py = arcY(arc1, sweepY, px)
    point = Point(px, py)

    return point



class Endpoint:
    edge = None
    left_arc = None
    right_arc = None

    def __init__(self, edge, left_arc=None, right_arc=None):
        self.edge = edge
        self.left_arc = left_arc
        self.right_arc = right_arc

    def calculateX(self, sweepY):
        # start = self.edge.start
        # vec = self.edge.vec
        if self.left_arc is not None and self.right_arc is not None:
            intersection = arcIntersect(self.left_arc, self.right_arc, sweepY)
            return intersection.x
        else:
            return -1000




class Beachline:
    endpoints = []
    arcs = []

    def _insert(self, endpoint):
        self.endpoints.append(endpoint)

    def _insert_at(self, index, endpoint):
        self.endpoints.insert(index, endpoint)
    
    def _search(self, sweep, value):
        key_list = KeyList(self.endpoints, key=lambda endpoint: endpoint.calculateX(sweep))
        beach_index = bisect.bisect_right(key_list, value)
        return beach_index


    def insert(self, site):
        sweepY = site.y
        new_arc = Arc(focus=site)

        if len(self.endpoints) == 0: # this is first arc
            self._insert(Endpoint(edge=None, left_arc=None, right_arc=new_arc)) # we indicate a single arc with None for left arc

            self.arcs.append(new_arc)

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

            if existing_arc.focus.y == sweepY: # both are on the same y-value!
                intersect_y = 0
                intersect_x = (existing_arc.focus.x + new_arc.focus.x) / 2.0 # bisector is in middle
            else:
                intersect_y = arcY(existing_arc, sweepY, x = site.x)

            edge_start = (intersect_x, intersect_y)

            tangent = normalize(connecting_vec)
            up_vec = np.array([0, 0, 1])
            orthogonal = np.cross(np.array([tangent[0], tangent[1], 0]), up_vec)
            
            right_vec = normalize(orthogonal[:2])
            left_vec = - right_vec

            edge_left = Edge(pt(edge_start), left_vec)
            edge_right = Edge(pt(edge_start), right_vec)

            arc_left = Arc(focus=existing_arc.focus, prev_arc=existing_arc.prev_arc, next_arc=new_arc)
            existing_arc.prev_arc = new_arc # make existing arc the new right arc split, so we don't have to modify existing's right endpoint
            
            new_endpoint_left = Endpoint(edge=edge_left, left_arc=arc_left, right_arc=new_arc)
            new_endpoint_right = Endpoint(edge=edge_right, left_arc=new_arc, right_arc=existing_arc)

            endpoint.right_arc = arc_left
            self._insert_at(beach_index + 1, new_endpoint_left)
            self._insert_at(beach_index + 2, new_endpoint_right)

            self.arcs.append(arc_left)
            self.arcs.append(new_arc)



        
            

        


        


    # test_array = [(1,2),(3,4),(5.2,6),(5.2,7000),(5.3,8),(9,10)]
    # print(bisect.bisect_right(KeyList(test_array, key=lambda x: x[0]+1), 6))



class Event:
    x = 0.0
    p = None
    a = None
    valid = True
    
    def __init__(self, x, p, a):
        self.x = x
        self.p = p
        self.a = a
        self.valid = True

     

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def push(self, item, type):
        # check for duplicate
        if item in self.entry_finder: return
        count = next(self.counter)
        # use flipped y-coordinate as a primary key (heapq in python is min-heap)
        combined = (item, type)
        entry = [-item.y, count, combined]
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