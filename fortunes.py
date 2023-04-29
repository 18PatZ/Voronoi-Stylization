import random
import math

import cv2

from utils import *

ARC_COLOR = (0,0,255)#(0, 128, 255)
EDGE_COLOR = (0, 128, 255)
COMPLETE_COLOR = EDGE_COLOR#(0, 255, 255)
SITE_COLOR = (0, 0, 0)

ANIM_SPEED = 10
DRAW_SWEEP = True
DRAW_THICKNESS = 6

TEXT_OFFSET = np.array([10, 10])

def arrToCvTup(a):
    return (int(a[0]), int(a[1]))


def drawText(img, position, text, color=(255,255,255), scale=1):
    img = cv2.putText(img, text, arrToCvTup(np.array(position) + TEXT_OFFSET), 
        cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=(0,0,0), lineType=cv2.LINE_AA, thickness=int(DRAW_THICKNESS/3 * 5))

    img = cv2.putText(img, text, arrToCvTup(np.array(position) + TEXT_OFFSET), 
        cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=color, lineType=cv2.LINE_AA, thickness=int(DRAW_THICKNESS/3))
    return img

def resizeImgToScreen(img):
    
    w = img.shape[1]
    h = img.shape[0]

    screen = np.array([5120, 2880])
    dim = np.array([w, h])

    if np.all(dim < screen): # both smaller than screen
        mult = np.min(screen / dim)
        new_dim = mult * dim
        
        img = cv2.resize(img, arrToCvTup(new_dim))
    
    return img


def vectorPortionInRegion(start, end, bounding_box):    
    start_inside = isPointInPolygon(pt(start), bounding_box)
    end_inside = isPointInPolygon(pt(end), bounding_box)

    if start_inside and end_inside:
        return start, end

    displacement = end-start
    
    intersection, _ = rayIntersectBoundingBox(start, displacement, bounding_box, only_closest=True, max_ray_length=1)
    if intersection is None:
        return None, None

    if not start_inside:
        start = intersection
    if not end_inside:
        nudged_start = start + normalize(displacement) * 0.01 # since start already intersects box, needs a little nudge
        intersection, _ = rayIntersectBoundingBox(nudged_start, displacement, bounding_box, only_closest=True, max_ray_length=1)
        
        if intersection is not None:
            end = intersection
        
    return start, end




def drawArc(img, arc, sweep, xMin, xMax):
    if sweep == arc.focus.y:
        return

    w = img.shape[1]
    for i in range(max(0, int(xMin)), min(int(xMax), w), 2):
        x1 = i
        x2 = i+2

        y1 = -arcY(arc, sweep, x1)
        y2 = -arcY(arc, sweep, x2)

        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color=ARC_COLOR, thickness=DRAW_THICKNESS)

def cap(tup, img):
    w = img.shape[1]
    h = img.shape[0]
    longest = math.sqrt(w**2 + h**2)
    tupL = math.sqrt(tup[0]**2 + tup[1]**2)
    if tupL >= longest:
        tup = np.array([tup[0] / tupL * longest, tup[1] / tupL * longest])
    return tup



class Fortunes:
    def __init__(self, sites, img=None, skip_events_outside_img=True):
        self.events = PriorityQueue()
        self.beachline = Beachline()
        self.completed_edges = []

        self.sites = sites
        
        # self.img = img
        self.img = np.copy(img)

        self.bounding_box = get_image_bounding_box(self.img)
        
        self.skip_events_outside_img = skip_events_outside_img
        self.min_intersection_y = -self.img.shape[0] # don't wait for circle events if they're outside of the image

        self.circle_counter = 0
        self.latest_arc_event_ids = {}

        self.lastY = 0
        self.circles = []

        for i in range(len(sites)):
            site = sites[i]
            self.events.push(
                value = -site[1], # image has y down as positive, we flip
                item = (pt((site[0], -site[1])), i), 
                type = "SITE") 


    def process(self, animate=True):
        while not self.events.empty():
            value, event, type = self.events.pop()
            
            if animate:
                if type != "CIRCLE" or self.is_circle_event_valid(event):
                    self.step_animation(value)

            if type == "SITE":
                self.handle_site_event(event)
            elif type == "CIRCLE":
                self.handle_circle_event(event)

        last_sweep = -self.lastY
        final_sweep_value = self.finish_edges(self.bounding_box)

        if animate:
            if final_sweep_value is not None and final_sweep_value < last_sweep:
                self.step_animation(final_sweep_value - ANIM_SPEED)

            print("Animation complete.")
            
            cv2.waitKey(0)

    
    def finish_edges(self, bounding_box):
        # find uncompleted edges
        w = self.img.shape[1]
        h = self.img.shape[0]

        final_sweep_value = None

        for edge in self.completed_edges:
            if not isPointInPolygon(edge.start, bounding_box):
                intersection, line = edgeIntersectBoundingBox(edge, bounding_box)

                if intersection is not None: # edge starts outside, need to cut it 
                    edge.start = pt(intersection)
                    edge.boundary = line

        for i in range(len(self.beachline.endpoints)):
            endpoint = self.beachline.endpoints[i]
            edge = endpoint.edge
            arc = endpoint.right_arc
        
            if edge is not None:
                intersection, line = edgeIntersectBoundingBox(edge, bounding_box)

                if intersection is not None:
                    dist_to_focus = np.linalg.norm(npa(arc.focus) - intersection)
                    intersection_sweep_value = intersection[1] - dist_to_focus

                    if final_sweep_value is None or intersection_sweep_value < final_sweep_value:
                        final_sweep_value = intersection_sweep_value

                    edge.end = pt(intersection)
                    edge.boundary = line
                    edge.ending_sweep = intersection_sweep_value
                    self.completed_edges.append(edge)

        return final_sweep_value


    # def makePolygons():



    def step_animation(self, target_sweep):
        
        w = self.img.shape[1]
        h = self.img.shape[0]

        for y in range(int(self.lastY), -int(target_sweep), ANIM_SPEED):
            sweep = -y

            img = np.copy(self.img)

            for i in range(len(self.sites)):
                site = self.sites[i]
                img = cv2.circle(img, center=arrToCvTup(site), radius=DRAW_THICKNESS, color=SITE_COLOR, thickness=-1)
            
            if DRAW_SWEEP:
                img = cv2.line(img, (0, int(y)), (w-1, int(y)), color=(255, 255, 255), thickness=DRAW_THICKNESS)
            
            for i in range(len(self.beachline.endpoints)):
                endpoint = self.beachline.endpoints[i]
            
                left, y = endpoint.calculateX(sweep, True)
                right = w

                if i < len(self.beachline.endpoints)-1:
                    right = self.beachline.endpoints[i+1].calculateX(sweep)

                if endpoint.right_arc is not None:
                    drawArc(img, endpoint.right_arc, sweep, left, right)

                if endpoint.edge is not None:
                    start = npa(endpoint.edge.start)
                    
                    current_end = np.array([left, y]) # current intersection
                    
                    start, end = vectorPortionInRegion(start, current_end, self.bounding_box)
                    if start is not None and end is not None:
                        img = cv2.line(img, (int(start[0]), int(-start[1])), (int(current_end[0]), int(-current_end[1])), color=EDGE_COLOR, thickness=DRAW_THICKNESS)

            for edge in self.completed_edges:
                start = npa(edge.start)
                end = npa(edge.end)
                if edge.ending_sweep >= sweep:
                    s = np.array([start[0], -start[1]])
                    e = np.array([end[0], -end[1]])
                    img = cv2.line(img, arrToCvTup(s), arrToCvTup(e), color=COMPLETE_COLOR, thickness=DRAW_THICKNESS)

            to_remove = []
            for circle in self.circles:
                (intersection, dist_to_focus, intersection_sweep_value) = circle
                img = cv2.circle(img, center=(int(intersection[0]), int(-intersection[1])), radius=int(dist_to_focus), color=(255, 0, 0), thickness=DRAW_THICKNESS)
                img = cv2.line(img, (0, int(-intersection_sweep_value)), (w-1, int(-intersection_sweep_value)), color=(255, 0, 0), thickness=DRAW_THICKNESS)

                if sweep <= intersection_sweep_value:
                    to_remove.append(circle)
            for circle in to_remove:
                self.circles.remove(circle)


            for i in range(len(self.sites)):
                site = self.sites[i]
                img = drawText(img, site, "S"+str(i), scale=1)
            
            for edge in self.completed_edges:
                start = npa(edge.start)
                end = npa(edge.end)
                if edge.ending_sweep >= sweep:
                    s = np.array([start[0], -start[1]])
                    e = np.array([end[0], -end[1]])
                    img = drawText(img, (s+e)/2, f"E{edge.site1_id},{edge.site2_id}", scale=0.8)

            
            # cv2.namedWindow("Fortune's Algorithm", cv2.WINDOW_NORMAL)
            # img = resizeImgToScreen(img) 
            cv2.imshow("Fortune's Algorithm", img)

            # print(sweep, target_sweep)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                exit()
            elif key & 0xFF == ord(' '):
                cv2.waitKey(0)

        self.lastY = -target_sweep + 1
        



    def handle_site_event(self, event):
        (site, site_id) = event
        arcs_modified = self.beachline.insert(site, site_id)

        for arc in arcs_modified:
            self.check_circle_event(arc)
            


    def check_circle_event(self, arc):
        if arc is None or arc.prev_endpoint is None or arc.next_endpoint is None:
            return
        
        edge_left = arc.prev_endpoint.edge
        edge_right = arc.next_endpoint.edge

        if edge_left is None or edge_right is None:
            return

        intersection = edgeIntersect(edge_left, edge_right)
        if intersection is None:
            return

        # intersection is center of circle
        dist_to_focus = np.linalg.norm(npa(arc.focus) - intersection)
        intersection_sweep_value = intersection[1] - dist_to_focus

        if self.skip_events_outside_img and intersection[1] < self.min_intersection_y:
            # don't wait for circle events if they're outside of the image
            # print("Ignoring circle event outside of image: INT", intersection, " SWP", intersection_sweep_value)
            return
        # print("Adding circle event INT", intersection, " SWP", intersection_sweep_value)

        self.events.push(
            value=intersection_sweep_value, 
            item=(pt(intersection), dist_to_focus, intersection_sweep_value, arc, self.circle_counter), 
            type="CIRCLE")

        self.latest_arc_event_ids[arc.id] = self.circle_counter
        self.circle_counter += 1

        # self.circles.append((intersection, dist_to_focus, intersection_sweep_value))

    def is_circle_event_valid(self, event):
        (intersection, dist_to_focus, intersection_sweep_value, arc, id) = event
        arc_latest_event = self.latest_arc_event_ids[arc.id]
        if arc_latest_event > id: # this event is outdated and no longer valid. Arc has new, later circle events
            return False
        return True

    def handle_circle_event(self, event):
        (intersection, dist_to_focus, intersection_sweep_value, arc, id) = event

        if not self.is_circle_event_valid(event):
            return
        
        arcs_modified, edges_completed = self.beachline.remove(arc, intersection)
        for arc in arcs_modified:
            self.check_circle_event(arc)
        for edge in edges_completed:
            edge.ending_sweep = intersection_sweep_value
            self.completed_edges.append(edge)
        


    def edges(self):
        edges = []
        for endpoint in self.beachline.endpoints:
            if endpoint.edge is not None:
                start = npa(endpoint.edge.start)
                vec = endpoint.edge.vec
                end = start + vec * 500

                edges.append([start[0], -start[1], end[0], -end[1]]) # image has y down as positive, we flip
        return edges
