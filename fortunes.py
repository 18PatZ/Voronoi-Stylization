import random
import math

import cv2

from utils import *

ARC_COLOR = (0, 128, 255)
COMPLETE_COLOR = ARC_COLOR#(0, 0, 0)
ANIM_SPEED = 5

def drawArc(img, arc, sweep, xMin, xMax):
    for i in range(max(0, int(xMin)), min(int(xMax), img.shape[0]), 2):
        x1 = i
        x2 = i+2

        y1 = -arcY(arc, sweep, x1)
        y2 = -arcY(arc, sweep, x2)

        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color=ARC_COLOR, thickness=3)


class Fortunes:
    def __init__(self, sites, img=None):
        self.events = PriorityQueue()
        self.beachline = Beachline()
        self.img = img

        self.circle_counter = 0
        self.latest_arc_event_ids = {}

        self.lastY = 0
        self.circles = []

        for site in sites:
            self.events.push(
                value = -site[1], # image has y down as positive, we flip
                item = pt((site[0], -site[1])), 
                type = "SITE") 


    def process(self):
        while not self.events.empty():
            value, event, type = self.events.pop()
            
            self.step_animation(value)

            if type == "SITE":
                self.handle_site_event(event)
            elif type == "CIRCLE":
                self.handle_circle_event(event)

        self.step_animation(-2 * self.img.shape[1])
        cv2.waitKey(0)


    def step_animation(self, target_sweep):
        for y in range(int(self.lastY), -int(target_sweep), ANIM_SPEED):
            sweep = -y

            img = np.copy(self.img)
            img = cv2.line(img, (0, int(y)), (img.shape[0]-1, int(y)), color=(255, 255, 255), thickness=3)
            
            for i in range(len(self.beachline.endpoints)):
                endpoint = self.beachline.endpoints[i]
            
                left, y = endpoint.calculateX(sweep, True)
                right = img.shape[0]

                if i < len(self.beachline.endpoints)-1:
                    right = self.beachline.endpoints[i+1].calculateX(sweep)

                if endpoint.right_arc is not None:
                    drawArc(img, endpoint.right_arc, sweep-0.01, left, right)

                if endpoint.edge is not None:
                    center = npa(endpoint.edge.start)
                    
                    start = np.array([left, y])
                    vec =  center - start
                    end = start + vec

                    img = cv2.line(img, (int(start[0]), int(-start[1])), (int(end[0]), int(-end[1])), color=ARC_COLOR, thickness=3)

            for edge in self.beachline.completed_edges:
                start = npa(edge.start)
                end = npa(edge.end)
                img = cv2.line(img, (int(start[0]), int(-start[1])), (int(end[0]), int(-end[1])), color=COMPLETE_COLOR, thickness=3)

            to_remove = []
            for circle in self.circles:
                (intersection, dist_to_focus, intersection_sweep_value) = circle
                img = cv2.circle(img, center=(int(intersection[0]), int(-intersection[1])), radius=int(dist_to_focus), color=(255, 0, 0), thickness=2)
                img = cv2.line(img, (0, int(-intersection_sweep_value)), (img.shape[0]-1, int(-intersection_sweep_value)), color=(255, 0, 0), thickness=2)

                if sweep <= intersection_sweep_value:
                    to_remove.append(circle)
            for circle in to_remove:
                self.circles.remove(circle)
            
            cv2.imshow('image3', img)
            w = cv2.waitKey(1)
            if w & 0xFF == ord('q'):
                exit()
            elif w & 0xFF == ord(' '):
                cv2.waitKey(0)

        self.lastY = -target_sweep + 1
        



    def handle_site_event(self, site):
        arcs = self.beachline.insert(site)

        for arc in arcs:
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

        self.events.push(
            value=intersection_sweep_value, 
            item=(pt(intersection), dist_to_focus, intersection_sweep_value, arc, self.circle_counter), 
            type="CIRCLE")

        self.latest_arc_event_ids[arc.id] = self.circle_counter
        self.circle_counter += 1

        # self.circles.append((intersection, dist_to_focus, intersection_sweep_value))


    def handle_circle_event(self, event):
        (intersection, dist_to_focus, intersection_sweep_value, arc, id) = event

        arc_latest_event = self.latest_arc_event_ids[arc.id]
        if arc_latest_event > id: # this event is outdated and no longer valid. Arc has new, later circle events
            return
        
        arcs = self.beachline.remove(arc, intersection)
        for arc in arcs:
            self.check_circle_event(arc)



    def edges(self):
        edges = []
        for endpoint in self.beachline.endpoints:
            if endpoint.edge is not None:
                start = npa(endpoint.edge.start)
                vec = endpoint.edge.vec
                end = start + vec * 500

                edges.append([start[0], -start[1], end[0], -end[1]]) # image has y down as positive, we flip
        return edges
