import random
import math

import cv2

from utils import *

class Fortunes:
    def __init__(self, sites, img=None):
        self.events = PriorityQueue()
        self.beachline = Beachline()
        self.img = img

        for site in sites:
            self.events.push(pt((site[0], -site[1])), "SITE") # image has y down as positive, we flip


    def process(self):
        while not self.events.empty():
            event, type = self.events.pop()

            if type == "SITE":
                self.handle_site_event(event)
            elif type == "CIRCLE":
                self.handle_circle_event(event)


    def handle_site_event(self, site):
        print("site",site)
        self.beachline.insert(site)
        for endpoint in self.beachline.endpoints:
            print(endpoint.calculateX(-1000))
            print("    ",endpoint.edge)

        img = np.copy(self.img)
        sweep = site.y-1
        print(len(self.beachline.arcs), "arcs")
        for arc in self.beachline.arcs:
            for i in range(0, img.shape[0], 2):
                x1 = i
                x2 = i+2

                y1 = -arcY(arc, sweep, x1)
                y2 = -arcY(arc, sweep, x2)

                img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)

        for endpoint in self.beachline.endpoints:
            if endpoint.edge is not None:
                start = npa(endpoint.edge.start)
                vec = endpoint.edge.vec
                end = start + vec * 2000

                img = cv2.line(img, (int(start[0]), int(-start[1])), (int(end[0]), int(-end[1])), color=(0, 0, 0), thickness=3)
        
        cv2.imshow('image3', img)
        cv2.waitKey(0)
            

    def handle_circle_event(self, event):
        pass


    def edges(self):
        edges = []
        for endpoint in self.beachline.endpoints:
            if endpoint.edge is not None:
                start = npa(endpoint.edge.start)
                vec = endpoint.edge.vec
                end = start + vec * 500

                edges.append([start[0], -start[1], end[0], -end[1]]) # image has y down as positive, we flip
        return edges
