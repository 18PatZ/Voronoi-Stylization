import random
import math

import cv2

from utils import *


def drawArc(img, arc, sweep, xMin, xMax):
    for i in range(max(0, int(xMin)), min(int(xMax), img.shape[0]), 2):
        x1 = i
        x2 = i+2

        y1 = -arcY(arc, sweep, x1)
        y2 = -arcY(arc, sweep, x2)

        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)


class Fortunes:
    def __init__(self, sites, img=None):
        self.events = PriorityQueue()
        self.beachline = Beachline()
        self.img = img

        self.lastY = 0

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
        # print("site",site)


        for y in range(int(self.lastY), -int(site.y), 5):
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
                    drawArc(img, endpoint.right_arc, sweep-0.1, left, right)

                if endpoint.edge is not None:
                    center = npa(endpoint.edge.start)
                    
                    start = np.array([left, y])
                    vec =  center - start
                    end = start + vec

                    img = cv2.line(img, (int(start[0]), int(-start[1])), (int(end[0]), int(-end[1])), color=(0, 0, 0), thickness=3)
            
            cv2.imshow('image3', img)
            w = cv2.waitKey(1)
            if w & 0xFF == ord('q'):
                exit()
            elif w & 0xFF == ord('w'):
                cv2.waitKey(0)

        
        self.beachline.insert(site)

        self.lastY = -site.y + 1
            

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
