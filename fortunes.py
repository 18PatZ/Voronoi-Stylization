import random
import math

import cv2

from utils import *

ARC_COLOR = (0,0,255)#(0, 128, 255)
EDGE_COLOR = (0, 128, 255)
COMPLETE_COLOR = (0, 255, 255)
SITE_COLOR = (0, 0, 0)

ANIM_SPEED = 30
DRAW_SWEEP = True
DRAW_THICKNESS = 6

TEXT_OFFSET = np.array([10, 10])


def drawText(img, position, text, color=(255,255,255), scale=1):

    font = cv2.FONT_HERSHEY_SIMPLEX
    t1 = int(DRAW_THICKNESS/3 * 5)
    t2 = int(DRAW_THICKNESS/3)

    s1, _ = cv2.getTextSize(text, font, fontScale=scale, thickness=t1)
    s2, _ = cv2.getTextSize(text, font, fontScale=scale, thickness=t2)

    img = cv2.putText(img, text, arrToCvTup(np.array(position) - npt(s2)/2 + TEXT_OFFSET), 
        font, fontScale=scale, color=(0,0,0), lineType=cv2.LINE_AA, thickness=t1)

    img = cv2.putText(img, text, arrToCvTup(np.array(position) - npt(s2)/2 + TEXT_OFFSET), 
        font, fontScale=scale, color=color, lineType=cv2.LINE_AA, thickness=t2)
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




class Fortunes:
    def __init__(self, sites, img=None, skip_events_outside_img=True):
        self.events = PriorityQueue()
        self.beachline = Beachline()
        self.completed_edges = []

        self.sites = sites
        
        # self.img = img
        self.img = np.copy(img)

        self.bounding_box = get_image_bounding_box(self.img)

        self.faces = {}
        
        self.skip_events_outside_img = skip_events_outside_img
        self.min_intersection_y = -self.img.shape[0] # don't wait for circle events if they're outside of the image

        self.circle_counter = 0
        self.latest_arc_event_ids = {}

        self.lastY = 0
        self.circles = []

        for i in range(len(sites)):
            site = sites[i]
            self.events.push(
                value = site[1],
                item = (pt(site), i), 
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

        self.makePolygons()
        
        if animate:
            if final_sweep_value is not None and final_sweep_value < last_sweep:
                self.step_animation(final_sweep_value - ANIM_SPEED)
                cv2.waitKey(0)
            print("Animation complete.")
            
            cv2.imshow("Fortune's Algorithm", self.draw())
            cv2.waitKey(0)

    
    def finish_edges(self, bounding_box):
        # find uncompleted edges
        w = self.img.shape[1]
        h = self.img.shape[0]

        final_sweep_value = None

        for edge in self.completed_edges:
            if not isPointInPolygon(edge.start, bounding_box):
                intersection, line = edgeIntersectBoundingBox(edge, bounding_box, max_ray_length=np.linalg.norm(npa(edge.end) - npa(edge.start)))

                if intersection is not None: # edge starts outside, need to cut it 
                    edge.start = pt(intersection)
                    edge.boundary_start = line

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
                    edge.boundary_end = line
                    edge.ending_sweep = intersection_sweep_value
                    self.completed_edges.append(edge)

        return final_sweep_value

    def connect_polygon_edges(self, id, edges):
        i = 0
        while i < len(edges):
            next = i+1
            if next >= len(edges):
                next = 0
            
            edge = edges[i]
            next_edge = edges[next]
            
            if edge.boundary_end is None and len(edges) > 1: # connecting unfinished edges
                if np.linalg.norm(next_edge.start - edge.end) >= 1: # gap
                    new_edge = Edge(edge.end, end=next_edge.start, site1_id=id, site2_id=id)
                    edges.insert(i+1, new_edge)
                    next_edge = new_edge

            if edge.boundary_end is not None and np.linalg.norm(next_edge.start - edge.end) >= 1:
                if next_edge.boundary_start is not None:
                    diff = next_edge.start - edge.end
                    boundary_line = edge.boundary_end[0]
                    boundary = normalize(boundary_line[1])

                    projection = np.dot(diff, boundary)
                    if projection < 0: # facing wrong direction, we go to corner of image instead
                        end = boundary_line[0] + boundary_line[1]
                    else:
                        end = edge.end + projection * boundary

                    new_edge = Edge(edge.end, end=end, site1_id=id, site2_id=id)
                    edges.insert(i+1, new_edge)

                    # might be corner and need more
                    new_edge.boundary_start = edge.boundary_end
                    
                    next_index = (edge.boundary_end[1] + 1) % len(self.bounding_box)
                    next_line = self.bounding_box[next_index]
                    new_edge.boundary_end = (next_line, next_index)
                else:
                    edges.insert(i+1, Edge(edge.end, end=next_edge.start, site1_id=id, site2_id=id))


            i += 1

    def makePolygons(self, check_unfinished=False, sweep=None):

        faces = {i: [] for i in range(len(self.sites))} # each face id corresponds to site
        self.faces = {}

        for edge in self.completed_edges:
            edge_copy = edge.np_copy()
            if not isPointInPolygon(edge.start, self.bounding_box) or not isPointInPolygon(edge.end, self.bounding_box):
                new_start, new_end = vectorPortionInRegion(edge_copy.start, edge_copy.end, self.bounding_box)
                if new_start is None or new_end is None: # this edge isn't even in region, skip
                    continue
                edge_copy.start = new_start
                edge_copy.end = new_end

            faces[edge.site1_id].append(edge_copy)
            faces[edge.site2_id].append(edge_copy.copy())

        if check_unfinished:
            for endpoint in self.beachline.endpoints:
                edge = endpoint.edge
                if edge is not None:
                    x, y = endpoint.calculateX(sweep, True)
                    faces[edge.site1_id].append(edge.np_copy(end_value = np.array([x, y])))
                    faces[edge.site2_id].append(edge.np_copy(end_value = np.array([x, y])))

        for id in faces:
            face_edges = faces[id]
            if len(face_edges) == 0:
                continue

            site = self.sites[id]

            for i in range(len(face_edges)):
                edge = face_edges[i]
          
                sign = np.cross(conv2d3d(edge.start - site), conv2d3d(edge.end - edge.start))[2]
                if sign < 0: # not counterclockwise, flip
                    edge.end, edge.start = edge.start, edge.end
                    edge.boundary_start, edge.boundary_end = edge.boundary_end, edge.boundary_start
                    
                a = vecAngle(edge.start - site)
                face_edges[i] = (a, edge)
            
            face_edges.sort(key=lambda element: element[0])

            refined_edges = []
            for i in range(len(face_edges)):
                edge = face_edges[i][1]
                
                if len(refined_edges) > 0:
                    prev_edge = refined_edges[-1]
                
                if np.linalg.norm(edge.end - edge.start) >= 0.1:
                    # if they divide the same faces, they are both colinear, merge them
                    if len(refined_edges) > 0 and edge.same_ids(prev_edge): 
                        refined_edges[-1].end = edge.end
                        refined_edges[-1].boundary_end = edge.boundary_end
                    else:
                        refined_edges.append(edge)
            
            if len(refined_edges) >= 2 and refined_edges[0].same_ids(refined_edges[-1]): # another merge check because its circular
                refined_edges[-1].end = refined_edges[0].end
                refined_edges[-1].boundary_end = refined_edges[0].boundary_end
                del refined_edges[0]

            self.connect_polygon_edges(id, refined_edges)

            self.faces[id] = Face(site = self.sites[id], id=id, edges=refined_edges)


    def draw_animation_frame(self, y):
        sweep = -y

        img = np.copy(self.img)
        
        w = img.shape[1]
        h = img.shape[0]

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
            img = drawText(img, flipY(site), f"S{i}", scale=1)
        
        for edge in self.completed_edges:
            start = npa(edge.start)
            end = npa(edge.end)
            if edge.ending_sweep >= sweep:
                s = np.array([start[0], -start[1]])
                e = np.array([end[0], -end[1]])
                img = drawText(img, (s+e)/2, f"E{edge.site1_id},{edge.site2_id}", scale=0.8)

        for id in self.faces:
            face = self.faces[id]
            edges = face.edges
            
            for i in range(len(edges)):
                edge = edges[i]
                point = edge.start + normalize(flipY(face.centroid) - edge.start) * 50

                img = cv2.line(img, arrToCvTup(flipY(edge.start)), arrToCvTup(flipY(point)), color=(255,0,0), thickness=2)
                img = cv2.line(img, arrToCvTup(flipY(point)), arrToCvTup(flipY(edge.end)), color=(128,128,0), thickness=2)
                
                img = drawText(img, (point[0], -point[1]), f"S{id}V{i}", scale=0.5)

                img = cv2.circle(img, center=arrToCvTup(flipY(edge.start)), radius=10, color=(0, 255, 0), thickness=-1)
                img = cv2.circle(img, center=arrToCvTup(flipY(edge.end)), radius=10, color=(255, 255, 0), thickness=-1)

        return img
    

    def draw(self, labelEdges=True, labelVertices=True, labelSites=True, labelCentroids=True, fontscale=1, thickness=DRAW_THICKNESS):
        img = np.copy(self.img)
        
        w = img.shape[1]
        h = img.shape[0]

        for i in range(len(self.sites)):
            site = self.sites[i]
            img = cv2.circle(img, center=arrToCvTup(site), radius=DRAW_THICKNESS, color=SITE_COLOR, thickness=-1)

        text = []
        
        if labelSites:
            for i in range(len(self.sites)):
                site = self.sites[i]
                text.append((flipY(site), "S"+str(i), fontscale))
        
        for id in self.faces:
            face = self.faces[id]
            edges = face.edges
            
            for i in range(len(edges)):
                edge = edges[i]
                nudge = 50
                point = edge.start + normalize(face.centroid - edge.start) * nudge
                
                img = cv2.line(img, arrToCvTup(flipY(edge.start)), arrToCvTup(flipY(edge.end)), color=COMPLETE_COLOR, thickness=thickness)

                edge_center = (edge.start+edge.end)/2
                edge_center += normalize(face.centroid - edge_center) * nudge

                # img = cv2.line(img, arrToCvTup(flipY(point)), arrToCvTup(flipY(face.site)), color=(128,128,0), thickness=1)
                img = cv2.line(img, arrToCvTup(flipY(edge.start)), arrToCvTup(flipY(face.site)), color=(128,128,0), thickness=1)
                
                if labelVertices:
                    text.append((flipY(point), f"S{id}V{i}", 0.5 * fontscale))

                if labelEdges:
                    text.append((flipY(edge_center), f"S{id}E{i}", 0.8 * fontscale))

                if labelCentroids:
                    text.append((flipY(face.centroid), f"S{id}C", 0.5 * fontscale))

        for t in text:
            drawText(img, position=t[0], text=t[1], scale=t[2])
            
        return img


    def step_animation(self, target_sweep):
        
        for y in range(int(self.lastY), -int(target_sweep), ANIM_SPEED):
            # cv2.namedWindow("Fortune's Algorithm", cv2.WINDOW_NORMAL)
            # img = resizeImgToScreen(img) 
            cv2.imshow("Fortune's Algorithm", self.draw_animation_frame(y))

            # print(sweep, target_sweep)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord(' '):
                key = cv2.waitKey(0)
            if key & 0xFF == ord('p'):
                self.makePolygons(check_unfinished=True, sweep=target_sweep)
                cv2.imshow("Fortune's Algorithm", self.draw_animation_frame(y))
                key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                exit()

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
        


    def get_faces(self):
        return self.faces

    def get_edges(self):
        edges = [] # caution - duplicates
        for id in self.faces:
            for edge in self.faces[id].edges:
                edges.append(edge)
        return edges

    def get_vertices(self):
        vertices = [] # caution - duplicates
        for id in self.faces:
            for edge in self.faces[id].edges:
                vertices.append(edge.start)
        return vertices
