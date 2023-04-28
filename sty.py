import cv2
from scipy.spatial import Voronoi
import numpy as np
import math
import random

import time

from Voronoi import Voronoi as Vr
from fortunes import Fortunes

filepath = "balloon.jpeg"
img = cv2.imread(f"input/{filepath}")

sample_points = []

height, width, channels = img.shape
diag = math.sqrt(height**2 + width**2)

random.seed(10)

n = 5

# def trunc(p1, p2, line):
#     # v1 = np.array([tup[0], tup[1]])
#     # v2 = np.array([tup[0], tup[1]])
#     v = p2 - p1
#     x = tup[0]
#     y = tup[1]
#     if x < 0:
#         x = 0
#     if x > img.shape[1] - 1:
#         x = img.shape[1]-1
#     if y < 0:
#         y = 0
#     if y > img.shape[0] - 1:
#         y = img.shape[0]-1
#     return (x,y)

def cap(tup, img):
    w = img.shape[1]
    h = img.shape[0]
    longest = math.sqrt(w**2 + h**2)
    tupL = math.sqrt(tup[0]**2 + tup[1]**2)
    if tupL >= longest:
        tup = np.array([tup[0] / tupL * longest, tup[1] / tupL * longest])
    return tup
    # x = tup[0]
    # y = tup[1]
    # if x < 0:
    #     x = 0
    # if x > img.shape[1] - 1:
    #     x = img.shape[1]-1
    # if y < 0:
    #     y = 0
    # if y > img.shape[0] - 1:
    #     y = img.shape[0]-1
    # return (x,y)

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp(axis=0).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def arrToCvTup(a):
    return (int(a[0]), int(a[1]))


# for i in range(0, n):
#     x = i * width / n
#     for j in range(0, n):
#         y = j * height / n

#         sample_points.append([x, y])
for i in range(0, n**2):
    sample_points.append([random.random() * width, random.random() * height])


sites = np.array(sample_points)


stylized = img.copy()

c = (0, 128, 255)#(0, 0, 0)
for point in sample_points:
    img = cv2.circle(img, center=arrToCvTup(point), radius=8, color=c, thickness=-1)
    # img = cv2.circle(img, center=arrToCvTup(point), radius=2, color=(0, 0, 0), thickness=-1)



start = time.time()
# vp = Vr(sites)
# vp.process()
# vp_lines = vp.get_output()
vp = Fortunes(sites, img=img)
vp.process()
vp_lines = vp.edges()
img = vp.img
print(vp_lines)
# if True:
#     exit()

lines_2 = [[[line[0], line[1]], [line[2], line[3]]] for line in vp_lines]

for l in lines_2:
    print(l)
print("Done in ", time.time()-start)

start = time.time()
vor = Voronoi(sites)

center = vor.points.mean(axis=0)

lines = []

for i in range(len(vor.ridge_points)):
    rpoints = vor.ridge_points[i]
    rverts = vor.ridge_vertices[i]

    s1 = sites[rpoints[0]]
    s2 = sites[rpoints[1]]
    ridge_center = (s1 + s2) / 2

    perp = normalize(s2 - s1)
    direction = np.cross(np.array([perp[0], perp[1], 0]), np.array([0, 0, 1]))
    direction = np.array([direction[0], direction[1]])

    v1 = None
    v2 = None

    if rverts[0] >= 0:
        v1 = vor.vertices[rverts[0]]
    if rverts[1] >= 0:
        v2 = vor.vertices[rverts[1]]

    if v1 is not None and v2 is not None:
        lines.append((v1, v2))
    elif v1 is not None or v2 is not None:
        endpoint = v1 if v1 is not None else v2
        vec = normalize(endpoint - ridge_center)
        other_endpoint_dir = np.sign(np.dot(ridge_center - center, direction)) * direction
        # if np.dot(vec, direction) > 0.9: # same direction
        #     other_endpoint_dir = -direction
        # else: 
        #     other_endpoint_dir = direction
        endpoint2 = ridge_center + other_endpoint_dir * diag

        lines.append((endpoint, endpoint2))
    else:
        endpoint = ridge_center + direction * diag
        endpoint2 = ridge_center - direction * diag
        lines.append((endpoint, endpoint2))

print("Done in ", time.time()-start)


# regions, vertices = voronoi_finite_polygons_2d(vor)
# for region in regions:
#     polygon = vertices[region]
#     mask = np.zeros(img.shape[:2], dtype=np.uint8)
#     roi_corners = np.array([[arrToCvTup(p) for p in polygon]], dtype=np.int32)#np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
#     # fill the ROI so it doesn't get wiped out when the mask is applied
#     ignore_mask_color = (255,) * channels
#     cv2.fillPoly(mask, roi_corners, ignore_mask_color)

#     avg = cv2.mean(img, mask)
#     cv2.fillPoly(stylized, roi_corners, avg)

    # masked_image = cv2.bitwise_and(img, mask)
    # cv2.imshow('masked', masked_image)
    # cv2.waitKey(0)


img2 = np.copy(img)

for line in lines:
    p1 = line[0]
    p2 = line[1]
    
    img = cv2.line(img, arrToCvTup(p1), arrToCvTup(p2), color=(0, 0, 0), thickness=1)

for line in lines_2:
    p1 = line[0]
    p2 = line[1]
    
    print(arrToCvTup(cap(p1, img2)), arrToCvTup(cap(p2, img2)))

    img2 = cv2.line(img2, arrToCvTup(cap(p1, img2)), arrToCvTup(cap(p2, img2)), color=(0, 0, 0), thickness=1)
    # stylized = cv2.line(stylized, arrToCvTup(p1), arrToCvTup(p2), color=(0, 0, 0), thickness=1)


# cv2.imshow('image',img)
# cv2.imshow('image2',img2)
# cv2.imshow('stylized', stylized)

spl = filepath.split(".")
filename = spl[0]
extension = spl[1]
cv2.imwrite(f'output/{filename}-{n}-voronoi.{extension}', img)
# cv2.imwrite(f'output/{filename}-{n}-stylized.{extension}', stylized)



#print(vor.ridge_points)
#print(vor.ridge_vertices)


cv2.waitKey(0)
cv2.destroyAllWindows()