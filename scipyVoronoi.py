from scipy.spatial import Voronoi
import numpy as np
from utils import *

import time
from stylizeUtils import *

import cv2

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


def getScipyVoronoi(sites, width, height):
    vor = Voronoi(sites)
    center = vor.points.mean(axis=0)

    diag = math.sqrt(height**2 + width**2)

    edges = []
    faces = []

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
            edges.append((v1, v2))
        elif v1 is not None or v2 is not None:
            endpoint = v1 if v1 is not None else v2
            vec = normalize(endpoint - ridge_center)
            other_endpoint_dir = np.sign(np.dot(ridge_center - center, direction)) * direction
            # if np.dot(vec, direction) > 0.9: # same direction
            #     other_endpoint_dir = -direction
            # else: 
            #     other_endpoint_dir = direction
            endpoint2 = ridge_center + other_endpoint_dir * diag

            edges.append((endpoint, endpoint2))
        else:
            endpoint = ridge_center + direction * diag
            endpoint2 = ridge_center - direction * diag
            edges.append((endpoint, endpoint2))


    regions, vertices = voronoi_finite_polygons_2d(vor)
    for region in regions:
        polygon = vertices[region]
        faces.append(polygon)

    return vor, edges, faces

def drawScipyLines(img, sLines):
    for line in sLines:
        img = cv2.line(img, arrToCvTup(line[0]), arrToCvTup(line[1]), color=(0, 0, 0), thickness=1)
    return img

def drawSites(img, sample_points):
    c = (0, 0, 0)
    for point in sample_points:
        img = cv2.circle(img, center=arrToCvTup(point), radius=4, color=c, thickness=-1)
    return img

def drawScipyFaces(img, sFaces):
    for polygon in sFaces:
        img, avg = averagePolygon(img, polygon)
    return img

    # masked_image = cv2.bitwise_and(img, mask)
    # cv2.imshow('masked', masked_image)
    # cv2.waitKey(0)
    