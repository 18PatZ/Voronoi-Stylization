import cv2
import numpy as np
import math
import random

import time

from scipyVoronoi import *
from Voronoi import Voronoi as Vr
from fortunes import Fortunes

from stylizeUtils import *

filepath = "titanfall2.png"
img = cv2.imread(f"input/{filepath}")

sample_points = []

height, width, channels = img.shape
diag = math.sqrt(height**2 + width**2)

random.seed(10)

n = 500

# for i in range(0, n):
#     x = (i+0.5) * width / n
#     for j in range(0, n):
#         y = (j+0.5) * height / n

#         sample_points.append([x, y])
for i in range(0, n**2):
    sample_points.append([random.random() * width, random.random() * height])
    # sample_points.append([random.random() * width, 200])

# sample_points = [
#     [200, 200],
#     [400, 200],
#     [600, 200],
#     [800, 200],
#     [350, 210],
# ]


sites = np.array(sample_points)
sites_flipped = np.array([flipY(p) for p in sites])



start = time.time()
vor, sLines, sFaces = getScipyVoronoi(sites, width, height)
print("SciPy done in ", time.time()-start)

# vp2 = Vr(sites)
# vp2.process()
# vp_lines = vp2.get_output()

# fImg = np.copy(img)
sImg = np.copy(img)

drawScipyLines(sImg, sLines)

fImg = np.copy(sImg)

# drawScipyPoints(img, sites)


sStylized = img.copy()
fStylized = img.copy()

drawScipyFaces(sStylized, sFaces)



# for line in lines_2:
#     p1 = line[0]
#     p2 = line[1]
    
#     print(arrToCvTup(cap(p1, img2)), arrToCvTup(cap(p2, img2)))

#     img2 = cv2.line(img2, arrToCvTup(cap(p1, img2)), arrToCvTup(cap(p2, img2)), color=(0, 0, 0), thickness=1)
#     # stylized = cv2.line(stylized, arrToCvTup(p1), arrToCvTup(p2), color=(0, 0, 0), thickness=1)

# c = (0, 128, 255)

    # img = cv2.circle(img, center=arrToCvTup(point), radius=2, color=(0, 0, 0), thickness=-1)

cv2.imshow('SciPy Voronoi', sImg)
cv2.imshow('SciPy Stylized', sStylized)

start = time.time()
vp = Fortunes(sites=sites_flipped, img=fImg)
vp.process(animate=False)
print("Fortune's done in ", time.time()-start)



faces = vp.get_faces()
for face_id in faces:
    face = faces[face_id]
    polygon = [flipY(edge.start) for edge in face.edges]
    fStylized = averagePolygon(fStylized, polygon)

# cv2.imshow('image2',img2)
cv2.imshow("Fortune's Stylized", fStylized)
# cv2.imshow("Fortune's Algorithm", vp.draw(labelEdges=False, labelVertices=True, labelCentroids=False))
cv2.imshow("Fortune's Algorithm", vp.draw(labelEdges=False, labelVertices=False, labelSites=False, labelCentroids=False, fontscale=0.5, thickness=1))





spl = filepath.split(".")
filename = spl[0]
extension = spl[1]

cv2.imwrite(f'output/{filename}-{n}-scipy-voronoi.{extension}', sImg)
cv2.imwrite(f'output/{filename}-{n}-scipy-stylized.{extension}', sStylized)

cv2.imwrite(f'output/{filename}-{n}-fortunes-voronoi.{extension}', fImg)
cv2.imwrite(f'output/{filename}-{n}-fortunes-stylized.{extension}', fStylized)

cv2.waitKey(0)

#print(vor.ridge_points)
#print(vor.ridge_vertices)


cv2.waitKey(0)
cv2.destroyAllWindows()