import cv2
import numpy as np
import math
import random

import time

from scipyVoronoi import *
from Voronoi import Voronoi as Vr
from fortunes import Fortunes

from stylizeUtils import *

filepath = "IMG_2374.jpeg"
img = cv2.imread(f"input/{filepath}")

spl = filepath.split(".")
filename = spl[0]
extension = spl[1]


sample_points = []

height, width, channels = img.shape
diag = math.sqrt(height**2 + width**2)

random.seed(10)

n = 5

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

fImg = np.copy(sImg)
fImgD = np.copy(sImg)

drawScipyLines(sImg, sLines)

fImgL = np.copy(sImg)

# drawScipyPoints(img, sites)


sStylized = img.copy()
fStylized = img.copy()
fStylizedD = img.copy()
fStylizedG = img.copy()

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


cv2.imwrite(f'output/{filename}-{n}-scipy-voronoi.{extension}', sImg)
cv2.imwrite(f'output/{filename}-{n}-scipy-stylized.{extension}', sStylized)


start = time.time()
vp = Fortunes(sites=sites_flipped, img=fImgL)
vp.process(animate=False, postprocess=False)
print("Fortune's done in ", time.time()-start)

sP = time.time()
vp.makePolygons()
print("Fortune's polygonized in ", time.time()-sP)

sT = time.time()
vp.triangulate()
print("Fortune's triangulated in ", time.time()-sT)

print(f"{len(sample_points)} sites, {len(vp.faces)} faces, {len(vp.triangles)} triangles")

print("Fortune's total ", time.time()-start)


s = time.time()

faces = vp.get_faces()
face_colors = {}
for face_id in faces:
    face = faces[face_id]
    # polygon = [flipY(edge.start) for edge in face.edges]
    polygon = []

    for edge in face.edges:
        fImg = cv2.line(fImg, arrToCvTup(flipY(edge.start)), arrToCvTup(flipY(edge.end)), color=(0, 0, 0), thickness=1)
        fImgD = cv2.line(fImgD, arrToCvTup(flipY(edge.start)), arrToCvTup(flipY(edge.end)), color=(0, 0, 0), thickness=1)

        polygon.append(flipY(edge.start))
    
    fStylized, avg = averagePolygon(fStylized, polygon)
    face_colors[face_id] = np.array(avg)

print("Voronoi stylized in ", time.time()-s)
s = time.time()


# fStylizedD = np.copy(fStylized)

# for i in range(100):
#     new_point = [random.random() * width, random.random() * height]
#     face = vp.get_closest_face(new_point)
#     if face is not None:
#         tup = arrToCvTup(new_point)
#         fStylized[tup[1]][tup[0]] = img[tup[1]][tup[0]]

# for face_id in faces:
for tri in vp.triangles:#[face_id]:
    polygon = []
    for i in range(len(tri.vertices)):
        fImgD = cv2.line(fImgD, arrToCvTup(flipY(tri.vertices[i])), arrToCvTup(flipY(tri.vertices[i-1])), color=(0,0,255), thickness=1)
        # fStylizedD = cv2.line(fStylizedD, arrToCvTup(flipY(tri.vertices[i])), arrToCvTup(flipY(tri.vertices[i-1])), color=(0,0,0), thickness=1)
        polygon.append(flipY(tri.vertices[i]))
    fStylizedD, avg = averagePolygon(fStylizedD, polygon)

print("Delaunay stylized in ", time.time()-s)


cv2.imwrite(f'output/{filename}-{n}-fortunes-voronoi.{extension}', fImg)
cv2.imwrite(f'output/{filename}-{n}-fortunes-delaunay.{extension}', fImgD)
cv2.imwrite(f'output/{filename}-{n}-fortunes-stylizedV.{extension}', fStylized)
cv2.imwrite(f'output/{filename}-{n}-fortunes-stylizedD.{extension}', fStylizedD)

s = time.time()
gourad(fStylizedG, vp.triangles, face_colors)
print("Gouraud shaded in ", time.time()-s)

cv2.imwrite(f'output/{filename}-{n}-fortunes-stylizedG.{extension}', fStylizedG)

# cv2.imshow('image2',img2)
cv2.imshow("Fortune's Voronoi", fImg)
cv2.imshow("Fortune's Stylized Voronoi", fStylized)
cv2.imshow("Fortune's Delaunay", fImgD)
cv2.imshow("Fortune's Stylized Delaunay", fStylizedD)
cv2.imshow("Fortune's Stylized Gouraud", fStylizedG)
# cv2.imshow("Fortune's Algorithm", vp.draw())
# cv2.imshow("Fortune's Algorithm - Delaunay", vp.draw(drawEdges=False, labelSites=False, thickness=1))
# cv2.imshow("Fortune's Algorithm", vp.draw(labelEdges=False, labelVertices=True, labelCentroids=False))
# cv2.imshow("Fortune's Algorithm", vp.draw(labelEdges=False, labelVertices=False, labelSites=False, labelCentroids=False, fontscale=0.5, thickness=1))





# cv2.imwrite(f'output/{filename}-{n}-scipy-voronoi.{extension}', sImg)
# cv2.imwrite(f'output/{filename}-{n}-scipy-stylized.{extension}', sStylized)

# cv2.imwrite(f'output/{filename}-{n}-fortunes-voronoi.{extension}', fImg)
# cv2.imwrite(f'output/{filename}-{n}-fortunes-delaunay.{extension}', fImgD)
# cv2.imwrite(f'output/{filename}-{n}-fortunes-stylizedV.{extension}', fStylized)
# cv2.imwrite(f'output/{filename}-{n}-fortunes-stylizedD.{extension}', fStylizedD)

cv2.waitKey(0)

#print(vor.ridge_points)
#print(vor.ridge_vertices)


cv2.waitKey(0)
cv2.destroyAllWindows()