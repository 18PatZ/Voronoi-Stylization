import cv2
import numpy as np
import math
import random

import time

from scipyVoronoi import *
from Voronoi import Voronoi as Vr
from fortunes import Fortunes

import os

from stylizeUtils import *

write = True
filepath = "balloon.jpeg"
img = cv2.imread(f"input/{filepath}")

# write = False
# filepath = "black.png"
# img = np.zeros((1440,2560,3), dtype=np.uint8)
# img = np.zeros((2560,5120,3), dtype=np.uint8)
# img = np.zeros((2880, 5120,3), dtype=np.uint8)

spl = filepath.split(".")
filename = spl[0]
extension = spl[1]

outPath = f"output/{filename}"

if write:
    if not os.path.exists(outPath):
        os.makedirs(outPath)

sample_points = []

height, width, channels = img.shape
diag = math.sqrt(height**2 + width**2)

random.seed(10)

n = 50
drawVD = True

# for i in range(0, n):
#     x = (i+0.5) * width / n
#     for j in range(0, n):
#         y = (j+0.5) * height / n

#         sample_points.append([x, y])
for i in range(0, n**2):
    sample_points.append([random.random() * width, random.random() * height])
    # p = np.array([i * width / (n**2) * 15.0/16.0 + width/32.0, i * height / (n**2) / 2 + height/4])
    # p += np.array([random.random() * width/16 - width/32, random.random() * height/16 - height/8])
    
    # angle = i * 90.0 / (n**2) * (math.pi / 180)
    # x = math.sin(angle) * width * 3.0/4.0 + width / 8
    # y = height-(math.cos(angle) * 3.0/4.0 * height + height / 8) + (i * height /8 / (n**2))

    # x = i / (n**2)
    # y = x**2 * height/8# - height/32

    # p = [width-(x * width * 6.0/8.0 + width / 4 + width * 0/8), height-y]

    ###
    # x = i / (n**2) * width/2
    # y = (((x - width/2) / 5) ** 2 ) / 70

    # y = height - y
    
    # p = [x * 2.4/4.0 + width/8, y/2 + height * 1.5/4.0]
    ###

    ###
    # x = i / (n**2) * width
    # y = (((x - width) / 5) ** 2 ) / 70

    # y = y
    
    # p = [x * 2.4/4.0 - width*4/16, y]
    ###

    ###
    # x = i / (n**2) * width
    # y = (height/2)*math.sin(x / width * 20) + height/2#(((x - 2500) / 5) ** 2 ) / 70
    
    # # p = [x * 2.4/4.0 + width/8, y]
    # p = [y, x]
    ###

    # p = [x, y]

    # sample_points.append(p)
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

if drawVD:
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

if write:
    if drawVD:
        cv2.imshow('SciPy Voronoi', sImg)
    cv2.imshow('SciPy Stylized', sStylized)


    if drawVD:
        cv2.imwrite(f'{outPath}/{filename}-{n}-scipy-voronoi.{extension}', sImg)
    cv2.imwrite(f'{outPath}/{filename}-{n}-scipy-stylized.{extension}', sStylized)


start = time.time()
vp = Fortunes(sites=sites_flipped, img=fImgL, filename=filename+f"-{n}")
vp.process(animate=not write, postprocess=not write)
print("Fortune's done in ", time.time()-start)

if write:
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
        if drawVD:
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
        if drawVD:
            fImgD = cv2.line(fImgD, arrToCvTup(flipY(tri.vertices[i])), arrToCvTup(flipY(tri.vertices[i-1])), color=(0,0,255), thickness=1)
        # fStylizedD = cv2.line(fStylizedD, arrToCvTup(flipY(tri.vertices[i])), arrToCvTup(flipY(tri.vertices[i-1])), color=(0,0,0), thickness=1)
        polygon.append(flipY(tri.vertices[i]))
    fStylizedD, avg = averagePolygon(fStylizedD, polygon)

print("Delaunay stylized in ", time.time()-s)

# drawSites(fImg, sample_points)

if write:
    if drawVD:
        cv2.imwrite(f'{outPath}/{filename}-{n}-fortunes-1voronoi.{extension}', fImg)
    cv2.imwrite(f'{outPath}/{filename}-{n}-fortunes-2stylizedV.{extension}', fStylized)
    if drawVD:
        cv2.imwrite(f'{outPath}/{filename}-{n}-fortunes-3delaunay.{extension}', fImgD)
    cv2.imwrite(f'{outPath}/{filename}-{n}-fortunes-4stylizedD.{extension}', fStylizedD)

    s = time.time()
    gourad(fStylizedG, vp.triangles, fStylized, softness=0)
    print("Gouraud shaded in ", time.time()-s)

    cv2.imwrite(f'{outPath}/{filename}-{n}-fortunes-5stylizedG.{extension}', fStylizedG)

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