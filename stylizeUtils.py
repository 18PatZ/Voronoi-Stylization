import numpy as np
from utils import *
import cv2

def averagePolygon(img, polygon_vertices):
    channels = img.shape[2]

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_corners = np.array([[arrToCvTup(p) for p in polygon_vertices]], dtype=np.int32)
    
    # fill the ROI so it doesn't get wiped out when the mask is applied
    ignore_mask_color = (255,) * channels
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    avg = cv2.mean(img, mask)
    cv2.fillPoly(img, roi_corners, avg)

    return img, avg

def topBottomTriVert(vertices):
    v = None
    ind = 0

    vT = None
    indT = 0

    for i in range(len(vertices)):
        if v is None or vertices[i][1] < v[1]:
            v = vertices[i]
            ind = i
        if vT is None or vertices[i][1] > vT[1]:
            vT = vertices[i]
            indT = i
    return v, ind, vT, indT


def getColor(vertice_colors, ind):
    if type(ind) is tuple:
        sum = np.array([0., 0., 0.])
        for site in ind:
            sum += vertice_colors[site][:3]
        return sum / len(ind)
    else:
        return vertice_colors[ind][:3]
        

def gourad(img, triangles, vertice_colors):
    for face_id in triangles:
        for tri in triangles[face_id]:
            lowest, lowest_ind, highest, highest_ind = topBottomTriVert(tri.vertices)

            lV = tri.vertices[lowest_ind-1]
            rV = tri.vertices[(lowest_ind+1)%len(tri.vertices)]

            cL = getColor(vertice_colors, tri.sites[lowest_ind-1])
            cR = getColor(vertice_colors, tri.sites[(lowest_ind+1)%len(tri.vertices)])
            cB = getColor(vertice_colors, tri.sites[lowest_ind])

            left = (lV - lowest)
            right = (rV - lowest)
            topleft = rV - lV

            height = int(highest[1]) - int(lowest[1])
            for i in range(height+1):
                y = int(lowest[1]) + i

                if i >= left[1]:
                    leftPos = lV + topleft * (i - left[1]) / topleft[1]
                    t = (i - left[1]) / topleft[1]
                    leftColor = (1-t) * cL + t * cR
                else:
                    leftPos = lowest + left * (i / left[1])
                    t = i / left[1]
                    leftColor = (1-t) * cB + t * cL

                if i >= right[1]:
                    rightPos = rV + (-topleft) * (i - right[1]) / (-topleft[1])
                    t = (i - right[1]) / (-topleft[1])
                    rightColor = (1-t) * cR + t * cL
                else:
                    rightPos = lowest + right * (i / right[1])
                    t = i / right[1]
                    rightColor = (1-t) * cB + t * cR
                    # print("B",rightPos,right, tri.vertices)

                # rightColor=leftColor

                width = int((rightPos - leftPos)[0])
                for j in range(width+1):
                    x = int(leftPos[0]) + j
                    t = float(j) / width if width > 0 else 0
                    color = (1-t) * leftColor + t * rightColor

                    if abs(y) < img.shape[0] and abs(x) < img.shape[1]:
                        img[-y][x] = tuple(color)

    # for face_id in triangles:
    #     for tri in triangles[face_id]:
    #         lowest, lowest_ind, highest, highest_ind = topBottomTriVert(tri.vertices)

    #         lV = tri.vertices[lowest_ind-1]
    #         rV = tri.vertices[(lowest_ind+1)%len(tri.vertices)]

    #         cL = getColor(vertice_colors, tri.sites[lowest_ind-1])
    #         cR = getColor(vertice_colors, tri.sites[(lowest_ind+1)%len(tri.vertices)])
    #         cB = getColor(vertice_colors, tri.sites[lowest_ind])

    #         img = cv2.circle(img, center=arrToCvTup(flipY(lV)), radius=10, color=(0,0,0), thickness=-1)
    #         img = cv2.circle(img, center=arrToCvTup(flipY(rV)), radius=10, color=(0,0,0), thickness=-1)
    #         img = cv2.circle(img, center=arrToCvTup(flipY(lowest)), radius=10, color=(0,0,0), thickness=-1)

    #         img = cv2.circle(img, center=arrToCvTup(flipY(lV)), radius=6, color=tuple(cL), thickness=-1)
    #         img = cv2.circle(img, center=arrToCvTup(flipY(rV)), radius=6, color=tuple(cR), thickness=-1)
    #         img = cv2.circle(img, center=arrToCvTup(flipY(lowest)), radius=6, color=tuple(cB), thickness=-1)

            # print(lV, idL, rV, idR, lowest, idB)
            # img = cv2.putText(img, f"LV{idL}", arrToCvTup(flipY(lV)), 
            #     cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), lineType=cv2.LINE_AA, thickness=2)
            # # if idR == 2:
            # img = cv2.putText(img, f"RL{idR}", arrToCvTup(flipY(rV)), 
            #     cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), lineType=cv2.LINE_AA, thickness=2)
            # # if idB == 2:
            # img = cv2.putText(img, f"LOW{idB}", arrToCvTup(flipY(lowest)), 
            #     cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), lineType=cv2.LINE_AA, thickness=2)