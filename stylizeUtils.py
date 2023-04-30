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

def mul(a, b) :
    # Multiply two vectors into a matrix
    return np.asmatrix(b).T @ np.asmatrix(a)

def shadeTri(img, v1, v2, v3, c1, c2, c3, flipY=True):
    # Source: https://stackoverflow.com/questions/61854897/how-to-apply-a-three-point-triangle-gradient-in-opencv
    
    # Make array of vertices
    # ax bx cx
    # ay by cy
    #  1  1  1
    a = v1
    b = v2
    c = v3

    triArr = np.asarray([int(a[0]), int(b[0]), int(c[0]), int(a[1]), int(b[1]), int(c[1]), 1,1,1]).reshape((3, 3))

    # Get bounding box of the triangle
    xleft = min(int(a[0]), int(b[0]), int(c[0]))
    xright = max(int(a[0]), int(b[0]), int(c[0]))

    ytop = min(int(a[1]), int(b[1]), int(c[1]))
    ybottom = max(int(a[1]), int(b[1]), int(c[1]))

    # Build np arrays of coordinates of the bounding box
    xs = range(xleft, xright)
    ys = range(ytop, ybottom)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten()
    yv = yv.flatten()

    # Compute all least-squares /
    p = np.array([xv, yv, [1] * len(xv)])
    alphas, betas, gammas = np.linalg.lstsq(triArr, p, rcond=-1)[0]

    # Apply mask for pixels within the triangle only
    mask = (alphas > 0) & (betas > 0) & (gammas > 0)
    alphas_m = alphas[mask]
    betas_m = betas[mask]
    gammas_m = gammas[mask]
    xv_m = xv[mask]
    yv_m = yv[mask]

    # Compute and assign colors
    colors = mul(c1, alphas_m) + mul(c2, betas_m) + mul(c3, gammas_m)

    if flipY:
        yv_m *= -1
    
    xv_m = np.clip(xv_m, 0, img.shape[1]-1)
    yv_m = np.clip(yv_m, 0, img.shape[0]-1)

    if len(xv_m) > 0 and len(yv_m) > 0: # i guess sometimes the triangles are lines
        img[yv_m, xv_m] = colors

    return img


def colorAtRegion(v, region_colors):
    x = int(v[0])
    y = int(-v[1])

    x = np.clip(x, 0, region_colors.shape[1]-1)
    y = np.clip(y, 0, region_colors.shape[0]-1)

    return region_colors[y][x]

def colorAroundRegion(v, region_colors, spread=1):
    c = np.array([0., 0., 0.])
    n = 0
    for i in range(-spread, spread+1):
        for j in range(-spread, spread+1):
            c += colorAtRegion(v + np.array([i, j]), region_colors)
            n += 1
    return c / n

def colorTup(c):
    return (int(c[0]), int(c[1]), int(c[2]))


def gourad(img, triangles, region_colors, softness=0):#vertice_colors):
    # for face_id in triangles:
    #     for tri in triangles[face_id]:
    for tri in triangles:
        v1 = tri.vertices[0]
        v2 = tri.vertices[1]
        v3 = tri.vertices[2]

        # c1 = colorAtRegion(v1, region_colors)
        # c2 = colorAtRegion(v2, region_colors)
        # c3 = colorAtRegion(v3, region_colors)
        c1 = colorAroundRegion(v1, region_colors, spread=softness)
        c2 = colorAroundRegion(v2, region_colors, spread=softness)
        c3 = colorAroundRegion(v3, region_colors, spread=softness)

        # c1 = getColor(vertice_colors, tri.sites[0])
        # c2 = getColor(vertice_colors, tri.sites[1])
        # c3 = getColor(vertice_colors, tri.sites[2])

        img = shadeTri(img, v1, v2, v3, c1, c2, c3, flipY=True)

    # for tri in triangles:
    #     v1 = tri.vertices[0]
    #     v2 = tri.vertices[1]
    #     v3 = tri.vertices[2]

    #     c1 = colorAroundRegion(v1, region_colors, spread=5)
    #     c2 = colorAroundRegion(v2, region_colors, spread=5)
    #     c3 = colorAroundRegion(v3, region_colors, spread=5)
    
    #     img = cv2.circle(img, center=arrToCvTup(flipY(v1)), radius=10, color=(0,0,0), thickness=-1)
    #     img = cv2.circle(img, center=arrToCvTup(flipY(v2)), radius=10, color=(0,0,0), thickness=-1)
    #     img = cv2.circle(img, center=arrToCvTup(flipY(v3)), radius=10, color=(0,0,0), thickness=-1)

    #     img = cv2.circle(img, center=arrToCvTup(flipY(v1)), radius=6, color=colorTup(c1), thickness=-1)
    #     img = cv2.circle(img, center=arrToCvTup(flipY(v2)), radius=6, color=colorTup(c2), thickness=-1)
    #     img = cv2.circle(img, center=arrToCvTup(flipY(v3)), radius=6, color=colorTup(c3), thickness=-1)


# def gourad(img, triangles, vertice_colors):
#     for face_id in triangles:
#         for tri in triangles[face_id]:
#             lowest, lowest_ind, highest, highest_ind = topBottomTriVert(tri.vertices)

#             lV = tri.vertices[lowest_ind-1]
#             rV = tri.vertices[(lowest_ind+1)%len(tri.vertices)]

#             cL = getColor(vertice_colors, tri.sites[lowest_ind-1])
#             cR = getColor(vertice_colors, tri.sites[(lowest_ind+1)%len(tri.vertices)])
#             cB = getColor(vertice_colors, tri.sites[lowest_ind])

#             left = (lV - lowest)
#             right = (rV - lowest)
#             topleft = rV - lV

#             height = int(highest[1]) - int(lowest[1])
#             for i in range(height+1):
#                 y = int(lowest[1]) + i

#                 if i >= left[1]:
#                     leftPos = lV + topleft * (i - left[1]) / topleft[1]
#                     t = (i - left[1]) / topleft[1]
#                     leftColor = (1-t) * cL + t * cR
#                 else:
#                     leftPos = lowest + left * (i / left[1])
#                     t = i / left[1]
#                     leftColor = (1-t) * cB + t * cL

#                  # need to vectorize, fix jaggies
#                 x = int(leftPos[0])
#                 if abs(y) < img.shape[0] and abs(x) < img.shape[1]:
#                     img[-y][x] = tuple(leftColor)

#                 if i >= right[1]:
#                     rightPos = rV + (-topleft) * (i - right[1]) / (-topleft[1])
#                     t = (i - right[1]) / (-topleft[1])
#                     rightColor = (1-t) * cR + t * cL
#                 else:
#                     rightPos = lowest + right * (i / right[1])
#                     t = i / right[1]
#                     rightColor = (1-t) * cB + t * cR
#                     # print("B",rightPos,right, tri.vertices)

#                 # rightColor=leftColor

#                 width = int((rightPos - leftPos)[0])
#                 for j in range(width+1):
#                     x = int(leftPos[0]) + j
#                     t = float(j) / width if width > 0 else 0
#                     color = (1-t) * leftColor + t * rightColor

#                     if abs(y) < img.shape[0] and abs(x) < img.shape[1]:
#                         img[-y][x] = tuple(color)

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