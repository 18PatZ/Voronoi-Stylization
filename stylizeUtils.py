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

    return img