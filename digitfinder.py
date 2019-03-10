import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

# flags
FILL = 0
CROP_RECTANGLE = 1
CROP_TIGHT = 2        

def getBoundingRectArea(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w * h, (x, y, w, h)


def getContours(image):
    # params
    blur = 1
    threshold = 240

    copy = np.array(image, dtype=np.uint8)
    blur = cv2.GaussianBlur(copy, (blur, blur), 0)
    _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    return copy, contours


def flagCropTight(image):
    copy, contours = getContours(image)

    max_area = 0
    for c in contours:
        area, _ = getBoundingRectArea(c)
        if  area > max_area:
            max_area = area
            max_countour = c

    for i in range(64):
        for j in range(64):
            if cv2.pointPolygonTest(max_countour,(j,i),True) < 0:
                copy[i,j] = 0
    return copy


def flagCropRectangle(image):
    copy, contours = getContours(image)

    max_rect = (0,0,0,0)
    max_area = 0
    for c in contours:
        area, rect = getBoundingRectArea(c)
        if  area > max_area:
            max_area = area
            max_rect = rect

    pad = 2
    x, y, w, h = max_rect
    for i in range(64):
        for j in range(64):
            if not (j > x - pad and j < x + w + pad and i > y - pad and i < y + h + pad):
                copy[i,j] = 0
    return copy


def flagFill(image):
    copy, contours = getContours(image)

    max_contour = contours[0]
    max_area = 0
    for c in contours:
        area, _ = getBoundingRectArea(c)
        if  area > max_area:
            max_area = area
            max_contour = c

    copy = np.zeros_like(image)
    cv2.fillPoly(copy, pts=[max_contour], color=(255,255,255))
    return copy

functions = {
    FILL: flagFill,
    CROP_RECTANGLE: flagCropRectangle,
    CROP_TIGHT: flagCropTight,
}

names = {
    FILL: "Fill",
    CROP_RECTANGLE: "Crop rectangle",
    CROP_TIGHT: "Crop tight",
}

def runWith(flag, imgs, print_first=False):
    for i in range(len(imgs)):
        imgs[i] = functions[flag](imgs[i])
        if print_first and i == 0:
            print("Using preprocessing: " + names[flag])
            plt.imshow(imgs[i])
            plt.title(names[flag])
            plt.show()
    return imgs

def test(train_data):
    imgs, _ = train_data.get_datas(random.randint(0, 40000 - 1), 10)
    imgs_rect = runWith(CROP_RECTANGLE, np.array(imgs, copy=True))
    imgs_tight = runWith(CROP_TIGHT, np.array(imgs, copy=True))
    imgs_fill = runWith(FILL, np.array(imgs, copy=True))
    for i in range(len(imgs)):
        fig = plt.figure()
        fig.add_subplot(1,4,1)
        plt.title("Original")
        plt.imshow(imgs[i] , cmap='gray')
        fig.add_subplot(1,4,2)
        plt.title("Fill")
        plt.imshow(imgs_fill[i], cmap='gray')
        fig.add_subplot(1,4,3)
        plt.title("Crop rectangle")
        plt.imshow(imgs_rect[i], cmap='gray')
        fig.add_subplot(1,4,4)
        plt.title("Crop tight")
        plt.imshow(imgs_tight[i], cmap='gray')
        plt.show()