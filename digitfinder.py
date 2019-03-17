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
    return x*y, (x, y, w, h)


def getContours(image):
    # params
    blur = 1
    threshold = 240

    copy = np.array(image, dtype=np.uint8)
    blur = cv2.GaussianBlur(copy, (blur, blur), 0)
    _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(binary,kernel,iterations = 1)
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    return copy, contours


def flagCropTight(image):
    copy, contours = getContours(image)

    if len(contours) == 0:
        return copy

    max_side = 0
    for c in contours:
        area, (x, y, w, h) = getBoundingRectArea(c)
        max_wh = max(w, h)
        #x+w < 64 and y+h < 64 and x > 0 and y > 0 and
        if max_wh > max_side and max_wh <= 36 and max_wh > 2:
            max_side = max_wh
            max_contour = c

        # cv2.rectangle(copy,(x,y),(x+max(w,h),y+max(w,h)),(0,255,0),2)

    # if max_side == 0:
    if max_side == 0:
        for c in contours:
            area, (x, y, w, h) = getBoundingRectArea(c)
            max_wh = max(w, h)
            #x+w < 64 and y+h < 64 and x > 0 and y > 0 and
            if max_wh > max_side and max_wh > 2:
                max_side = max_wh
                max_contour = c


    # print(max_side)
    for i in range(64):
        for j in range(64):
            if max_side > 0:
                if cv2.pointPolygonTest(max_contour,(j,i),True) < 0:
                    copy[i,j] = 0
    return copy

def showBoundingBox(image):
    copy, contours = getContours(image)

    max_side = 0
    for c in contours:
        _, (x, y, w, h) = getBoundingRectArea(c)
        max_wh = max(w, h)
        if max_wh <= 29:
            if max_wh > max_side:
                max_side = max_wh

            cv2.rectangle(copy,(x,y),(x+max(w,h),y+max(w,h)),(0,255,0),2)

    # for i in range(64):
    #     for j in range(64):
    #         if cv2.pointPolygonTest(max_countour,(j,i),True) < 0:
    #             copy[i,j] = 0
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
    if print_first:
        i = random.randint(0, len(imgs) - 1)
        print("Using preprocessing: " + names[flag])
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.title("Original")
        plt.imshow(imgs[i])
        fig.add_subplot(1,2,2)
        plt.title(names[flag])
        plt.imshow(functions[flag](imgs[i]))
        plt.show()

    for i in range(len(imgs)):
        imgs[i] = functions[flag](imgs[i])
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