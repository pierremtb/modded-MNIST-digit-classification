import numpy as np
import cv2
from matplotlib import pyplot as plt
from DataContainer import DataContainer
from DeepConvNN import DeepConvNN

train_data = DataContainer("./input/train_images.pkl", "./input/train_labels.csv")
imgs_train, y_train = train_data.get_datas(0, 35000)

def getBoundingRectArea(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w * h

def getBiggestDigit(im):
    img = np.array(im, dtype=np.uint8)
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    t, binary = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    max_area = 0
    for c in contours:
        area = getBoundingRectArea(c)
        print(area)
        if  area > max_area:
            max_area = area
            max_countour = c

    for i in range(64):
        for j in range(64):
            if cv2.pointPolygonTest(max_countour,(j,i),True) < 0:
                img[i,j] = 0
    return img

def getBiggestDigitOnTop(im):
    img = np.array(im, dtype=np.uint8)
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    t, binary = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    max_rect = (0,0,0,0)
    max_area = 0
    for c in contours:
        area, rect = getBoundingRectArea(c)
        print(area)
        if  area > max_area:
            max_area = area
            max_rect = rect

    pad = 2
    x, y, w, h = max_rect
    for i in range(64):
        for j in range(64):
            if not (j > x - pad and j < x + w + pad and i > y - pad and i < y + h + pad):
                img[i,j] = 0
    return img

for i in range(10):
    im = np.asarray(imgs_train[i], dtype=np.uint8)
    cim = getBiggestDigit(im)
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(im , cmap='gray')
    fig.add_subplot(1,2,2)
    plt.imshow(cim, cmap='gray')
    plt.show()