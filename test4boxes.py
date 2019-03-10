import numpy as np
import cv2
from matplotlib import pyplot as plt
from DataContainer import DataContainer
from DeepConvNN import DeepConvNN
from helpers import *

train_data = DataContainer("./input/train_images.pkl", "./input/train_labels.csv")
imgs_train, y_train = train_data.get_datas(0, 35000)
print(imgs_train[0])

img = np.array(imgs_train[0], dtype = np.uint8)
blur = cv2.GaussianBlur(img, (5, 5), 0)
t, binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))

rects = []
for c in contours:
    rects.append(cv2.boundingRect(c))

print(rects)
x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.putText(img,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
cv2.imshow("Show",img)
cv2.waitKey()  
cv2.destroyAllWindows()