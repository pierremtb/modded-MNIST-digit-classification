# Michael Segev
# Pierre Jacquier
# Albert Faucher
# Group 70
# COMP 551 MP3
# March 18 2019

from DataContainer import DataContainer
import pandas as pd
import pickle
from helpers import *
import matplotlib.pyplot as plt

# load training data from files
train_data = DataContainer("./input/train_images.pkl", "./input/train_labels.csv")

# get training data and val data
imgs, _ = train_data.get_datas(0, 40000)
fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(df.showBoundingBox(imgs[3445]))
plt.subplot(1,2,2)
plt.imshow(df.flagCropTight(imgs[3445]))
plt.show()