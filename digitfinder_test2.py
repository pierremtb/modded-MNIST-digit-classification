# Michael Segev
# Pierre Jacquier
# Albert Faucher
# Group 70
# COMP 551 MP3
# March 18 2019

from DataContainer import DataContainer
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mmg
from helpers import *
from digitfinder import *
import time

# load training data from files
train_data = DataContainer("./input/train_images.pkl", "./input/train_labels.csv")

imgs, labels = train_data.get_datas(35000, 5000)

for i, img in enumerate(imgs):
    mmg.imsave("./{}-{}-c.png".format(i, labels[i]), flagCropTight(img))
    mmg.imsave("./{}-{}-b.png".format(i, labels[i]), showBoundingBox(img))
    mmg.imsave("./{}-{}-a.png".format(i, labels[i]), img)
