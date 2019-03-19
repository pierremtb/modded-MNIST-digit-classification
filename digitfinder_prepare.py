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

# load training data from files
train_data = DataContainer("./input/train_images.pkl", "./input/train_labels.csv")


# get training data and val data
imgs, _ = train_data.get_datas(0, 40000)
for i in range(len(imgs)):
    imgs[i] = df.flagCropTight(imgs[i])

with open("./input/train_images_crop_tight.pkl", "wb") as f:
    pickle.dump(imgs, f)