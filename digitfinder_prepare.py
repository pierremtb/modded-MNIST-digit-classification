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