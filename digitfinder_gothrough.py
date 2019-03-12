from DataContainer import DataContainer
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from helpers import *
from digitfinder import *
import time

# load training data from files
train_data = DataContainer("./input/train_images_crop_tight.pkl", "./input/train_labels.csv")

imgs, labels = train_data.get_datas(0, 500)
imgs = preprocess(imgs, find_digit=True, flag=CROP_TIGHT)
for i, img in enumerate(imgs):
    print(img)
    plt.imsave("hey/{}-{}.png".format(i, labels[i]), img)
    
