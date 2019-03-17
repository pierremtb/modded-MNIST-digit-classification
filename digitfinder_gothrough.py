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
# train_pp_data = DataContainer("./input/train_images_crop_tight.pkl", "./input/train_labels.csv")

imgs, labels = train_data.get_datas(35000, 5000)
# imgs_pp, labels = train_pp_data.get_datas(0, 4000)

# plt.imshow(imgs[8])
# plt.show()
# plt.imshow(flagCropTight(imgs[8]))
# plt.show()
# mmg.imsave("hey/8-1-in.png", flagCropTight(imgs[8]))
for i, img in enumerate(imgs):
    # mmg.imsave("/Users/pierremtb/hey/{}-{}-d.png".format(i, labels[i]), imgs_pp[i])
    mmg.imsave("/Users/pierremtb/hey2/{}-{}-c.png".format(i, labels[i]), flagCropTight(img))
    mmg.imsave("/Users/pierremtb/hey2/{}-{}-b.png".format(i, labels[i]), showBoundingBox(img))
    mmg.imsave("/Users/pierremtb/hey2/{}-{}-a.png".format(i, labels[i]), img)
