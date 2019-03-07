# Michael Segev
# COMP 551 MP3
# March 3 2019

from DataContainer import DataContainer
from SimpleNN import SimpleNN
import torch
from helpers import *
import numpy as np
from timeit import default_timer as timer

# load training data from files
train_data = DataContainer("./input/train_images.pkl", "./input/train_labels.csv")

# create model and load it on cuda core
model = SimpleNN(d_in=4096, h=200, d_out=10)
model.cuda()
# model = SimpleNN(d_in=4096, h=2048, d_out=10)
model.init_optimizer()

imgs, labels = train_data.get_datas(0, 35000)

# convert labels to neural network format (1 output neuron per label)
label_array = labels_to_array(labels, 10)

# flatten and normalize image since we have a fully connected model
imgs_flatten = flatten_imgs(imgs)

t = timer()
model.train_all_batches(x=imgs_flatten, y=label_array, batch_size=64, num_epochs=10)
endTimer("Training", t)

model.plot_loss()

# Validation step
imgs, labels = train_data.get_datas(35000, 5000)

# convert labels to neural network format (1 output neuron per label)
label_array = labels_to_array(labels, 10)

# flatten and normalize image since we have a fully connected model
imgs_flatten = flatten_imgs(imgs)

# validate model using validation data
accuracy = validate_data(model, imgs_flatten, labels)
print("accuracy is = " + str(accuracy * 100))