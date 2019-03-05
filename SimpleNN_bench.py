# Michael Segev
# COMP 551 MP3
# March 3 2019

from DataContainer import DataContainer
from SimpleNN import SimpleNN
import torch
from helpers import *
import numpy as np

# load training data from files
train_data = DataContainer("./Data/train_images.pkl", "./Data/train_labels.csv")

# create model and load it on cuda core
model = SimpleNN(d_in=4096, h=200, d_out=10).cuda()
# model = SimpleNN(d_in=4096, h=2048, d_out=10)
model.init_optimizer()

imgs, labels = train_data.get_datas(0, 35000)

# convert labels to neural network format (1 output neuron per label)
label_array = labels_to_array(labels, 10)

# flatten and normalize image since we have a fully connected model
imgs_flatten = flatten_imgs(imgs)

model.train_all_batches(x=imgs_flatten, y=label_array, batch_size=64, num_epochs=50)

model.plot_loss()

# Validation step
imgs, labels = train_data.get_datas(35000, 5000)

# convert labels to neural network format (1 output neuron per label)
label_array = labels_to_array(labels, 10)

# flatten and normalize image since we have a fully connected model
imgs_flatten = flatten_imgs(imgs)

# create tensors and load them on cuda core
cuda0 = torch.device('cuda:0')
x_valid = torch.tensor(imgs_flatten,
                       dtype=torch.float32, requires_grad=True, device=cuda0)
y_valid = torch.tensor(label_array,
                       dtype=torch.float32, requires_grad=True, device=cuda0)
labels_predict = model(x_valid)

label_predict_max = []
label_validate_max = []
for label in labels_predict:
    # print(label)
    value, idx = label.max(0)
    label_predict_max.append(idx)
    # print(value)
    # print(idx)

for label in y_valid:
    # print(label)
    value, idx = label.max(0)
    label_validate_max.append(idx)

numCorrectPredictions = 0
for idx, prediction in enumerate(label_predict_max):
    if prediction == label_validate_max[idx]:
        numCorrectPredictions += 1
accuracy = numCorrectPredictions / len(label_predict_max)
print("accuracy is = " + str(accuracy * 100))
