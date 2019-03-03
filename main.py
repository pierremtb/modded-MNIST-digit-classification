# Michael Segev
# COMP 551 MP3
# March 3 2019

from DataContainer import DataContainer
from SimpleNN import SimpleNN
import torch
from helpers import label_to_array
import numpy as np

# load training data from files
train_data = DataContainer("./Data/train_images.pkl", "./Data/train_labels.csv")

# create model and load it on cuda core
model = SimpleNN(d_in=4096, h=100, d_out=10).cuda()

# batch train the model
batch_size = 64

# get required number of samples for this batch
imgs, labels = train_data.get_datas(0, batch_size)

# convert labels to neural network format (1 output neuron per label)
# TODO: move this to DataContainer class...do we ever want the numerical label?
label_array = np.empty((batch_size, 10))
for idx, label in enumerate(labels):
    label_array[idx] = label_to_array(label, 10)

# flatten image since we have a fully connected model
imgs_flatten = np.empty((imgs.shape[0], imgs.shape[1]*imgs.shape[2]))
for idx, image in enumerate(imgs):
    imgs_flatten[idx] = image.ravel()

# create tensors from batch and load them on cuda core
cuda0 = torch.device('cuda:0')
x_batch = torch.tensor(imgs_flatten, dtype=torch.float32, requires_grad=False, device=cuda0)
y_batch = torch.tensor(label_array, dtype=torch.float32, requires_grad=False, device=cuda0)

# train model using batch
model.train_batch(x_batch, y_batch)
model.plot_loss()
