# Michael Segev
# COMP 551 MP3
# March 3 2019

from DataContainer import DataContainer
from ConvNN import ConvNN
import torch
from helpers import *

# load training data from files
train_data = DataContainer("./Data/train_images.pkl", "./Data/train_labels.csv")

# create model and load it on cuda core
model = ConvNN().cuda()
# model = SimpleNN(d_in=4096, h=2048, d_out=10)
model.init_optimizer()

imgs, labels = train_data.get_datas(0, 35000)

# convert labels to neural network format (1 output neuron per label)
label_array = labels_to_array(labels, 10)

# normalize images
imgs_norm = normalize_imgs(imgs)
print(imgs_norm.shape)
imgs_norm_ch = add_channel_to_imgs(imgs_norm)
print(imgs_norm_ch.shape)

model.train_all_batches(x=imgs_norm_ch, y=label_array, batch_size=64, num_epochs=10)

model.plot_loss()

# Validation step
imgs, labels = train_data.get_datas(35000, 5000)

# convert labels to neural network format (1 output neuron per label)
label_array = labels_to_array(labels, 10)

# normalize images
imgs_norm = normalize_imgs(imgs)
print(imgs_norm.shape)
imgs_norm_ch = add_channel_to_imgs(imgs_norm)
print(imgs_norm_ch.shape)

# create tensors and load them on cuda core
cuda0 = torch.device('cuda:0')
x_valid = torch.tensor(imgs_norm_ch, dtype=torch.float32, requires_grad=False, device=cuda0)
y_valid = torch.tensor(label_array, dtype=torch.float32, requires_grad=False, device=cuda0)
print(x_valid[0])
quit()
labels_predict = model(x_valid)

label_predict_max = []
label_validate_max = []
for label in labels_predict:
    value, idx = label.max(0)
    label_predict_max.append(idx)
    # print(label)
    # print(value)
    # print(idx)

for label in y_valid:
    value, idx = label.max(0)
    label_validate_max.append(idx)
    # print(label)

numCorrectPredictions = 0
for idx, prediction in enumerate(label_predict_max):
    if prediction == label_validate_max[idx]:
        numCorrectPredictions += 1
accuracy = numCorrectPredictions / len(label_predict_max)
print("accuracy is = " + str(accuracy * 100))
