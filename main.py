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
model = SimpleNN(d_in=4096, h=2048, d_out=10).cuda()
# model = SimpleNN(d_in=4096, h=2048, d_out=10)
model.init_optimizer()

imgs, labels = train_data.get_datas()

# convert labels to neural network format (1 output neuron per label)
# TODO: move this to DataContainer class...do we ever want the numerical label?
label_array = np.empty((labels.shape[0], 10))
for idx, label in enumerate(labels):
    label_array[idx] = label_to_array(label, 10)

# flatten and normalize image since we have a fully connected model
imgs_flatten = np.empty((imgs.shape[0], imgs.shape[1]*imgs.shape[2]))
for idx, image in enumerate(imgs):
    imgs_flatten[idx] = (image.ravel()/255.0)  # normalize the data
    # imgs_flatten[idx] = image.ravel()  # normalize the data
# create tensors and load them on cuda core
cuda0 = torch.device('cuda:0')
# cuda0 = torch.device('cpu')

# batch train the model
batch_size = 64

# figure out how many batches we can make
num_batches = int(labels.shape[0]/batch_size)
last_batch_size = batch_size
print("Number of batches = {}".format(num_batches))

if labels.shape[0] % batch_size != 0:
    num_batches += 1
    last_batch_size = labels.shape[0] % batch_size

for epoch in range(100):
    for batch_num in range(num_batches):
        #  slice tensors according into requested batch
        if batch_num == num_batches - 1:
            # last batch logic!
            # print("Last batch!")
            current_batch_size = last_batch_size
        else:
            current_batch_size = batch_size

        x_batch = torch.tensor(imgs_flatten[batch_num*current_batch_size:batch_num*current_batch_size+current_batch_size],
                                   dtype=torch.float32, requires_grad=True, device=cuda0)
        y_batch = torch.tensor(label_array[batch_num*current_batch_size:batch_num*current_batch_size+current_batch_size],
                                     dtype=torch.float32, requires_grad=True, device=cuda0)
        loss = model.train_batch(x_batch, y_batch)
        print("Epoch: {} Loss : {}".format(epoch, loss.data.item()))

model.plot_loss()


# Validation step
# TODO: don't use training data as validation!
x_valid = torch.tensor(imgs_flatten[5000:6000],
                       dtype=torch.float32, requires_grad=True, device=cuda0)
y_valid = torch.tensor(label_array[5000:6000],
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

print(label_validate_max)

numCorrectPredictions = 0
for idx, prediction in enumerate(label_predict_max):
    if prediction == label_validate_max[idx]:
        numCorrectPredictions += 1
accuracy = numCorrectPredictions / len(label_predict_max)
print("accuracy is = " + str(accuracy * 100))
