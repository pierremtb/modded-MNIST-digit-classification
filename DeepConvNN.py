# Michael Segev
# COMP 551 MP3
# March 3 2019

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from helpers import *

class DeepConvNN(torch.nn.Module):

    def __init__(self):
        super(DeepConvNN, self).__init__()  # call the inherited class constructor

        print("Model: DeepConvNN")
        # define the architecture of the neural network 
        # width_out = (width_in - kernel_size + 2*padding) / stride + 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),  # output is 60x60
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),  # output is 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # output is 28x28
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=2, stride=1),  # output is 29x29
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=1),  # output is 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1),  # output is 24x24
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1),  # output is 20x20
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # output is 10x10
        )
        self.linear1 = nn.Sequential(
            torch.nn.Linear(128*10*10, 1000),
            nn.ReLU(True)
        )
        self.linear2 = torch.nn.Linear(1000, 10)

        self.losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.loss_LPF = 2.3
        self.criterion = None
        self.optimizer = None

    def init_optimizer(self):
        # loss function
        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.criterion = torch.nn.CrossEntropyLoss()

        # optimizer
        lr = 1e-2
        print("Learning rate: {}".format(lr))
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = h.reshape(h.size(0), -1)
        h = self.linear1(h)
        y_pred = self.linear2(h)
        return y_pred

    def train_batch(self, x, y):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = self(x)

        # Compute and print loss
        loss = self.criterion(y_pred, y)
        self.losses.append(float(loss.data.item()))

        # Record accuracy
        total = y.size(0)
        _, predicted = torch.max(y_pred.data, 1)
        correct = (predicted == y).sum().item()
        acc = correct / total
        self.accuracies.append(acc)

        # Reset gradients to zero, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, acc

    def train_all_batches(self, x, y, batch_size, num_epochs, loss_target, device, x_val=[], y_val=[], val_skip=0):
        # figure out how many batches we can make
        num_batches = int(y.shape[0] / batch_size)
        last_batch_size = batch_size
        print("Number of batches = {}".format(num_batches))

        if y.shape[0] % batch_size != 0:
            num_batches += 1
            last_batch_size = y.shape[0] % batch_size

        for epoch in range(num_epochs):
            if self.loss_LPF < loss_target:
                print("reached loss target, ending early!")
                break
            for batch_num in range(num_batches):
                #  slice tensors according into requested batch
                if batch_num == num_batches - 1:
                    # last batch logic!
                    # print("Last batch!")
                    current_batch_size = last_batch_size
                else:
                    current_batch_size = batch_size

                x_batch = torch.tensor(
                    x[batch_num * current_batch_size:batch_num * current_batch_size + current_batch_size],
                    dtype=torch.float32, requires_grad=True, device=device)
                y_batch = torch.tensor(
                    y[batch_num * current_batch_size:batch_num * current_batch_size + current_batch_size],
                    dtype=torch.long, requires_grad=False, device=device)
                loss, acc = self.train_batch(x_batch, y_batch)
                self.loss_LPF = 0.01 * float(loss.data.item()) + 0.99*self.loss_LPF

                val_acc = 0
                if batch_num % ((val_skip + 1) * 40) == 0 and len(x_val) == len(y_val) and len(x_val) > 0:
                    val_acc = validate_data(self, x_val, y_val, device)
                    self.val_accuracies.append(val_acc)

                if batch_num % 40 == 0:
                    toPrint = "Epoch: {}, Loss: {}, Acc: {}%".format(epoch, self.loss_LPF, round(acc * 100, 3))
                    if (val_acc > 0):
                        toPrint += ", ValAcc: {}%".format(round(val_acc * 100, 3))
                    print(toPrint)
            

    def plot_loss(self):
        plt.title('Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.losses)
        plt.show()

    def plot_acc(self):
        plt.title('Accuracy over time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(self.accuracies)
        plt.plot(self.val_accuracies)
        plt.show()

