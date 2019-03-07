# Michael Segev
# COMP 551 MP3
# March 3 2019

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


class ConvNN(torch.nn.Module):

    def __init__(self):
        super(ConvNN, self).__init__()  # call the inherited class constructor

        # define the architecture of the neural network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)  # output is 60x60
        self.pool = nn.MaxPool2d(2, 2)  # output is 30x30
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)  # output is 26x26
        # after another max pooling we get 13x13 matrix per channel
        self.linear1 = torch.nn.Linear(64*13*13, 1000)
        self.linear2 = torch.nn.Linear(1000, 200)
        self.linear3 = torch.nn.Linear(200, 10)
        self.losses = []
        self.accuracies = []
        self.loss_LPF = 2.3
        self.criterion = None
        self.optimizer = None

    def init_optimizer(self):
        # define trainer
        # loss function
        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.criterion = torch.nn.CrossEntropyLoss()
        # optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 13 * 13)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = self.linear3(x)
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

    def train_all_batches(self, x, y, batch_size, num_epochs, loss_target, device):
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
                if batch_num % 40 == 0:
                    print("Epoch: {}, Loss: {}, Acc: {}%".format(epoch, self.loss_LPF, round(acc * 100, 3)))

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
        plt.show()
