# Michael Segev
# COMP 551 MP3
# March 3 2019

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


class SimpleNN(torch.nn.Module):

    def __init__(self, d_in, h, d_out):
        super(SimpleNN, self).__init__()  # call the inherited class constructor

        # define the architecture of the neural network
        self.linear1 = torch.nn.Linear(d_in, h)
        self.linear2 = torch.nn.Linear(h, int(h/2))
        self.linear3 = torch.nn.Linear(int(h/2), d_out)
        self.losses = []
        self.criterion = None
        self.optimizer = None

    def init_optimizer(self):
        # define trainer
        # loss function
        self.criterion = torch.nn.MSELoss(reduction='sum')
        # self.criterion = torch.nn.CrossEntropyLoss()
        # optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)

    def forward(self, x):
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        y_pred = torch.sigmoid(self.linear3(h2))
        return y_pred

    def train_batch(self, x, y):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = self(x)
        # Compute and print loss
        loss = self.criterion(y_pred, y)
        self.losses.append(loss.data.item())
        # Reset gradients to zero, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def plot_loss(self):
        plt.title('Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.losses)
        plt.show()
