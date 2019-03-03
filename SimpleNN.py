# Michael Segev
# COMP 551 MP3
# March 3 2019

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SimpleNN(torch.nn.Module):

    def __init__(self, d_in, h, d_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        Args:
            - D_in : input dimension of the data
            - H : size of the first hidden layer
            - D_out : size of the output/ second layer
        """
        super(SimpleNN, self).__init__()  # call the inherited class constructor

        # define the architecture of the neural network
        self.linear1 = torch.nn.Linear(d_in, h)  # create a linear layer
        self.linear2 = torch.nn.Linear(h, d_out)

        # define trainer
        # loss function
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.losses = []
        # optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and
        return a tensor of output data. We can use
        Modules defined in the constructor as well as arbitrary
        operators on Variables.
        """
        h1 = self.linear1(x)
        y_pred = self.linear2(h1)
        return y_pred

    def train_batch(self, x, y):
        self.losses = []
        for epoch in range(50):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self(x)

            # Compute and print loss
            loss = self.criterion(y_pred, y)
            self.losses.append(loss.data.item())
            print(f"Epoch : {epoch}    Loss : {loss.data.item()}")

            # Reset gradients to zero, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def plot_loss(self):
        plt.title('Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.losses)
        plt.show()
