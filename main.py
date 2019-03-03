from DataContainer import DataContainer
from SimpleNN import SimpleNN
import torch

#
# x = torch.tensor([2.0], requires_grad=True)
#
# print(x)
#
# y = x**3
#
# print(y)
#
# y.backward()
#
# print(x.grad)

# x = x.cuda()
# print(x)


#
# train_data = DataContainer.DataContainer("./Data/train_images.pkl", "./Data/train_labels.csv")
# train_data.plot_image(10)

N, D_in, H, D_out = 64, 1000, 100, 10
# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out, requires_grad=False)

model = SimpleNN(d_in=1000, h=100, d_out=10)

model.train(x, y)
model.plot_loss()
