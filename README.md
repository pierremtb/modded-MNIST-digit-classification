# Mini Project 3: Code
Albert Faucher, Pierre Jacquier, Michael Segev (Group 70) — COMP551

## Models
### SimpleNN
Basic fully-connected model with two hidden layers. ReLU activation. Very low accuracy.

### ConvNN
Two convolutionnal layers, with batchnorm, max-pooling and ReLU activation, followed by three fully-connected layers. Adam optimizer. Reached 0.77066 on Kaggle (batchnorm made a big difference, dropout not at all).

### DeepConvNN
Attempt at creating a deeper convolutional network. WIP.

## Packages used
- `timeit`
- `numpy`
- `torch`
- `random`
- `matplotlib`
- `os`
- `math`
- `pandas`

## Copyright
[MIT license](LICENSE.md)
