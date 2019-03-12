# Mini Project 3: Code
Albert Faucher, Pierre Jacquier, Michael Segev (Group 70) — COMP551

## Models
### SimpleNN
Basic fully-connected model with two hidden layers. ReLU activation. Very low accuracy.

### ConvNN
Two convolutionnal layers, with batchnorm, max-pooling and ReLU activation, followed by three fully-connected layers. SGD optimizer. Reached 0.77066 on Kaggle (batchnorm made a big difference, dropout not at all).

### DeepConvNN [MVP]
Attempt at creating a deeper convolutional network. 6 conv layers, got almost 0.96 on Kaggle.

### DeeperConvNN
Experiment with 7 conv layer model, got Adam to work properly but only gets 92% or so.

### DigitDeepConvNN
Same as DeepConvNN but using boundinx-box preprocessing. WIP

## Packages used
- `timeit`
- `numpy`
- `torch`
- `random`
- `matplotlib`
- `os`
- `math`
- `pandas`
- `pickle`
- `cv2`

## Copyright
[MIT license](LICENSE.md)
