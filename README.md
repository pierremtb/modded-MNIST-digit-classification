# Mini Project 3: Code
Albert Faucher, Pierre Jacquier, Michael Segev (Group 70) — COMP551

## Models
### SimpleNN
Basic fully-connected model with two hidden layers. ReLU activation. Very low accuracy.
Run SimpleNN_bench.py to replicate results.

### ConvNN
Two convolutionnal layers, with batchnorm, max-pooling and ReLU activation, followed by three fully-connected layers. SGD optimizer. Reached 0.77066 on Kaggle (batchnorm made a big difference, dropout not at all).
Run ConvNN_bench.py to replicate results.

### DeepConvNN [MVP]
Attempt at creating a deeper convolutional network. 6 conv layers, got almost 0.96 on Kaggle.
Run DeepConvNN_bench.py to replicate results.

### DeeperConvNN
Experiment with 7 conv layer model, got Adam to work properly but only gets 92% or so.
Run DeeperConvNN_bench.py to replicate results.

### DigitDeepConvNN
Same as DeepConvNN but using bounding-box preprocessing. Three methods were proposed, with CROP_TIGHT being the default and best-performing one.
Run DigitDeepConvNN_bench.py to replicate results.

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
