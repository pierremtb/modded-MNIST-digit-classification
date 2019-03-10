from timeit import default_timer as timer
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def endTimer(name, t):
    print("{0}: {1} s\n".format(name, round(timer() - t, 3)))

def label_to_array(label, num_label_types):
    #  takes an integer label and converts it to array
    array = np.zeros(num_label_types)
    array[label] = 1
    return array


def labels_to_array(labels, num_label_types):
    label_array = np.empty((labels.shape[0], num_label_types))
    for idx, label in enumerate(labels):
        label_array[idx] = label_to_array(label, num_label_types)
    return label_array


def array_to_labels(labels_array):
    labels = []
    for label_array in labels_array:
        value, idx = label_array.max(0)
        labels.append(idx)
    return labels


def flatten_imgs(imgs):
    # flatten and normalize image since we have a fully connected model
    imgs_flatten = np.empty((imgs.shape[0], imgs.shape[1] * imgs.shape[2]))
    for idx, image in enumerate(imgs):
        imgs_flatten[idx] = (image.ravel() / 255.0)  # normalize the data
        # imgs_flatten[idx] = image.ravel()  # normalize the data
    return imgs_flatten


def normalize_imgs(imgs):
    imgs_norm = np.empty(imgs.shape)
    for idx, image in enumerate(imgs):
        imgs_norm[idx] = (image / 255.0)  # normalize the data
    return imgs_norm


def add_channel_to_imgs(imgs):
    # since image is greyscale we only have 1 channel
    imgs_ch = np.reshape(imgs, (imgs.shape[0], 1, imgs.shape[1], imgs.shape[2]))
    return imgs_ch

def getBoundingRectArea(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w * h

def getBiggestDigit(im):
    img = np.array(im, dtype=np.uint8)
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    t, binary = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    max_area = 0
    for c in contours:
        area = getBoundingRectArea(c)
        if  area > max_area:
            max_area = area
            max_countour = c

    for i in range(64):
        for j in range(64):
            if cv2.pointPolygonTest(max_countour,(j,i),True) < 0:
                img[i,j] = 0
    return img

def preprocess(imgs, find_digit=False):
    # normalize and add channel to images
    print("find_digit: {}".format(find_digit))
    if find_digit:
        for i in range(len(imgs)):
            imgs[i] = getBiggestDigit(imgs[i])
    imgs_norm = normalize_imgs(imgs)
    imgs_norm_ch = add_channel_to_imgs(imgs_norm)
    return imgs_norm_ch

def validate_data(model, x, y, device):
    # validate in batches to save on gpu memory usage
    # figure out how many batches we can make
    batch_size = 64
    num_batches = int(y.shape[0] / batch_size)
    last_batch_size = batch_size
    # print("Number of validation batches = {}".format(num_batches))

    if y.shape[0] % batch_size != 0:
        num_batches += 1
        last_batch_size = y.shape[0] % batch_size

    numCorrectPredictions = 0
    totalSamples = 0
    for batch_num in range(num_batches):
        #  slice tensors according into requested batch
        if batch_num == num_batches - 1:
            # last batch logic!
            # print("Last batch!")
            current_batch_size = last_batch_size
        else:
            current_batch_size = batch_size

        x_batch_valid = torch.tensor(
            x[batch_num * current_batch_size:batch_num * current_batch_size + current_batch_size],
            dtype=torch.float32, requires_grad=True, device=device)
        y_batch_valid = torch.tensor(
            y[batch_num * current_batch_size:batch_num * current_batch_size + current_batch_size],
            dtype=torch.long, requires_grad=False, device=device)
        y_batch_predict_array = model(x_batch_valid)

        # model outputs array of 10 scores, max value idx is the predicted class
        y_predict = array_to_labels(y_batch_predict_array)

        for idx, prediction in enumerate(y_predict):
            totalSamples += 1
            if prediction == y_batch_valid[idx]:
                numCorrectPredictions += 1
    accuracy = numCorrectPredictions / totalSamples
    return accuracy

def getDevice():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Running on:")
    print(device)
    print()
    return device

def run_model_in_batches(model, x, batch_size, device):
    # figure out how many batches we can make
    num_batches = int(x.shape[0] / batch_size)
    last_batch_size = batch_size
    y_predict = torch.zeros(x.shape[0], dtype=torch.long)
    idx = 0
    for batch_num in range(num_batches):
        #  slice tensors according into requested batch
        if batch_num == num_batches - 1:
            # last batch logic!
            # print("Last batch!")
            current_batch_size = last_batch_size
        else:
            current_batch_size = batch_size

        x_batch_valid = torch.tensor(
            x[batch_num * current_batch_size:batch_num * current_batch_size + current_batch_size],
            dtype=torch.float32, requires_grad=True, device=device)

        y_batch_predict_array = model(x_batch_valid)

        # model outputs array of 10 scores, max value idx is the predicted class
        y_predict_batch = array_to_labels(y_batch_predict_array)

        for prediction in y_predict_batch:
            y_predict[idx] = prediction
            idx += 1

    return y_predict
