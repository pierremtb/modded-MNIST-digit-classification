import numpy as np


def label_to_array(label, num_label_types):
    #  takes an integer label and converts it to array
    array = np.zeros(num_label_types)
    array[label] = 1.0
    return array


def labels_to_array(labels, num_label_types):
    label_array = np.empty((labels.shape[0], num_label_types))
    for idx, label in enumerate(labels):
        label_array[idx] = label_to_array(label, num_label_types)
    return label_array


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
    imgs_ch = np.reshape(imgs, (imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    return imgs_ch
