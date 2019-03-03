import numpy as np


def label_to_array(label, num_labels):
    #  takes an integer label and converts it to array
    array = np.zeros(num_labels)
    array[label] = 1.0
    return array

