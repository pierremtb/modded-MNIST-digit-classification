# Michael Segev
# COMP 551 MP3
# March 3 2019

import pandas as pd
import matplotlib.pyplot as plt
from helpers import *

class DataContainer:

    def __init__(self, img_pkl_file_path, label_csv_file_path=None):
        self.images = pd.read_pickle(img_pkl_file_path)
        if label_csv_file_path is not None:
            self.labels = pd.read_csv(label_csv_file_path)
        else:
            self.labels = None

    def plot_image(self, idx):
        if self.labels is not None:
            label = self.labels.iloc[idx]['Category']
        else:
            label = '?'

        plt.title('Image #{}    Label: {}'.format(idx, label))
        plt.imshow(self.images[idx])
        plt.show()

    def get_images(self, start_idx=0, num_images=-1):
        if num_images == -1:
            return self.images
        else:
            return self.images[start_idx:start_idx+num_images]

    def get_labels(self, start_idx=0, num_labels=-1):
        if self.labels is not None:
            if num_labels == -1:
                return self.labels['Category'].values
            else:
                return (self.labels.iloc[start_idx:start_idx+num_labels]['Category']).values

    def get_datas(self, start_idx=0, num_datas=-1):
        # returns a tuple containing the image and associated label
        if self.labels is not None:
            return self.get_images(start_idx, num_datas), self.get_labels(start_idx, num_datas)
        else:
            return self.get_images(start_idx, num_datas)