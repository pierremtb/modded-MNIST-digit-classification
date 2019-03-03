import pandas as pd
import matplotlib.pyplot as plt


class DataContainer:

    def __init__(self, img_pkl_file_path, label_csv_file_path=None):
        self.images = pd.read_pickle(img_pkl_file_path)
        if label_csv_file_path is not None:
            self.labels = pd.read_csv(label_csv_file_path)
        else:
            self.labels = None

    def plot_image(self, idx):
        plt.title('Image #{}    Label: {}'.format(idx, self.labels.iloc[idx]['Category']))
        plt.imshow(self.images[idx])
        plt.show()

    def get_image(self, idx):
        return self.images[idx]

    def get_label(self, idx):
        if self.labels is not None:
            return self.labels.iloc[idx]['Category']

    def get_data(self, idx):
        # returns a tuple containing the image and associated label
        if self.labels is not None:
            return self.get_image(idx), self.get_label(idx)
        else:
            return self.get_image(idx)
