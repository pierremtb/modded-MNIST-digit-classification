import DataContainer

train_data = DataContainer.DataContainer("./Data/train_images.pkl", "./Data/train_labels.csv")
train_data.plot_image(10)
