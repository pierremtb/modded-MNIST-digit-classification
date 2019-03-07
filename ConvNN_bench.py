# Michael Segev
# COMP 551 MP3
# March 3 2019

from DataContainer import DataContainer
from ConvNN import ConvNN
from helpers import *

# load training data from files
train_data = DataContainer("./input/train_images.pkl", "./input/train_labels.csv")

# create model and load it on cuda core
model = ConvNN().cuda()
model.init_optimizer()

# get training data
imgs, labels = train_data.get_datas(0, 35000)

# normalize and add channel to images
imgs_norm = normalize_imgs(imgs)
imgs_norm_ch = add_channel_to_imgs(imgs_norm)

# train model
model.train_all_batches(x=imgs_norm_ch, y=labels, batch_size=64, num_epochs=30, loss_target=0.001)

model.plot_loss()

# Get data for validation step
imgs, labels = train_data.get_datas(35000, 5000)

# normalize and add channel to images
imgs_norm = normalize_imgs(imgs)
imgs_norm_ch = add_channel_to_imgs(imgs_norm)

# validate model using validation data
accuracy = validate_data(model, imgs_norm_ch, labels)
print("accuracy is = " + str(accuracy * 100))

# Running model on test data and producing output
test_data = DataContainer("./input/test_images.pkl")
# get training data
test_imgs = test_data.get_datas()

# normalize and add channel to images
test_imgs_norm = normalize_imgs(test_imgs)
test_imgs_norm_ch = add_channel_to_imgs(test_imgs_norm)

y_predict_test = run_model_in_batches(model, test_imgs_norm_ch, 64)

out_csv_file = open('./output/submission.csv', 'x')
out_csv_file.write("Id,Category\n")
for idx, label in enumerate(y_predict_test):
    out_csv_file.write("{},{}\n".format(idx, label))

out_csv_file.close()
