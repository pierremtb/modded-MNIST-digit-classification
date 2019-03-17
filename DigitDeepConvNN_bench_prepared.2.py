# Michael Segev
# COMP 551 MP3
# March 3 2019

from DataContainer import DataContainer
from SlightlyDeepConvNN import SlightlyDeepConvNN
from helpers import *
import matplotlib.pyplot as plt

# auto fallback to cpu
device = getDevice()

# load training data from files
train_data = DataContainer("./input/train_images_crop_tight.pkl", "./input/train_labels.csv")

# create model and load it on cuda core
model = SlightlyDeepConvNN().to(device)
model.init_optimizer()

# get training data and val data
imgs_train, y_train = train_data.get_datas(0, 35000)
imgs_val, y_val = train_data.get_datas(35000, 5000)
# x_train = preprocess(imgs_train, find_digit=True, flag=df.CROP_TIGHT, print_first=True)
# x_val = preprocess(imgs_val, find_digit=True, flag=df.CROP_TIGHT)
x_train = preprocess(imgs_train)
x_val = preprocess(imgs_val)


# train model
model.train_all_batches(
    x=x_train, y=y_train,
    batch_size=64, num_epochs=30, loss_target=0.001,
    device=device,
    x_val=x_val, y_val=y_val, val_skip=5
)
model.plot_loss()
model.plot_acc()

# validate model using validation data
accuracy = validate_data(model, x_val, y_val, device)
print("\n Validation accuracy is: {}%".format(round(accuracy * 100, 3)))

# run model on test data and producing output
test_data = DataContainer("./input/test_images.pkl")
x_test = preprocess(test_data.get_datas(), find_digit=True, flag=df.CROP_TIGHT)
y_predict_test = run_model_in_batches(model, x_test, 16, device)

out_csv_file = open('./output/submission.csv', 'w')
out_csv_file.write("Id,Category\n")
for idx, label in enumerate(y_predict_test):
    out_csv_file.write("{},{}\n".format(idx, label))

out_csv_file.close()
