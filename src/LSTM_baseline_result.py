# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras import optimizers
from keras.utils import to_categorical
from numpy.random import seed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
import pdb
from sklearn.model_selection import train_test_split

time_steps = 240


def scale_test(a, mean_train, std_train):
    a = (a - mean_train) / std_train
    return a


# Create file test

def create_data(file_path):
    train = pd.read_csv(file_path, usecols=[2, 3])
    len_train = len(train)
    input = train.iloc[:, 0:-1]
    output = train.iloc[:, -1]

    input = input.as_matrix()
    output = output.as_matrix()

    # each companies dataset as each row of matrix input and output
    X = []
    y = []
    for i in range(time_steps, len_train):
        X.append(input[i - time_steps:i])
        y.append(output[i - 1])
    X, y = np.array(X), np.array(y)
    return X, y


# Create input-output for test
X_train, y_train = create_data("D:\\Lab\\Stock\\File\\Baseline_data\\LSTM_base\\KOSPI200_dataset_test\\LG화학.csv")
X_test, y_test = create_data("D:\\Lab\\Stock\\File\\Baseline_data\\LSTM_base\\KOSPI200_dataset_test\\LG화학.csv")
mean_train = np.mean(X_train)
std_train = np.std(X_train)
X_test = scale_test(X_test, mean_train, std_train)
y_test_1 = to_categorical(y_test)

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau

model = load_model('D:\\Lab\\Financial project\\Save_Model\\model_LSTM_baseline_checkpoint_027.h5')

scores = model.evaluate(X_test, y_test_1, verbose=0)

y_pred = model.predict_classes(X_test)
print(y_pred)
print(y_pred.shape)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

confusion = confusion_matrix(y_test, y_pred)
print(confusion)
y_pred = y_pred.ravel()
precision = confusion[0][0] / (confusion[0][0] + confusion[0][1])
recall = confusion[0][0] / (confusion[0][0] + confusion[1][0])
print(precision)
print(recall)
