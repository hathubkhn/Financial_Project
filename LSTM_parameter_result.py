# How to load and use weights from a checkpoint
from keras.callbacks import ModelCheckpoint
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

time_steps = 20

mean_train = 0.0005461857129680816
std_train = 0.02167197278763545


def scale_test(a):
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
    # output = np.reshape(output, (len(output), 1))

    # each companies dataset as each row of matrix input and output
    X = []
    y = []
    for i in range(time_steps, len_train):
        X.append(input[i - time_steps:i])
        y.append(output[i - 1])
    X, y = np.array(X), np.array(y)
    return X, y


file_path_index_test = "D:\Lab\Stock\KOSPI200_index\KOSPI_test.csv"

# Create input-output for test
X_test, y_test = create_data("D:\\Lab\\Stock\\KOSPI200_dataset_test_label\\삼성전자.csv")
X_test_index, y_test_index = create_data(file_path_index_test)

X_test = np.concatenate((X_test, X_test_index), axis=2)


from keras.models import load_model

from keras.models import Model

# model = load_model('D:\Lab\Financial project\Save_Model\model_hyundai_othercompany-3260.h5')
model = load_model(
    'D:\\Lab\\Financial project\\Save_similar\\model_Com1_KOSPI_checkpoint_16_16_256.h5')

scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
y_pred = model.predict_classes(X_test)

confusion = confusion_matrix(y_test, y_pred)
y_pred = y_pred.ravel()
precision = confusion[0][0] / (confusion[0][0] + confusion[0][1])
recall = confusion[0][0] / (confusion[0][0] + confusion[1][0])
F_score = 2 * precision * recall / (precision + recall)
print(precision)
print(recall)
print(F_score)