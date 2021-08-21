import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pdb
from keras.utils import to_categorical

time_steps = 20


# Function to read file
def read_file(ffile):
    with open(ffile + '.csv', 'r') as f:
        reader = csv.reader(f)
        examples = list(reader)
    list_ = []
    for row in examples:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        list_.append(nwrow)
    return list_


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


# Load file
X_train_index, y_train_index = create_data("D:\\Lab\\Stock\\KOSPI200_index\\KOSPI_train.csv")
X_val_index, y_val_index = create_data("D:\\Lab\\Stock\\KOSPI200_index\\KOSPI_val.csv")
X_test_index, y_val_index = create_data("D:\\Lab\\Stock\\KOSPI200_index\\KOSPI_test.csv")

X_train, y_train = create_data("D:\\Lab\\Stock\\KOSPI200_dataset_train_label\\삼성전자.csv")
X_val, y_val = create_data("D:\\Lab\Stock\\KOSPI200_dataset_val_label\\삼성전자.csv")
X_test, y_test = create_data("D:\\Lab\\Stock\\KOSPI200_dataset_test_label\\삼성전자.csv")

X_train = np.concatenate((X_train, X_train_index), axis=2)
X_val = np.concatenate((X_val, X_val_index), axis=2)
X_test = np.concatenate((X_test, X_test_index), axis=2)

# Create model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import LSTM
from keras import optimizers
from sklearn.metrics import confusion_matrix
from keras import regularizers
import keras.utils
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy

from keras.models import load_model
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU, PReLU

classification = Sequential()
classification.add(LSTM(16, input_shape=(time_steps, 2), kernel_initializer='he_normal',return_sequences=True))
classification.add(Dropout(0.6))
classification.add(LSTM(16, kernel_initializer='he_normal', return_sequences=False))
classification.add(Dropout(0.5))
classification.add(Dense(16, kernel_initializer='he_normal', activation='elu'))
classification.add(Dropout(0.6))
classification.add(Dense(16, kernel_initializer='he_normal', activation='elu'))
classification.add(Dropout(0.5))
classification.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))

opt = optimizers.RMSprop(0.001)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                               patience=10, min_lr=0.001)

checkpoint = ModelCheckpoint(
    'D:\Lab\Financial project\Save_similar\model_Com1_KOSPI_checkpoint_16_16_{epoch:03d}.h5',
    verbose=1, monitor='val_acc', save_best_only=True, mode='auto')

classification.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = classification.fit(X_train, y_train, epochs=5000, verbose=1, batch_size=718,
                             validation_data=(X_val, y_val), callbacks=[checkpoint])

history_dict = history.history
history_dict.keys()
plt.clf()

import matplotlib.pyplot as plt

plt.plot(history_dict['acc'])
plt.plot(history_dict['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_dict['loss'])
plt.plot(history_dict['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()
