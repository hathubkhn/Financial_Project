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

time_steps = 240


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



# Load file
train_input_file = "D:\Lab\Stock\File\Study3\SP500\Baseline_data\LSTM_base\Train_LSTM_input"
train_output_file = "D:\Lab\Stock\File\Study3\SP500\Baseline_data\LSTM_base\Train_LSTM_output"

train_input = np.array(read_file(train_input_file))
train_output = np.array(read_file(train_output_file))
train_output = to_categorical(train_output)

# Create model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import LSTM
from keras import optimizers
from sklearn.metrics import confusion_matrix

from keras.callbacks import EarlyStopping

classification = Sequential()
classification.add(LSTM(25, input_shape=(time_steps, 1), return_sequences=False))
classification.add(Dropout(0.1))
classification.add(Dense(2, activation='softmax'))

opt = optimizers.RMSprop(0.001)
early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=10,
                           verbose=0, mode='auto')

checkpoint = ModelCheckpoint(
    'D:\Lab\Financial project\Baseline\Save_LSTM_checkpoint\Study3\model_LSTM_baseline_SP500_Study3_checkpoint_{epoch:03d}.h5',
    verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

classification.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = classification.fit(train_input, train_output, epochs=1000, verbose=1, batch_size=512,
                             validation_split=0.2, callbacks=[early_stop, checkpoint])

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
