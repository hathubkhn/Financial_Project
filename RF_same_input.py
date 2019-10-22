from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot
import os
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import pandas as pd
import pdb
import csv

time_steps = 20


def standard_scale(a):
    mean = np.mean(a)
    std = np.std(a)
    a = (a - mean) / std
    return a


def create_data(file_path):
    train = pd.read_csv(file_path, usecols=[2, 3])
    len_train = len(train)
    input = train.iloc[:, 0:-1]
    output = train.iloc[:, -1]

    input = input.as_matrix()
    output = output.as_matrix()
    # each companies dataset as each row of matrix input and output
    X_train = []
    y_train = []
    for i in range(time_steps, len_train):
        X_train.append(input[i - time_steps:i])
        y_train.append(output[i - 1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train


path_train = "D:\\Lab\\Stock\\KOSPI200_dataset_train_label\\SK하이닉스.csv"
path_val = "D:\\Lab\\Stock\\KOSPI200_dataset_val_label\\SK하이닉스.csv"
path_test = "D:\\Lab\\Stock\\KOSPI200_dataset_test_label\\SK하이닉스.csv"

train_input, train_output = create_data(path_train)
train_input = train_input.reshape(train_input.shape[0],train_input.shape[1])
train_output = train_output.ravel()
val_input, val_output = create_data(path_val)
val_input = val_input.reshape(val_input.shape[0],val_input.shape[1])
val_output = val_output.ravel()
test_input, test_output = create_data(path_test)
test_input = test_input.reshape(test_input.shape[0],test_input.shape[1])
test_output = test_output.ravel()

# pdb.set_trace()
from sklearn.metrics import confusion_matrix

# Random Forest model
from sklearn.ensemble import RandomForestClassifier
def random_forest_model(n_trees,depth):
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth,random_state=0)
    rf.fit(train_input, train_output)
    y_pred_train = rf.predict(train_input)
    train_acc = accuracy_score(train_output,y_pred_train)

    y_pred_val = rf.predict(val_input)
    val_acc = accuracy_score(val_output, y_pred_val)

    y_pred_test = rf.predict(test_input)
    test_acc = accuracy_score(test_output,y_pred_test)
    confusion = confusion_matrix(test_output, y_pred_test)
    precision = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    recall = confusion[0][0] / (confusion[0][0] + confusion[1][0])
    return train_acc,val_acc,test_acc,precision,recall

N_trees = list(range(10,500,10))
Max_depth = list(range(2,22,1))
_train_acc, _val_acc, _test_acc, _precision, _recall = [], [],[], [],[]
with  open('random_forest_result_same_input_Com_5_2.txt','w') as f:
    f.write('[N_trees, Max_depth, train_acc, val_acc, test_acc, precision, recall]')
    f.write('\n')
    for n_trees in N_trees:
        for max_depth in Max_depth:
            train_acc, val_acc, test_acc, precision, recall = random_forest_model(n_trees,max_depth)
            _train_acc.append(train_acc)
            _val_acc.append(val_acc)
            _test_acc.append(test_acc)
            _precision.append(precision)
            _recall.append(recall)
            f.write('\n')
            f.write('[{}, {}, {}, {}, {}, {}, {}]'.format(n_trees,max_depth,train_acc,val_acc,test_acc,precision,recall))

m = max(_val_acc)
print(m)