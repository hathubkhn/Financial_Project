import pandas as pd
import os
import csv

pd.options.mode.chained_assignment = None
import numpy as np

# Folder include dataset
FILE_DIR = 'D:\\Lab\\Stock\\KOSPI200\\New folder'


# Create list of stock to get directory of file
def create_list_stock(dirName):
    listofFile = os.listdir(dirName)
    allFiles = list()
    for entry in listofFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + create_list_stock(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


listofFiles = create_list_stock(FILE_DIR)


# Clean dataset
def delete_row(dirName):
    a = pd.read_csv(dirName)
    a = a.as_matrix()
    for i in range(a.shape[0]):
        if a[i, 6] == 0:
            b = np.delete(a, (i), axis=0)
    df = pd.DataFrame(b)
    df.to_csv(dirName, header=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], index=False,
              index_label=None)


# delete_row("D:\Lab\Stock\KOSPI200_dataset_train_label_1\POSCO.csv")
# Calculate label:
# if close of next day > close of today => return 1
# if close of next day < close of today => return 0
def calculateLabel(df):
    label = []
    for i in range(0, df.index[-1]):
        return_stock = (df.loc[i + 1, 'Close'] - df.loc[i, 'Close'])
        if return_stock > 0:
            label.append(1)
        if return_stock <= 0:
            label.append(0)
    label = pd.Series(label)
    return label.to_frame()


# Calculate return, that is difference between next day and today
def calculateReturn(df):
    return_stock = [0]
    for i in range(0, df.index[-1]):
        return_stock_1 = ((df.loc[i + 1, 'Close'] - df.loc[i, 'Close'])) / df.loc[i, 'Close']
        return_stock.append(return_stock_1)
    return_stock = pd.Series(return_stock)
    return return_stock.to_frame()


for stock_idx in range(len(listofFiles)):
    stock_dir = listofFiles[stock_idx]
    # delete_row(stock_dir)
    df = pd.read_csv(stock_dir, usecols=[0,2])
    # df = pd.concat([df[col].str.split()
    #                        .str[0]
    #                        .str.replace(',','').astype(float) for col in df], axis=1)
    # df['Close'] = df['Close'].astype(float)
    df['Return'] = calculateReturn(df)
    df['Label'] = calculateLabel(df)
    final_data = df.dropna()
    final_data.to_csv(stock_dir, index=False)
