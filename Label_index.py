import pandas as pd
import os

pd.options.mode.chained_assignment = None

FILE_DIR = 'D:\Lab\Stock\KOSPI200\Study1\Index\Val.csv'


# FILE_DIR = 'D:\Gold.csv'

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


def calculateReturn(df):
    return_stock = [0]
    for i in range(0, df.index[-1]):
        # return_stock_1 = ((df.loc[i + 1, 'Receive remittance'] - df.loc[i, 'Receive remittance']))/df.loc[i,'Receive remittance']
        return_stock_1 = ((df.loc[i + 1, 'Close'] - df.loc[i, 'Close'])) / df.loc[i, 'Close']
        return_stock.append(return_stock_1)
    return_stock = pd.Series(return_stock)
    return return_stock.to_frame()


df = pd.read_csv(FILE_DIR,usecols=[2])
# df = pd.concat([df[col].str.split()
#                        .str[0]
#                        .str.replace(',','').astype(float) for col in df], axis=1)
df['Close'] = df['Close'].astype(float)
df['Return'] = calculateReturn(df)
df['Label'] = calculateLabel(df)
final_data = df.dropna()
final_data.to_csv(FILE_DIR, index=False)
