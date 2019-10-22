import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ss = pd.read_csv("D:\\Lab\\Stock\\KOSPI200_dataset_train_label\\셀트리온.csv", usecols=[2])
ss = pd.read_csv("D:\\Lab\\Stock\\KOSPI200_dataset_train_label\\삼성전자.csv", usecols=[2])
ss = np.array(ss)

KOSPI_index = pd.read_csv("D:\Lab\Stock\KOSPI200_index\KOSPI_train.csv", usecols=[2])
KOSPI_index= np.array(KOSPI_index)

cosine_value = cosine_similarity(ss, KOSPI_index)
print(ss)
print(cosine_value)
