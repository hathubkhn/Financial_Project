# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import operator
from sklearn.metrics.pairwise import cosine_similarity

# 주가 데이터셋은 2013-01-02 ~ 2018-11-28 까지 존재.

FILE_DIR = 'D:\Lab\Stock\SP500\SP500_cosine_label'


# 디렉토리 내에 있는 모든 코스피200 주가 데이터셋 파일의 경로명+이름을 가져오기
def create_list_stock(dirName):
    listofFile = os.listdir(dirName)
    allFiles = list()
    for entry in listofFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + create_list_stock(fullPath)  # 해당파일의 디렉토리및 파일명을 allFiles에 저장
        else:
            allFiles.append(fullPath)

    return allFiles


listofFiles = create_list_stock(FILE_DIR)
for i in range(len(listofFiles)):
    print("file_path_%d = '%s'" %(i, listofFiles[i]))

allStockArray = np.zeros((460, 1467))
# 파일이름 리스트를 활용해 주가 등락률 데이터 가져오기
for stock_idx in range(len(listofFiles)):
    stock_dir = listofFiles[stock_idx]
    df = pd.read_csv(stock_dir, engine='python', usecols=[2])
    stocklist = list(df['Return'])
    stocklen = len(stocklist)
    if stocklen < 1467:
        # print("formerprocess")
        startIndex = 1467 - stocklen
        for i in range(1466, startIndex - 1, -1):
            allStockArray[stock_idx][i] = stocklist[i - startIndex]
    else:
        # print("direct")
        allStockArray[stock_idx] = stocklist

# primeCorp = {69:'Boeing Company',237:'Johnson & Johnson',47: 'Apple Inc.',27:'Amazon.com Inc',239:'JPMorgan Chase & Co'}
primeCorp = {47: 'Apple Inc.'}
# print(allStockArray.shape)
for corp_idx in primeCorp:
    corplist = []
    for i in range(0, 460):
        list = []
        if corp_idx == i:
            list = 0, 0
            corplist.append(list)
        else:
            corp_tmp = allStockArray[corp_idx].reshape(1, 1467)
            i_tmp = allStockArray[i].reshape(1, 1467)
            sim = cosine_similarity(corp_tmp, i_tmp)
            list = i, sim
            corplist.append(list)

    bestsimil = sorted(corplist, key=operator.itemgetter(1), reverse=True)
    print(bestsimil)
