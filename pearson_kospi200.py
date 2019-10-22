# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import operator
from sklearn.metrics.pairwise import cosine_similarity

# 주가 데이터셋은 2013-01-02 ~ 2018-11-28 까지 존재.
# 따라서 해당 기간동안 kospi에 상장되어 있던 기업의 주가 데이터는 총 1451개이다.
# 몇 몇 기업들은 kospi 상장날짜가  2013-01-02 이후인 기업들도 존재.
# 그러므로 numpy를 활용하여 zeros행렬을 202x1451 비율로 선언해둔 뒤 기업들의 데이터셋을 차례대로 뒤에서부터 채운다.

# 유사도를 비교할 10개의 주요 기업 목록



FILE_DIR = 'D:\Lab\Stock\KOSPI200_dataset'

#디렉토리 내에 있는 모든 코스피200 주가 데이터셋 파일의 경로명+이름을 가져오기
def create_list_stock(dirName):
    listofFile = os.listdir(dirName)
    allFiles = list()
    for entry in listofFile:
        fullPath = os.path.join(dirName,entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + create_list_stock(fullPath) # 해당파일의 디렉토리및 파일명을 allFiles에 저장
        else:
            allFiles.append(fullPath)

    return allFiles
listofFiles = create_list_stock(FILE_DIR)
for i in range(len(listofFiles)):
    print((i), listofFiles[i])


### csv파일 가져오기 ###
# df = pd.read_csv(listofFiles[0], engine='python', usecols=[2])
# dfl = list(df['Return'])
# print(df['Return'])

allStockArray = np.zeros((202, 1451))
# 파일이름 리스트를 활용해 주가 등락률 데이터 가져오기
for stock_idx in range(len(listofFiles)):
    stock_dir = listofFiles[stock_idx]
    df = pd.read_csv(stock_dir, engine='python', usecols=[2])
    stocklist = list(df['Return'])
    stocklen = len(stocklist)
    if stocklen < 1451 :
        #print("formerprocess")
        startIndex = 1451 - stocklen
        for i in range(1450, startIndex-1, -1):
            allStockArray[stock_idx][i] = stocklist[i-startIndex]
    else:
        #print("direct")
        allStockArray[stock_idx] = stocklist

# 202개의 기업들의 주가 데이터를 담고있는 allStockArray를 활용하여 각각의 유사도 비교.
primeCorp = {107:'삼성전자', 49:'SK하이닉스', 117:'셀트리온', 18:'KB금융', 194:'현대차', 38:'POSCO', 31:'LG화학', 35:'NAVER', 186:'현대모비스', 48:'SK텔레콤'}

#print(allStockArray.shape)
for corp_idx in primeCorp:
    corplist = []
    for i in range(0, 202):
        list = []
        if corp_idx == i:  ## 자기자신과의 유사도 비교를 피하기 위함.
            list = 0, 0
            corplist.append(list)
        else:
            corp_tmp = allStockArray[corp_idx].reshape(1, 1451)
            i_tmp = allStockArray[i].reshape(1, 1451)
            sim = np.corrcoef(corp_tmp, i_tmp)
            list = i, sim[0][1]
            corplist.append(list)

    bestsimil = sorted(corplist, key=operator.itemgetter(1), reverse=True)
    print(bestsimil)
    # arrange the result of pearson correlation coefficient calculation