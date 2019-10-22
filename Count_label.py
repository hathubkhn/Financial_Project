import pandas as pd
import matplotlib.pyplot as plt

list_stock = ['삼성전자', 'SK하이닉스', '셀트리온', '현대차', 'POSCO', 'LG화학', 'NAVER', 'SK텔레콤', 'KB금융', '현대모비스']


def countLabel(df):
    num_fall = 0
    num_raise = 0
    num_hold = 0
    for i in range(0, df.index[-1]):
        if df.loc[i, 'Label'] == -1:
            num_fall += 1
        if df.loc[i, 'Label'] == 0:
            num_hold += 1
        if df.loc[i, 'Label'] == 1:
            num_raise += 1
    return num_raise, num_hold, num_fall


#
# def draw():
#     name = ["Raise","Fall", "Hold"]
#     colors = ["#3498db"]
#     num_arr = [num_raise,num_fall,num_hold]
#
#     xs = [i + 0.05 for i, _ in enumerate(name)]
#     plt.bar(xs, num_arr,color= colors)
#     plt.xticks([i + 0.1 for i, _ in enumerate(name)], name)
#     plt.show()

for stock_idx in range(len(list_stock)):
    stock = list_stock[stock_idx]
    df = pd.read_csv('D:\\Lab\\Stock\\KOSPI200_dataset_classification_LSTM\\{}.csv'.format(stock))
    total_raise, total_hold, total_fall = countLabel(df)
    total_series_length = len(df.index)
    train_len = int(total_series_length * 0.8)
    Train = df[:train_len]

    num_raise_train, num_hold_train, num_fall_train = countLabel(Train)
    num_raise_test = total_raise - num_raise_train
    num_fall_test = total_fall - num_fall_train
    num_hold_test = total_hold - num_hold_train
    print(list_stock[stock_idx])
    print("With train dataset has Raise : %d--" % num_raise_train, "Fall: %d--" % num_fall_train,
          "Hold: %d" % num_hold_train)
    print("With test dataset has Raise : %d--" % num_raise_test, "Fall: %d--" % num_fall_test,
          "Hold: %d" % num_hold_test)
    # draw()
