# ----- Example Python program to create a database in PostgreSQL using Psycopg2 -----

# import the PostgreSQL client for Python
import requests
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
from datetime import datetime
from threading import Timer
import urllib, pymysql, calendar, time, json
import pandas_datareader.data as web
import yfinance as yf
import os

class DBUpdater:
    def __init__(self, start_date, end_date = None):
       #============================= Create Table =============================
        self.postgresConnection = psycopg2.connect(user = "postgres", password = "Vc07062011@", database = 'stock_price', host = '127.0.0.1')
        
        with self.postgresConnection.cursor() as curs:
             name_table = 'nasdaq_company_info'
             sql = """
             create table if not exists nasdaq_company_info (
                 code varchar(100),
                 company varchar(1000),
                 sector varchar(100),
                 industry varchar(100),
                 PRIMARY KEY (code))
             """
             curs.execute(sql)
             
             sql = """
             create table if not exists nasdaq_daily_price (
                 code varchar(20),
                 date DATE, 
                 open bigint, 
                 high bigint,
                 low bigint, 
                 close bigint,
                 volume bigint, 
                 PRIMARY KEY (Code, Date))
             """ 
             curs.execute(sql)
        self.postgresConnection.commit()
        self.codes = dict()
        self.start_date = start_date
        if end_date == None:
            self.end_date = datetime.today().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date

    
    def read_NASDAQ_code(self):
        csv_file = os.path.join(os.path.abspath('./'), 'NASDAQ_list.csv')
        df = pd.read_csv(csv_file, sep = ',')
        df['MarketCap'] = df['Market Cap'].fillna('')
        df = df.sort_values(['Market Cap'], ascending=False)
        df = df.reset_index(drop=True)
        df = df.rename(columns={'industry':'Industry'})
        print(df.head(10))
        return df[['Symbol', 'Name', 'Sector', 'Industry']]

    def update_comp_info(self):
        df = self.read_NASDAQ_code()
        sql = "SELECT * FROM nasdaq_company_info"
        with self.postgresConnection.cursor() as curs:
            for r in df.itertuples():
                name = r.Name
                self.codes[name] = r.Symbol
                if name.find("'") != -1:
                    name = name.replace("'", "")
                sql = f"INSERT INTO nasdaq_company_info (code, company, sector, industry) VALUES ('{r.Symbol}', '{name}', '{r.Sector}', '{r.Industry}');"

                curs.execute(sql)
            self.postgresConnection.commit()

    # ============================== INSERT DATA INTO DAILY PRICE TABLE ===============

    def read_yahoo(self, code):
        try:
            df = yf.download(code, self.start_date, self.end_date)
            df['Date'] = list(df.index)
            df = df.reset_index(drop = True)
        except Exception as e:
            print('Exception occured :', str(e))
            return none
        return df
    
    def replace_into_db(self, df, code):
        with self.postgresConnection.cursor() as curs:
            for r in df.itertuples():
                sql = f"INSERT INTO nasdaq_daily_price (code, date, open, high, low, close, volume) VALUES ('{code}', "\
                    f"'{r.Date}', {r.Open}, {r.High}, {r.Low}, {r.Close}, "\
                    f"{r.Volume}) ON CONFLICT (code, date) DO UPDATE SET open = {r.Open}, High = {r.High}, low = {r.Low}, close = {r.Close}, volume = {r.Volume};"

                curs.execute(sql)
            self.postgresConnection.commit()
            print('[{}] {} : {} rows > INSERT INTO nasdaq_daily_price'\
                'price [OK]'.format(datetime.now().strftime('%Y-%m-%d'), code, len(df)))

    def update_daily_price(self):
        for idx, code in enumerate(self.codes):
            df = self.read_yahoo(code)
            if df is None:
                continue
            self.replace_into_db(df, code)
   
if __name__ == '__main__':
    dbu = DBUpdater('2020-01-01')
    dbu.update_comp_info()
    dbu.update_daily_price()




