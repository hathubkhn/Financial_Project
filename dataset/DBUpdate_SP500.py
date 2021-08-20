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

class DBUpdater:
    def __init__(self, start_date, end_date = None):
       #============================= Create Table =============================
        self.postgresConnection = psycopg2.connect(user = "postgres", password = "Vc07062011@", database = 'stock_price', host = '127.0.0.1')
        
        with self.postgresConnection.cursor() as curs:
             name_table = 'SP500_company_info'
             sql = """
             create table if not exists sp500_company_info (
                 code varchar(20),
                 company varchar(40),
                 last_update date,
                 PRIMARY KEY (code))
             """
             curs.execute(sql)
             
             sql = """
             create table if not exists sp500_daily_price (
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

    
    def read_SP500_code(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        myPage = urlopen(url) # opens this url

        mySoup = BeautifulSoup(myPage, "html.parser") # parse html soup 
        
        table = mySoup.find('table', {'class': 'wikitable sortable'}) # finds wiki sortable table in webpage html
        
        for row in table.findAll('tr'): # find every row in the table
            col = row.findAll('td') # find every column in that row
            if len(col) > 0: # if there are columns in that row
                print(col)
                sector = str(col[1].text)
                ticker = str(col[0].text.strip()) # identify the ticker in the row
                
                self.codes[sector] = ticker
        df = pd.DataFrame()
        df['code'] = list(self.codes.values())
        df['company'] = list(self.codes.keys())
        print(df)
        
        return df

    def update_comp_info(self):
        sql = "SELECT * FROM sp500_company_info"
        df = pd.read_sql(sql, self.postgresConnection)
        for idx in range(len(df)):
            self.codes[df['code'].values[idx]] = df['company'].values[idx]
        with self.postgresConnection.cursor() as curs:
            sql = "SELECT max(last_update) FROM sp500_company_info"
            curs.execute(sql)
            rs = curs.fetchone()

            today = datetime.today().strftime('%Y-%m-%d')
            if rs[0] == None or rs[0].strftime('%Y-%m-%d') < today:
                sp500 = self.read_SP500_code()

                for idx in range(len(sp500)):
                    code = sp500.code.values[idx]
                    company = sp500.company.values[idx]
                    if company.find("'") != -1:
                        company = company.replace("'", "")

                    sql = f"INSERT INTO sp500_company_info (code, company, last_update) VALUES ('{code}', '{company}', '{today}') ON CONFLICT (code) DO UPDATE SET company = '{company}', last_update = '{today}';"
                    curs.execute(sql)
                    tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                    print(f"[{tmnow}] #{idx+1:04d} INSERT INTO sp500_company_info VALUES ({code}, '{company}', '{today}')")
                self.postgresConnection.commit()
                print('')

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
                sql = f"INSERT INTO sp500_daily_price (code, date, open, high, low, close, volume) VALUES ('{code}', "\
                    f"'{r.Date}', {r.Open}, {r.High}, {r.Low}, {r.Close}, "\
                    f"{r.Volume}) ON CONFLICT (code, date) DO UPDATE SET open = {r.Open}, High = {r.High}, low = {r.Low}, close = {r.Close}, volume = {r.Volume};"

                curs.execute(sql)
            self.postgresConnection.commit()
            print('[{}] {} : {} rows > INSERT INTO sp500_daily_price'\
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




