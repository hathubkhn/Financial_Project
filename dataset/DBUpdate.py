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

class DBUpdater:
    def __init__(self):
       #============================= Create Table =============================
        self.postgresConnection = psycopg2.connect(user = "postgres", password = "Vc07062011@", database = 'stock_price', host = '127.0.0.1')
        
        with self.postgresConnection.cursor() as curs:
             name_table = 'company_info'
             sql = """
             create table if not exists company_info (
                 code varchar(20),
                 company varchar(40),
                 last_update date,
                 PRIMARY KEY (code))
             """
             curs.execute(sql)
             
             sql = """
             create table if not exists daily_price (
                 code varchar(20),
                 date DATE, 
                 open bigint, 
                 high bigint,
                 low bigint, 
                 close bigint,
                 volumn bigint, 
                 PRIMARY KEY (Code, Date))
             """ 
             curs.execute(sql)
        self.postgresConnection.commit()
        self.codes = dict()

    
    def read_krx_code(self):
        # KOSPI list
        url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        krx = pd.read_html(url, header = 0)[0]
        krx = krx[['종목코드', '회사명']]
        kosdex = pd.DataFrame([[122630, 'KODEX 레버리지'], [114800, 'KODEX 인버스']], columns=list(krx.columns))
        krx = krx.append(kosdex, ignore_index = True)
        krx = krx.rename(columns = {'종목코드': 'code', '회사명': 'company'})
        krx.code = krx.code.map('{:06d}'.format)
        return krx

    def update_comp_info(self):
        sql = "SELECT * FROM company_info"
        df = pd.read_sql(sql, self.postgresConnection)
        for idx in range(len(df)):
            self.codes[df['code'].values[idx]] = df['company'].values[idx]
        with self.postgresConnection.cursor() as curs:
            sql = "SELECT max(last_update) FROM company_info"
            curs.execute(sql)
            rs = curs.fetchone()

            today = datetime.today().strftime('%Y-%m-%d')
            if rs[0] == None or rs[0].strftime('%Y-%m-%d') < today:
                krx = self.read_krx_code()
                for idx in range(len(krx)):
                    code = krx.code.values[idx]
                    company = krx.company.values[idx]
                    sql = f"INSERT INTO company_info (code, company, last_update) VALUES ({code}, '{company}', '{today}') ON CONFLICT (code) DO UPDATE SET company = '{company}', last_update = '{today}';"
                    curs.execute(sql)
                    self.codes[code] = company
                    tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                    print(f"[{tmnow}] #{idx+1:04d} INSERT INTO company_info VALUES ({code}, '{company}', '{today}')")
                self.postgresConnection.commit()
                print('')

    def read_naver(self, code, company, pages_to_fetch):
        try:
            url = f"http://finance.naver.com/item/sise_day.nhn?code={code}"
            with urlopen(url) as doc:
                if doc is None:
                    return None
                html = BeautifulSoup(requests.get(url,
                headers={'User-agent': 'Mozilla/5.0'}).text, "lxml")
                pgrr = html.find("td", class_ = "pgRR")
                if pgrr is None:
                    return None

            s = str(pgrr.a["href"]).split("=")
            lastpage = s[-1]

            df = pd.DataFrame()
            pages = int(int(lastpage), pages_to_fetch)
            for page in range(1, pages + 1):
                pg_url = '{}&page={}'.format(url, page)
                df = df.append(pd.read_html(requests.get(pg_url,
                    headers={'User-agent': 'Mozilla/5.0'}).text)[0])
                tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                print('[{}] {} ({}) : {:04d}/{:04d} pages are downloading...'.
                    format(tmnow, company, code, page, pages), end="\r")

            df = df.rename(columns={'날짜':'date','종가':'close','시가':'open','고가':'high','저가':'low','거래량':'volume'})
            df['date'] = df['date'].replace('.', '-')
            df = df.dropna()
            df[['close', 'open', 'high', 'low', 'volume']] = df[['close', 'open', 'high', 'low', 'volume']].astype(int)
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print('Exception occured :', str(e))
            return none
        return df
    
    def replace_into_db(self, df, num, code, company):
        with self.postgresConnection.cursor() as curs:
            for r in df.itertuples():
                sql = f"INSERT INTO daily_price (code, date, open, high, low, close, volume) VALUES ('{code}', "\
                    f"'{r.date}', {r.open}, {r.high}, {r.low}, {r.close}, "\
                    f"{r.volume}) ON CONFLICT (code, date) DO UPDATE SET open = {r.open}, high = {r.high}, low = {r.low}, close = {r.close}, volume = {r.volume};"

                curs.execute(sql)
            self.postgresConnection.commit()
            print('[{}] #{:04d} {} ({}) : {} rows > INSERT INTO daily_'\
                'price [OK]'.format(datetime.now().strftime('%Y-%m-%d'\
                ' %H:%M'), num+1, company, code, len(df)))

    def update_daily_price(self, pages_to_fetch):
        """KRX 상장법인의 주식 시세를 네이버로부터 읽어서 DB에 업데이트"""
        for idx, code in enumerate(self.codes):
            df = self.read_naver(code, self.codes[code], pages_to_fetch)
            if df is None:
                continue
            self.replace_into_db(df, idx, code, self.codes[code])
   
    def execute_daily(self):
         self.update_comp_info()
         try:
             with open('config.json', 'r') as in_file:
                 config = json.load(in_file)
                 pages_to_fetch = config['pages_to_fetch']
         except FileNotFoundError:
             with open('config.json', 'w') as out_file:
                 pages_to_fetch = 100
                 config = {'pages_to_fetch': 1}
                 json.dump(config, out_file)
         self.update_daily_price(pages_to_fetch)
    
         tmnow = datetime.now()
         lastday = calendar.monthrange(tmnow.year, tmnow.month)[1]
         if tmnow.month == 12 and tmnow.day == lastday:
             tmnext = tmnow.replace(year=tmnow.year+1, month=1, day=1,
                 hour=17, minute=0, second=0)
         elif tmnow.day == lastday:
             tmnext = tmnow.replace(month=tmnow.month+1, day=1, hour=17,
                 minute=0, second=0)
         else:
             tmnext = tmnow.replace(day=tmnow.day+1, hour=17, minute=0,
                 second=0)
         tmdiff = tmnext - tmnow
         secs = tmdiff.seconds
         t = Timer(secs, self.execute_daily)
         print("Waiting for next update ({}) ... ".format(tmnext.strftime
             ('%Y-%m-%d %H:%M')))
         t.start()
    

if __name__ == '__main__':
    dbu = DBUpdater()
    dbu.execute_daily()




