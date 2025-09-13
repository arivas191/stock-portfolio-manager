import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import mysql.connector as mc

"""
Arguements
- tickers
- start
- end

Field
| Ticker | Date | Close | High | Low | Open | Volume |
"""

def data_retrieval(tickers, start=None, end=None, interval='1d'):
    # retrieve data
    df = yf.download(tickers=tickers, start=start, end=end, interval=interval)

    # get price
    price = df['Price'] if 'Price' in df.columns.get_level_values(0) else df

    # transform index into (Date, Ticker)
    long_df = (
        price
        .stack(level=1)
        .rename_axis(index=['Date', 'Ticker'])
        .reset_index()
    )

    # define data field
    field = ['Ticker', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    long_df = long_df[field]
    
    return long_df

def db_connection():
    conn = mc.connect(
        host="127.0.0.1",
        port=3306,
        user="cindy", # replace with your own username
        password="NewP@ssw0rd!", # replace with your own password
        charset="utf8mb4"
    )
    
    return conn


def table_creation(conn, df):
    cur = conn.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS stock CHARACTER SET utf8mb4")
    cur.execute("USE stock")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prices (
      Ticker  VARCHAR(16) NOT NULL,
      Date    DATE        NOT NULL,
      Close   DOUBLE,
      High    DOUBLE,
      Low     DOUBLE,
      Open    DOUBLE,
      Volume  BIGINT,
      PRIMARY KEY (Ticker, Date)
    )
    """)
    
    insert = """
    INSERT INTO prices (Ticker, Date, Close, High, Low, Open, Volume) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    rows = list(df.itertuples(index=False, name=None))
    
    cur.executemany(insert, rows)
    conn.commit()
    

if __name__=='__main__':
    parser = argparse.ArgumentParser("Fetch stock prices for a portfolio")
    parser.add_argument("--tickers", required=True, help="Comma separated list of tickers, e.g. MSFT,AAPL,GOOG")
    parser.add_argument("--start", required=False, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=False, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1d", help="Interval, e.g. 1d, 1wk, 1mo")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    df = data_retrieval(tickers, start=args.start, end=args.end, interval=args.interval)
    conn = db_connection()
    table_creation(conn, df)
