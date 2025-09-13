import pandas as pd
import numpy as np
import yfinance as yf
import mysql.connector as mc

# connect to the database
def db_connection():
    conn = mc.connect(
        host="localhost",
        port=3306,
        user="phpmyadmin", # replace with your own username
        password="password", # replace with your own password
        charset="utf8mb4",
        database="stock"
    )
    
    return conn

conn = db_connection()

# get table name
cur = conn.cursor()
cur.execute("SHOW TABLES;")
tables = [row[0] for row in cur.fetchall()]
print(tables)

# load sql table into pandas dataframe
df = pd.read_sql("SELECT * FROM `prices`", conn)

conn.close()

print(df.head())

print(df.shape)

# check that there is no missing values in the table
print(df.isna().sum())

# convert timestamps column to the datetime type
df["Date"] = pd.to_datetime(df["Date"])

# daily returns: measures precentage change from previous day's close
df["daily_return"] = df["Close"].pct_change()

# log returns: more stable measures of precentage change from previous day's close
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# daily range: measure the daily volatility
df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

# volatility: measure the fluctuation (risk) in a 5-day period and 20-day period
df["volatility_5d"] = df["daily_return"].rolling(window=5).std()
df["volatility_20d"] = df["daily_return"].rolling(window=20).std()

# moving average: smooths the price trend in a 5-day period and 20-day period 
# (helpful to see the underlying trend in a period by filtering out day-to-day noise)
df["ma_5"] = df["Close"].rolling(window=5).mean()
df["ma_20"] = df["Close"].rolling(window=20).mean()

# price move: measure percentage change in volumn of stock
df["volume_change"] = df["Volume"].pct_change()

# moving average of price move in a 5-day period
df["volume_ma_5"] = df["Volume"].rolling(window=5).mean()

# open-close gap: measure overnight price jump
df["oc_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

print(df.head())
