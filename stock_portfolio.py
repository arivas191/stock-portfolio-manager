import os
import datetime as dt
from typing import List, Tuple

import pandas as pd
import yfinance as yf
import mysql.connector as mc


# Configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "phpmyadmin")
DB_PASS = os.getenv("DB_PASS", "root")
DB_NAME = os.getenv("DB_NAME", "stock")


def db_connection():
    return mc.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        charset="utf8mb4",
        autocommit=False,
    )


def ensure_schema(conn):
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4")
    cur.execute(f"USE `{DB_NAME}`")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
          Ticker  VARCHAR(16) NOT NULL,
          Date    DATE        NOT NULL,
          Close   DOUBLE,
          High    DOUBLE,
          Low     DOUBLE,
          Open    DOUBLE,
          Volume  BIGINT,
          PRIMARY KEY (Ticker, Date)
        ) ENGINE=InnoDB
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolios (
          id INT AUTO_INCREMENT PRIMARY KEY,
          name VARCHAR(128) NOT NULL,
          creation_date DATETIME NOT NULL,
          current_value DOUBLE
        ) ENGINE=InnoDB
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_stocks (
          portfolio_id INT NOT NULL,
          ticker VARCHAR(16) NOT NULL,
          shares DOUBLE,
          PRIMARY KEY (portfolio_id, ticker),
          FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
        ) ENGINE=InnoDB
        """
    )
    conn.commit()
    cur.close()


def validate_ticker(ticker: str) -> bool:
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period="1d")
        return not hist.empty
    except Exception:
        return False


def data_retrieval(tickers: List[str], start: str = None, end: str = None, period: str = None, interval: str = "1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Date", "Close", "High", "Low", "Open", "Volume"])

    yf_kwargs = {"tickers": tickers, "interval": interval}
    if start and end:
        yf_kwargs.update({"start": start, "end": end})
    elif period:
        yf_kwargs.update({"period": period})
    else:
        yf_kwargs.update({"period": "1y"})

    df = yf.download(**yf_kwargs, group_by="ticker", threads=True, progress=False)

    # Normalize columns for multi-ticker or single-ticker
    rows = []
    if isinstance(df.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in df.columns.levels[0]:
                continue
            sub = df.xs(ticker, axis=1, level=0, drop_level=True)
            sub = sub.reset_index()
            sub["Ticker"] = ticker
            rows.append(sub)
        if rows:
            long_df = pd.concat(rows, ignore_index=True)
        else:
            long_df = pd.DataFrame()
    else:
        df = df.reset_index()
        if len(tickers) == 1:
            df["Ticker"] = tickers[0]
        long_df = df

    # standardize column names
    col_map_candidates = {c.lower(): c for c in long_df.columns}
    def pick(colnames):
        for k in colnames:
            if k in col_map_candidates:
                return col_map_candidates[k]
        return None

    date_col = pick(["date", "Date"])
    ticker_col = pick(["ticker", "Ticker"])
    close_col = pick(["close", "Close", "adj close", "Adj Close"])
    high_col = pick(["high", "High"])
    low_col = pick(["low", "Low"])
    open_col = pick(["open", "Open"])
    vol_col = pick(["volume", "Volume"])

    rename_map = {}
    if date_col: rename_map[date_col] = "Date"
    if ticker_col: rename_map[ticker_col] = "Ticker"
    if close_col: rename_map[close_col] = "Close"
    if high_col: rename_map[high_col] = "High"
    if low_col: rename_map[low_col] = "Low"
    if open_col: rename_map[open_col] = "Open"
    if vol_col: rename_map[vol_col] = "Volume"
    long_df = long_df.rename(columns=rename_map)

    field = ['Ticker', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    for f in field:
        if f not in long_df.columns:
            long_df[f] = pd.NA

    long_df = long_df[field].dropna(subset=["Ticker", "Date"]).copy()
    long_df["Date"] = pd.to_datetime(long_df["Date"]).dt.date
    long_df["Ticker"] = long_df["Ticker"].astype(str).str.strip().str.upper()
    return long_df



def insert_prices(conn, df: pd.DataFrame):
    if df.empty:
        return 0

    df = df.where(pd.notnull(df), None)

    # Ensure correct column order
    df = df[['Ticker', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

    cur = conn.cursor()
    cur.execute(f"USE `{DB_NAME}`")

    insert = """
    INSERT INTO prices (Ticker, Date, Close, High, Low, Open, Volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      Close=VALUES(Close),
      High=VALUES(High),
      Low=VALUES(Low),
      Open=VALUES(Open),
      Volume=VALUES(Volume)
    """

    # Use .to_records to guarantee tuple order matches columns
    rows = df.values.tolist()
    cur.executemany(insert, rows)
    conn.commit()
    cur.close()
    return len(rows)



def fetch_prices_for_tickers(conn, tickers: List[str], start: str = None, end: str = None, period: str = None, interval: str = "1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=['Ticker', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume'])

    cur = conn.cursor(dictionary=True)
    cur.execute(f"USE `{DB_NAME}`")

    # Build query to fetch any existing rows
    conditions = []
    params = []
    if start:
        conditions.append("Date >= %s")
        params.append(start)
    if end:
        conditions.append("Date <= %s")
        params.append(end)
    where_clause = " AND ".join(conditions)

    existing_rows = []
    for t in tickers:
        q = f"SELECT Date, Close, High, Low, Open, Volume FROM prices WHERE Ticker=%s"
        if where_clause:
            q += " AND " + where_clause
        cur.execute(q, [t] + params)
        existing_rows.extend([(t, r['Date'], r['Close'], r['High'], r['Low'], r['Open'], r['Volume']) for r in cur.fetchall()])

    df_existing = pd.DataFrame(existing_rows, columns=['Ticker','Date','Close','High','Low','Open','Volume'])

    # Identify tickers missing any data
    missing_tickers = []
    for t in tickers:
        if df_existing.empty or t not in df_existing['Ticker'].unique():
            missing_tickers.append(t)

    # Fetch only missing tickers
    df_new = pd.DataFrame()
    if missing_tickers:
        df_new = data_retrieval(missing_tickers, start=start, end=end, period=period, interval=interval)
        if not df_new.empty:
            insert_prices(conn, df_new)

    if df_existing.empty:
        return df_new
    if df_new.empty:
        return df_existing
    return pd.concat([df_existing, df_new], ignore_index=True)



def create_portfolio(conn, name: str, tickers: List[str]) -> int:
    conn_cursor = conn.cursor()
    conn_cursor.execute(f"USE `{DB_NAME}`")
    now = dt.datetime.now(dt.UTC)
    conn_cursor.execute("INSERT INTO portfolios (name, creation_date) VALUES (%s, %s)", (name, now))
    portfolio_id = conn_cursor.lastrowid
    valid = []
    for tk in tickers:
        tk_u = tk.strip().upper()
        if validate_ticker(tk_u):
            valid.append((portfolio_id, tk_u))
    if valid:
        conn_cursor.executemany("INSERT IGNORE INTO portfolio_stocks (portfolio_id, ticker) VALUES (%s, %s)", valid)
        fetch_prices_for_tickers(conn, [tk[1] for tk in valid])
    conn.commit()
    conn_cursor.close()
    return portfolio_id


def add_stock_to_portfolio(conn, portfolio_id: int, ticker: str) -> bool:
    t = ticker.strip().upper()
    if not validate_ticker(t):
        return False
    cur = conn.cursor()
    cur.execute(f"USE `{DB_NAME}`")
    cur.execute("INSERT IGNORE INTO portfolio_stocks (portfolio_id, ticker) VALUES (%s, %s)", (portfolio_id, t))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    if affected > 0:
        fetch_prices_for_tickers(conn, [t])
    return affected > 0


def remove_stock_from_portfolio(conn, portfolio_id: int, ticker: str) -> bool:
    t = ticker.strip().upper()
    cur = conn.cursor()
    cur.execute(f"USE `{DB_NAME}`")
    cur.execute("DELETE FROM portfolio_stocks WHERE portfolio_id=%s AND ticker=%s", (portfolio_id, t))
    conn.commit()
    affected = cur.rowcount
    cur.close()
    return affected > 0


def list_portfolios(conn) -> List[Tuple]:
    cur = conn.cursor(dictionary=True)
    cur.execute(f"USE `{DB_NAME}`")
    cur.execute("""
      SELECT p.id, p.name, p.creation_date, GROUP_CONCAT(ps.ticker ORDER BY ps.ticker SEPARATOR ',') AS tickers
      FROM portfolios p
      LEFT JOIN portfolio_stocks ps ON p.id = ps.portfolio_id
      GROUP BY p.id
      ORDER BY p.creation_date DESC
    """)
    rows = cur.fetchall()
    cur.close()
    return rows


def get_portfolio_tickers(conn, portfolio_id: int) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"USE `{DB_NAME}`")
    cur.execute("SELECT ticker FROM portfolio_stocks WHERE portfolio_id=%s ORDER BY ticker", (portfolio_id,))
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    return rows


def get_portfolio_id_by_name(conn, name: str) -> int:
    cur = conn.cursor()
    cur.execute(f"USE `{DB_NAME}`")
    cur.execute("SELECT id FROM portfolios WHERE name=%s", (name,))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def cli_menu():
    conn = db_connection()
    ensure_schema(conn)

    while True:
        print("\nSelect an option:")
        print("1. Create portfolio")
        print("2. Add stock to portfolio")
        print("3. Remove stock from portfolio")
        print("4. List portfolios")
        print("5. Fetch prices for portfolio")
        print("6. Exit")

        choice = input("Enter option number: ").strip()

        if choice == "1":
            name = input("Enter portfolio name: ").strip()
            tickers = input("Enter tickers (comma separated): ").strip().split(",")
            tickers = [t.strip().upper() for t in tickers if t.strip()]
            pid = get_portfolio_id_by_name(conn, name)
            if pid:
                print(f"Portfolio '{name}' already exists.")
            else:
                pid = create_portfolio(conn, name, tickers)
                print(f"Created portfolio '{name}' with id {pid}")

        elif choice == "2":
            name = input("Enter portfolio name: ").strip()
            pid = get_portfolio_id_by_name(conn, name)
            if not pid:
                print(f"Portfolio '{name}' not found.")
            else:
                ticker = input("Enter ticker to add: ").strip().upper()
                ok = add_stock_to_portfolio(conn, pid, ticker)
                print("Added" if ok else "Failed to add (invalid or already in portfolio).")

        elif choice == "3":
            name = input("Enter portfolio name: ").strip()
            pid = get_portfolio_id_by_name(conn, name)
            if not pid:
                print(f"Portfolio '{name}' not found.")
            else:
                ticker = input("Enter ticker to remove: ").strip().upper()
                ok = remove_stock_from_portfolio(conn, pid, ticker)
                print("Removed" if ok else "Failed to remove (not in portfolio).")

        elif choice == "4":
            rows = list_portfolios(conn)
            if not rows:
                print("No portfolios found.")
            for r in rows:
                tickers = r["tickers"] or ""
                print(f'{r["id"]} | {r["name"]} | {r["creation_date"]} | {tickers}')

        elif choice == "5":
            name = input("Enter portfolio name: ").strip()
            pid = get_portfolio_id_by_name(conn, name)
            if not pid:
                print(f"Portfolio '{name}' not found.")
            else:
                start = input("Enter start date (YYYY-MM-DD, blank for none): ").strip() or None
                end = input("Enter end date (YYYY-MM-DD, blank for none): ").strip() or None
                period = input("Enter period (e.g., 1y, 6mo, blank if dates given): ").strip() or None
                interval = input("Enter interval (default 1d): ").strip() or "1d"

                tickers = get_portfolio_tickers(conn, pid)
                if not tickers:
                    print(f"No tickers in portfolio '{name}'.")
                else:
                    df = fetch_prices_for_tickers(conn, tickers, start=start, end=end, period=period, interval=interval)
                    if df.empty:
                        print("No price data available.")
                    else:
                        print(df.to_string(index=False))

        elif choice == "6":
            print("Exiting.")
            break

        else:
            print("Invalid option, try again.")

    conn.close()


if __name__ == "__main__":
    cli_menu()
