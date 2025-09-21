import pandas as pd
import mysql.connector as mc
import datetime as dt
import sys
import numpy as np

# DB connection config
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "phpmyadmin"
DB_PASS = "root"
DB_NAME = "stock"

# Excel file with predictions
CSV_FILE = "prices_prediction.csv"

def db_connection():
    """Return a MySQL DB connection."""
    return mc.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, charset="utf8mb4", autocommit=True
    )

def run_simulation(portfolio_name: str, initial_investment: float):
    """Run a mock trading simulation for the given portfolio and initial investment."""
    conn = db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(f"USE `{DB_NAME}`")

    # Fetch portfolio id
    cur.execute("SELECT id FROM portfolios WHERE name=%s", (portfolio_name,))
    row = cur.fetchone()
    if not row: 
        raise ValueError("Portfolio not found")
    pid = row['id']

    # Fetch portfolio stocks (tickers) and initialize shares to 0
    cur.execute("SELECT ticker, 0 AS shares FROM portfolio_stocks WHERE portfolio_id=%s", (pid,))
    portfolio = {r['ticker']: r['shares'] for r in cur.fetchall()}

    # Initialize cash and update portfolio's current value in DB
    cash = initial_investment
    cur.execute("UPDATE portfolios SET current_value=%s WHERE id=%s", (initial_investment, pid))
    cur.execute("UPDATE portfolio_stocks SET shares = 0 WHERE portfolio_id = %s", (pid,))

    # Read predictions from csv
    df = pd.read_csv(CSV_FILE)
    # Ignore rows with missing prediction/signal
    df = df.dropna(subset=['PREDICTION', 'SIGNAL'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Filter to test period
    test_start = dt.date(2025, 3, 21)
    test_end = dt.date(2025, 9, 18)
    df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
    df = df.sort_values(['Date', 'Ticker'])


    # Shift SIGNAL forward so today's trade is based on next day's signal
    df['Next_Signal'] = df.groupby('Ticker')['SIGNAL'].shift(-1)

    # Simulation loop
    daily_values = []
    for date, group in df.groupby('Date'):
        buys = []
        sells = []

        for _, row_data in group.iterrows():
            ticker = row_data['Ticker']
            if ticker not in portfolio:
                continue
            signal = row_data['Next_Signal']  # use next day's signal
            if pd.isna(signal):
                continue
            signal = signal.lower()
            execution_price = row_data['Close']  # execute at today's close

            if signal == 'increase':
                buys.append((ticker, execution_price))
            elif signal == 'decrease' and portfolio[ticker] > 0:
                sells.append((ticker, execution_price))

        # Execute buys
        if buys and cash > 0:
            fraction_per_stock = cash / len(buys)
            for ticker, price in buys:
                shares_to_buy = int(fraction_per_stock / price)
                if shares_to_buy > 0:
                    portfolio[ticker] += shares_to_buy
                    cash -= shares_to_buy * price

        # Execute sells
        for ticker, price in sells:
            cash += portfolio[ticker] * price
            portfolio[ticker] = 0

        # Portfolio valuation at close
        total_value = cash + sum(portfolio[t] * group[group['Ticker']==t]['Close'].iloc[0] for t in portfolio if portfolio[t] > 0)
        daily_values.append(total_value)
        for ticker, shares in portfolio.items():
            cur.execute("UPDATE portfolio_stocks SET shares=%s WHERE portfolio_id=%s AND ticker=%s", (shares, pid, ticker))
        cur.execute("UPDATE portfolios SET current_value=%s WHERE id=%s", (total_value, pid))


    conn.close()
    # print(f"Simulation complete. Final portfolio value: {total_value:.2f}, Cash: {cash:.2f}")
    final_value = total_value
    # Use number of simulated trading days, not calendar days
    trading_days = len(daily_values)  # actual trading days from your data
    annualized_return = (final_value / initial_investment - 1) * (252 / trading_days)

    returns = np.diff(daily_values) / daily_values[:-1]  # daily returns
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) != 0 else 0

    print(f"Final portfolio value: {final_value:.2f}, Cash: {cash:.2f}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    run_simulation(sys.argv[1], float(sys.argv[2]))