# Stock Portfolio Manager

A Python application for managing stock portfolios and retrieving historical price data. The system allows users to create portfolios, add/remove stocks, and fetch price data using the Yahoo Finance API. All data is stored in a MySQL database for efficient retrieval.

Using the database as a foundation, a hybrid stock-price forecasting model combining ARIMA and LSTM was implemented. On top of this, a mock trading environment simulates day-to-day execution and evaluates strategies with the objective of maximizing user returns.

## Features
- Create and manage multiple portfolios
- Add/remove stocks from portfolios  
- Fetch historical stock price data
- Store price data locally to avoid redundant API calls
- Command-line interface for easy interaction
- Pre-process stock price data
- Stock price prediction
- Mock trading environment


## Usage
Create and activate a virtual environment, install dependencies, then run:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python stock_portfolio.py
```

This launches a CLI interface that prompts for available options including portfolio creation, stock management, and price data retrieval.
