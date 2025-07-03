import yfinance as yf
import pandas as pd

# Define the stock symbol and date range
ticker = "AAPL"
start_date = "2019-01-01"
end_date = "2024-12-31"

# Download the data
data = yf.download(ticker, start=start_date, end=end_date)

# Keep only the 'Close' column
data = data[["Close"]]

# Save to CSV (optional)
data.to_csv("aapl_stock.csv")

# Print first few rows
print(data.head())
