import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("aapl_stock.csv", index_col="Date", parse_dates=True)

# Create a column for prediction (30 days into the future)
forecast_days = 30
df["Prediction"] = df[["Close"]].shift(-forecast_days)

# Create feature dataset (X) and target dataset (y)
X = np.array(df[["Close"]][:-forecast_days])
y = np.array(df["Prediction"][:-forecast_days])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model confidence (R^2 score)
confidence = model.score(X_test, y_test)
print(f"Model Accuracy: {confidence:.2f}")

# Predict the next 30 days
x_future = np.array(df[["Close"]][-forecast_days:])
forecast = model.predict(x_future)

# Create new DataFrame for plotting
df_forecast = df[["Close"]].copy()
df_forecast["Forecast"] = np.nan
last_date = df_forecast.index[-forecast_days]

for i, pred in enumerate(forecast):
    next_date = df_forecast.index[-forecast_days + i]
    df_forecast.loc[next_date, "Forecast"] = pred

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_forecast["Close"], label="Actual Close Price")
plt.plot(df_forecast["Forecast"], label="Predicted (30-day Forecast)", linestyle="--")
plt.title("Stock Price Prediction - AAPL")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
