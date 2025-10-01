import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------
# 1. Load Stock Data        
# ---------------------------
stock_data = yf.download("AAPL", start="2023-01-01", end="2025-08-20")

# Use only Close prices
stock_data = stock_data[['Close']]

# ---------------------------
# 2. Create Features (Lag values)
# ---------------------------
# Predict today's Close using last 5 days
for i in range(1, 6):
    stock_data[f"lag_{i}"] = stock_data['Close'].shift(i)

stock_data.dropna(inplace=True)

# Features and Target
X = stock_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]
y = stock_data['Close']

# ---------------------------
# 3. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # keep time order
)

# ---------------------------
# 4. Train Model
# ---------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ---------------------------
# 5. Evaluate
# ---------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# ---------------------------
# 6. Plot Actual vs Predicted
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label="Actual Price", color="blue", linewidth=2)
plt.plot(y_test.index, y_pred, label="Predicted Price", color="red", linewidth=2, alpha=0.7)

plt.title("AAPL Stock Price Prediction (Actual vs Predicted)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()