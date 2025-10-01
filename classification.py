import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Retrieve stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2022-02-26')

# Calculate features
stock_data['Moving_Avg'] = stock_data['Close'].rolling(window=5).mean()
stock_data['Target'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)

# Drop NaN values
stock_data = stock_data.dropna()

# Split data into features and target
X = stock_data[['Open', 'High', 'Low', 'Moving_Avg']]
y = stock_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification model
classification_model = LogisticRegression()
classification_model.fit(X_train, y_train)
y_pred = classification_model.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))