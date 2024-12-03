from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf

def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Returns'] = df['Close'].pct_change()
    
    return df

def dataprocessing(tech_list):
    end = datetime.now()
    start = end - timedelta(days=365*15)

    stock_data = {}
    for stock in tech_list:
        df = yf.download(stock, start, end)
        df = df.sort_index()
        df = add_technical_indicators(df)
        stock_data[stock] = df.dropna()
    
    scaler = StandardScaler()
    data = {}
    x_train = {}
    y_train = {}
    x_test = {}
    y_test = {}
    shape = None

    for stock in tech_list:
        if stock_data[stock].empty:
            print(f"Error: No data downloaded for {stock}")
            exit()

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'Returns']
        data[stock] = stock_data[stock][features].values
        
        scaled_data = scaler.fit_transform(data[stock])
        
        sequence_length = 30
        x_temp, y_temp = [], []

        for i in range(sequence_length, len(scaled_data)):
            x_temp.append(scaled_data[i-sequence_length:i])
            y_temp.append(scaled_data[i, 3])
        
        x_temp = np.array(x_temp)
        y_temp = np.array(y_temp)
        
        x_train[stock], x_test[stock], y_train[stock], y_test[stock] = train_test_split(
            x_temp, y_temp, test_size=0.2, random_state=42, shuffle=False
        )
        
        if shape is not None and x_train[stock].shape[1:] != shape:
            raise ValueError(f"Shape mismatch for {stock}: expected {shape}, got {x_train[stock].shape[1:]}")
        shape = x_train[stock].shape[1:]
        
    return x_train, y_train, x_test, y_test, shape, scaler