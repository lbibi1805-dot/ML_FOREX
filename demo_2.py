import os
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Hàm tải dữ liệu
def crawl_forex_data(forex_pair, start_date, end_date):
    print(f"Downloading data for {forex_pair} from {start_date} to {end_date}...")
    data = yf.download(forex_pair, start=start_date, end=end_date, interval='1d')
    data.reset_index(inplace=True)
    print(data.head())  # In ra vài dòng đầu tiên của DataFrame
    print(data.columns)  # In ra các cột trong DataFrame
    return data

# Hàm tính toán các chỉ số kỹ thuật
def add_technical_indicators(data):
    if 'Close' not in data.columns:
        raise ValueError("The 'Close' column is missing from the DataFrame.")
    
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / data['Close']
    return data

# Hàm xử lý dữ liệu
def preprocess_data(data):
    data['Target'] = data['Close'].shift(-1)  # Dự đoán giá đóng cửa ngày hôm sau
    data.dropna(inplace=True)
    return data

# Hàm xây dựng mô hình
def build_model(X_train, y_train):
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly_features', PolynomialFeatures(degree=3)),
        ('scaler', StandardScaler()),
        ('linear_reg', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

# Hàm dự đoán
def predict(model, X):
    return model.predict(X)

def main():
    forex_pair = 'GBPJPY=X'
    start_date = '2020-01-01'  # Tải dữ liệu từ năm 2020
    end_date = '2023-10-11'  # Ngày kết thúc

    # Bước 1: Tải dữ liệu
    data = crawl_forex_data(forex_pair, start_date, end_date)

    # Bước 2: Thêm các chỉ số kỹ thuật
    data = add_technical_indicators(data)

    # Bước 3: Xử lý dữ liệu
    data = preprocess_data(data)

    # Bước 4: Chuẩn bị dữ liệu cho mô hình
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD', 'BB_Width']]
    y = data['Target']
    
    # Chia dữ liệu thành tập huấn luyện
    X_train = X
    y_train = y

    # Bước 5: Xây dựng mô hình
    model = build_model(X_train, y_train)

    # Dự đoán cho các ngày cụ thể
    future_dates = pd.date_range(start='2024-10-09', end='2024-10-11', freq='B')
    future_data = pd.DataFrame(future_dates, columns=['Date'])
    
    # Sử dụng giá trị cuối cùng trong dữ liệu để dự đoán
    last_row = data.iloc[-1]
    future_data['Open'] = last_row['Close']  # Giả định giá mở cửa bằng giá đóng cửa cuối cùng
    future_data['High'] = future_data['Open'] * (1 + np.random.uniform(0, 0.01))  # Tăng nhẹ
    future_data['Low'] = future_data['Open'] * (1 - np.random.uniform(0, 0.01))   # Giảm nhẹ
    future_data['Volume'] = 0  # Giả định khối lượng là 0 cho ngày tương lai
    
    # Giả định giá 'Close' cho future_data
    future_data['Close'] = future_data['Open']  # Giả định giá đóng cửa bằng giá mở cửa

    # Thêm các chỉ số kỹ thuật cho dữ liệu tương lai
    future_data = add_technical_indicators(future_data)

    # Dự đoán cho các ngày tương lai
    predictions = predict(model, future_data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50', 'RSI', 'MACD', 'BB_Width']])

    # Hiển thị kết quả
    future_data['Predicted_Close'] = predictions
    print(future_data[['Date', 'Predicted_Close']])

    # Vẽ biểu đồ
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Historical Prices', color='blue')
    plt.plot(future_data['Date'], future_data['Predicted_Close'], label='Predicted Prices', color='orange', marker='o')
    plt.title('GBPJPY Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()