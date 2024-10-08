import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import lightgbm as lgb

# **SYSTEM CONFIGURATIONS **


# **CLASS DEFINITIONS**

class Visualization:
    @staticmethod
    def plot_yearly_data(data, title='Yearly Price Data'):
        plt.figure(figsize=(14, 7))
        plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_predictions_vs_actual(y_actual, y_predicted, title='Predicted vs Actual Prices'):
        plt.figure(figsize=(14, 7))
        plt.plot(y_actual, label='Actual', color='blue')
        plt.plot(y_predicted, label='Predicted', color='orange')
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def save_plot(y_actual, y_predicted, title='Predicted vs Actual Prices', filename='prediction.png'):
        plt.figure(figsize=(14, 7))
        plt.plot(y_actual, label='Actual', color='blue')
        plt.plot(y_predicted, label='Predicted', color='orange')
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f'Plot saved to {filename}')

class FineTuning:
    @staticmethod
    def light_gbm_fine_tuning(X_train, y_train):
        # param_grid = {
        #     'n_estimators': [100, 200, 500],
        #     'max_depth': [3, 5, 7, 10],
        #     'learning_rate': [0.01, 0.05, 0.1],
        #     'subsample': [0.8, 0.9, 1.0],
        #     'colsample_bytree': [0.8, 0.9, 1.0],
        #     'reg_alpha': [0, 0.1, 0.5],
        #     'reg_lambda': [0, 0.1, 0.5]
        # }
        
        param_grid = {
            'n_estimators': [100],      # Chỉ thử với một giá trị
            'max_depth': [3],           # Chỉ thử với một giá trị
            'learning_rate': [0.1],     # Chỉ thử với một giá trị
            'subsample': [0.8],         # Chỉ thử với một giá trị
            'colsample_bytree': [0.8],  # Chỉ thử với một giá trị
            'reg_alpha': [0],           # Chỉ thử với một giá trị
            'reg_lambda': [0]           # Chỉ thử với một giá trị
        }

               
        lgb_model = lgb.LGBMRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        joblib.dump(grid_search, 'saved_objects/LGBMRegressor_gridsearch.pkl')

        print("\n====== Fine-tune LGBMRegressor ======")
        print('Best hyperparameter combination: ', grid_search.best_params_)
        print('Best RMSE: ', np.sqrt(-grid_search.best_score_)) 

        return grid_search.best_estimator_

    @staticmethod
    def polynomial_regression_fine_tuning(X_train, y_train):
        param_grid = {
            'polynomialfeatures__degree': [2, 3, 4],
            'linearregression__fit_intercept': [True, False],
        }
        
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('polynomialfeatures', PolynomialFeatures()),
            ('scaler', StandardScaler()),
            ('linearregression', LinearRegression())
        ])
        
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        joblib.dump(grid_search, 'saved_objects/PolynomialRegression_gridsearch.pkl')

        print("\n====== Fine-tune Polynomial Regression ======")
        print('Best hyperparameter combination: ', grid_search.best_params_)
        print('Best RMSE: ', np.sqrt(-grid_search.best_score_)) 

        return grid_search.best_estimator_

    @staticmethod
    def decision_tree_fine_tuning(X_train, y_train):
        # param_grid = {
        #     'max_depth': [3, 5, 7, 10, None],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'max_features': ['auto', 'sqrt', 'log2', None]
        # }
        
        # param_grid = {
        #     'max_depth': [3, 5, None],       # Giảm số lượng giá trị
        #     'min_samples_split': [2, 5],     # Giảm số lượng giá trị
        #     'min_samples_leaf': [1, 2],      # Giảm số lượng giá trị
        #     'max_features': ['auto', 'sqrt'] # Giảm số lượng giá trị
        # }
           
        # param_grid = {
        #     'max_depth': [None, 3, 5, 10],
        #     'max_features': ['sqrt', 'log2'],  # Đảm bảo chỉ sử dụng các giá trị hợp lệ
        #     'min_samples_leaf': [1, 2, 4],
        #     'min_samples_split': [2, 5, 10]
        # }
        param_grid = {
            'max_depth': [None, 3, 5],  # Giữ lại None, 3 và 5
            'max_features': ['sqrt'],    # Chỉ giữ lại 'sqrt'
            'min_samples_leaf': [2, 4],  # Chỉ giữ lại 2 và 4
            'min_samples_split': [2, 5]   # Chỉ giữ lại 2 và 5
        }

        dt_model = DecisionTreeRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        joblib.dump(grid_search, 'saved_objects/DecisionTreeRegressor_gridsearch.pkl')

        print("\n====== Fine-tune Decision Tree Regressor ======")
        print('Best hyperparameter combination: ', grid_search.best_params_)
        print('Best RMSE: ', np.sqrt(-grid_search.best_score_)) 

        return grid_search.best_estimator_

    @staticmethod
    def random_forest_fine_tuning(X_train, y_train):
        # param_grid = {
        #     'n_estimators': [100, 200, 500],
        #     'max_depth': [3, 5, 7, 10, None],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4],
        #     'max_features': ['auto', 'sqrt', 'log2']
        # }
        # param_grid = {
        #     'n_estimators': [100, 200],     # Giảm số lượng giá trị
        #     'max_depth': [3, 5, None],      # Giảm số lượng giá trị
        #     'min_samples_split': [2, 5],    # Giảm số lượng giá trị
        #     'min_samples_leaf': [1, 2],     # Giảm số lượng giá trị
        #     'max_features': ['auto', 'sqrt']# Giảm số lượng giá trị
        # }

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2'],  # Chỉ sử dụng các giá trị hợp lệ
            'min_samples_leaf': [1, 2],
            'min_samples_split': [2, 5]
        }
        
        param_grid = {
            'n_estimators': [100],         # Chỉ giữ lại 100
            'max_depth': [None, 10],       # Giữ lại None và 10
            'max_features': ['sqrt'],       # Chỉ giữ lại 'sqrt'
            'min_samples_leaf': [2],        # Chỉ giữ lại 2
            'min_samples_split': [2]        # Chỉ giữ lại 2
        }


        rf_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        joblib.dump(grid_search, 'saved_objects/RandomForestRegressor_gridsearch.pkl')

        print("\n====== Fine-tune Random Forest Regressor ======")
        print('Best hyperparameter combination: ', grid_search.best_params_)
        print('Best RMSE: ', np.sqrt(-grid_search.best_score_)) 

        return grid_search.best_estimator_

# **FUNCTION DEFINITIONS**

def crawl_forex_data(forex_pair, directory='Dataset'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    print(f"Downloading data for {forex_pair}...")
    data = yf.download(forex_pair, period="max", interval='1d')
    data.reset_index(inplace=True)
    
    file_path = os.path.join(directory, f'{forex_pair}_data.csv')
    data.to_csv(file_path, index=False)
    print(f"Saved data for {forex_pair} at {file_path}")

    return data

def feature_engineering(data):
    if 'Close' in data.columns:
        if len(data) >= 10: data['MA_10'] = data['Close'].rolling(window=10).mean()
        if len(data) >= 50: data['MA_50'] = data['Close'].rolling(window=50).mean()
        if len(data) >= 200: data['MA_200'] = data['Close'].rolling(window=200).mean()

    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / data['Close']
    data['Volume_24h'] = data['Volume'].rolling(window=1440).sum()
    data['ADL'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
    aroon = ta.trend.AroonIndicator(data['Close'], data['Low'], window=25)
    data['Aroon_Up'] = aroon.aroon_up()
    data['Aroon_Down'] = aroon.aroon_down()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    data['AO'] = ta.momentum.AwesomeOscillatorIndicator(data['High'], data['Low'], window1=5, window2=34).awesome_oscillator()
    data['BOP'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'])
    data['Bull_Power'] = data['High'] - data['MA_50']
    data['Bear_Power'] = data['Low'] - data['MA_50']
    data['Chaikin_Osc'] = data['ADL'].ewm(span=3).mean() - data['ADL'].ewm(span=10).mean()
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14)
    data['Stoch_Osc'] = stoch.stoch()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close'], window=20).cci()
    vortex = ta.trend.VortexIndicator(data['High'], data['Low'], data['Close'], window=14)
    data['Vortex_Plus'] = vortex.vortex_indicator_pos()
    data['Vortex_Minus'] = vortex.vortex_indicator_neg()

    return data

def handle_missing_values(data):
    data.interpolate(method='linear', inplace=True)
    if data.isna().sum().sum() > 0:
        # data.fillna(method='ffill', inplace=True)
        data.ffill(inplace=True)
        if data.isna().sum().sum() > 0:
            # data.fillna(method='bfill', inplace=True)
            data.bfill(inplace=True)
    return data.dropna(subset=['Close', 'Open'])

def find_unrelated_features(data, targets):
    X = data.drop(columns=['Date'] + targets)
    mi_results = pd.DataFrame()

    for target in targets:
        y = data[target]
        mi_scores = mutual_info_regression(X, y)
        mi_df = pd.DataFrame(mi_scores, index=X.columns, columns=[f'MI Score_{target}'])
        mi_results = pd.concat([mi_results, mi_df], axis=1)

    return mi_results

def drop_unrelated_features(data, mi_results, targets, threshold=0.25):
    features_to_drop = set()
    for target in targets:
        low_mi_features = mi_results[mi_results[f'MI Score_{target}'] < threshold].index.tolist()
        features_to_drop.update(low_mi_features)

    return data.drop(columns=list(features_to_drop))

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, dataframe, labels=None):
        return self

    def transform(self, dataframe):
        return dataframe[self.feature_names].values

def prepare_data(data, targets):
    k = -1
    for target_column in targets:
        data[target_column] = data[target_column].shift(k)

    data = data.dropna()
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    feature_columns = data.columns.difference(['Date'] + targets).tolist()
    X_reduced = data.drop(columns=['Date'] + targets)
    X_train = train_set[X_reduced.columns]
    X_test = test_set[X_reduced.columns]

    y_train_dict = {target: train_set[target] for target in targets}
    y_test_dict = {target: test_set[target] for target in targets}

    return X_train, X_test, y_train_dict, y_test_dict

def build_numerical_pipeline(X_reduced):
    num_feat_names = X_reduced.columns.tolist()
    num_pipeline = Pipeline(steps=[
        ('selector', ColumnSelector(num_feat_names)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler(with_mean=True, with_std=True))
    ])

    return num_pipeline

# New function for prediction
def predict(model, X_data, num_pipeline):
    X_transformed = num_pipeline.transform(X_data)
    # predictions = model.predict(X_transformed)
    return model.predict(X_transformed)

def feature_engineering_for_future_predictions(data, past_data):
    # Tính toán các chỉ số kỹ thuật cho dữ liệu trong tương lai
    if 'Close' in data.columns:
        if len(data) >= 10: data['MA_10'] = data['Close'].rolling(window=10).mean()
        if len(data) >= 50: data['MA_50'] = data['Close'].rolling(window=50).mean()
        if len(data) >= 200: data['MA_200'] = data['Close'].rolling(window=200).mean()

    if 'Volume' in data.columns: data['Volume_24h'] = data['Volume'].rolling(window=1440).sum()
        
    # Tính toán RSI
    if 'Close' in data.columns: data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    
    # Tính toán MACD
    if 'Close' in data.columns: data['MACD'] = ta.trend.MACD(data['Close']).macd()
    
    # Tính toán Bollinger Bands
    if 'Close' in data.columns:
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_High'] = bollinger.bollinger_hband()
        data['BB_Low'] = bollinger.bollinger_lband()
        data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / data['Close']
    
    # Tính toán ADL
    if 'High' in data.columns and 'Low' in data.columns: 
        data['ADL'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
    
    # Tính toán Aroon
    if 'Close' in data.columns:
        aroon = ta.trend.AroonIndicator(data['Close'], data['Low'], window=25)
        data['Aroon_Up'] = aroon.aroon_up()
        data['Aroon_Down'] = aroon.aroon_down()

    # Thêm các cột khác mà mô hình yêu cầu
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    data['AO'] = ta.momentum.AwesomeOscillatorIndicator(data['High'], data['Low'], window1=5, window2=34).awesome_oscillator()
    data['BOP'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'])
    data['Bull_Power'] = data['High'] - data['MA_50']
    data['Bear_Power'] = data['Low'] - data['MA_50']
    
    return data


def handle_missing_prediction_values(data):
    """
    Hàm này xử lý các giá trị NaN trong DataFrame mà có thể tạo thêm cột tạm thời, 
    nhưng sẽ xóa đi sau khi hoàn tất.
    
    Args:
        data (pd.DataFrame): DataFrame chứa dữ liệu cần xử lý.
    
    Returns:
        pd.DataFrame: DataFrame đã được xử lý với các giá trị NaN được lấp đầy.
    """
    # Tạo một bản sao của DataFrame để không làm thay đổi dữ liệu gốc
    temp_data = data.copy()
    
    # Nội suy các giá trị NaN
    temp_data.interpolate(method='linear', inplace=True, limit_direction='both')
    
    # Sử dụng forward fill để lấp đầy các giá trị NaN còn lại
    temp_data.ffill(inplace=True)
    
    # Sử dụng backward fill để lấp đầy các giá trị NaN còn lại
    temp_data.bfill(inplace=True)
    
    # Cập nhật các giá trị trong DataFrame gốc
    data.update(temp_data)
    
    return data

# Funtion for generating future predictions:
def future_predictions(model, start_date, end_date, past_data):
    # Tạo một DataFrame mới để lưu trữ dữ liệu dự đoán
    future_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' cho ngày làm việc
    future_data = pd.DataFrame(future_dates, columns=['Date'])
    
    # Giả định giá mở cửa và các giá trị khác cho các ngày trong tương lai
    future_data['Open'] = past_data['Close'].iloc[-1]  # Giá mở cửa bằng giá đóng cửa cuối cùng
    future_data['Close'] = np.nan  # Chưa có giá trị Close
    future_data['High'] = future_data['Open'] * (1 + np.random.uniform(-0.01, 0.01))  # Giá cao ngẫu nhiên
    future_data['Low'] = future_data['Open'] * (1 - np.random.uniform(-0.01, 0.01))  # Giá thấp ngẫu nhiên
    future_data['Volume'] = 0  # Giả định không có giao dịch trong tương lai

    # Tính toán các chỉ số cho dữ liệu trong tương lai
    future_data = feature_engineering_for_future_predictions(future_data, past_data)
    
    # Xử lý các giá trị NaN trong future_data
    future_data = handle_missing_prediction_values(future_data)
    
    # Lấy danh sách các cột đầu vào của mô hình
    input_columns = model.feature_names_in_  # Hoặc bạn có thể định nghĩa danh sách cột này thủ công

    # Chỉ giữ lại các cột cần thiết trong future_data
    future_data = future_data[input_columns]

    # Thực hiện dữ liệu dự đoán cho tương lai
    future_data['Predicted_Close'] = model.predict(future_data)

    return future_data

    
    
# **MAIN FUNCTION**
def main():
    # Step 1: Define Label
    forex_label = ['GBPJPY=X']
    targets = ['Close']

    # Step 2: Crawl Forex Data
    data = crawl_forex_data(forex_label[0])

    # Step 3: Feature Engineering
    data = feature_engineering(data)

    # Step 3.1: Handle Missing Values
    data = handle_missing_values(data)
    
    # Step 4: Plot Yearly Data
    # Visualization.plot_yearly_data(data)

    # Step 5: Find and Drop Unrelated Features
    targets = ['Close', 'Open', 'High', 'Low']
    mi_results = find_unrelated_features(data, targets)
    data = drop_unrelated_features(data, mi_results, targets)

    # Step 6: Prepare Data
    X_train, X_test, y_train_dict, y_test_dict = prepare_data(data, targets)
    
    # **6.1 Lấy y_train và y_test từ dictionary
    y_train = y_train_dict['Close']
    y_test = y_test_dict['Close']
    
    # **6.2 Kiểm tra kích thước
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    # Step 7: Build Numerical Pipeline
    num_pipeline = build_numerical_pipeline(X_train)

    # Step 8: Fine-tune Models
    run_new_fine_tune = 0
    if run_new_fine_tune == 1:
        best_model = FineTuning.light_gbm_fine_tuning(X_train, y_train_dict['Close'])
        # Uncomment below lines to fine-tune other models
        best_model = FineTuning.polynomial_regression_fine_tuning(X_train, y_train_dict['Close'])
        best_model = FineTuning.decision_tree_fine_tuning(X_train, y_train_dict['Close'])
        best_model = FineTuning.random_forest_fine_tuning(X_train, y_train_dict['Close'])
    else:
        try:
            # Chỉ tải mô hình tốt nhất từ LightGBM
            loaded_model = joblib.load('saved_objects/LGBMRegressor_gridsearch.pkl')
            best_model = loaded_model.best_estimator_
            print("Loaded saved LightGBM Model.")
        except FileNotFoundError:
            print("No saved model found. Please ensure the model is trained and saved correctly.")
            return
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return

    # Step 9: Make predictions
    # predictions = predict(best_model, X_test, num_pipeline)
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    # print("\nPredictions: ", predictions[:9])
    # print("Actual Labels: ", list(y_test_dict['Close'][:9])
    
    # Step 10: Future Predictions:
    start_date = '2024-10-08'
    end_date = '2024-10-12'
    future_data = future_predictions(best_model, start_date, end_date, data)
    
    # Hiển thị kết quả dự đoán 
    if future_data is not None: print(future_data)

    
    # # Step 10: Plot Predictions vs Actual
    # Visualization.plot_predictions_vs_actual(y_test_dict['Close'].values, predictions)

if __name__ == "__main__":
    main()