import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_sequences(data, window_size):
    """
    根据滑动窗口构造 LSTM 训练数据
    data: ndarray, shape (n_samples, 1)
    window_size: int, 窗口大小
    return: X, y
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, units=70):
    """
    构建 LSTM 模型
    input_shape: (time_steps, features)
    units: LSTM 单元数量
    """
    model = Sequential()
    model.add(LSTM(units=units, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def evaluate_model(y_true, y_pred, name="LSTM Model"):
    """
    计算 RMSE、MAE、R²
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")
    print("-" * 30)


def run_lstm_pipeline(df, window_size=120, lstm_units=70, epochs=30, batch_size=16):
    """
    运行 LSTM 预测流程
    df: DataFrame, 必须包含 'date' 和 'global_irradiation' 列
    """
    # 1. 按日期聚合（防止同一天多条数据）
    daily_avg = df.groupby('date')['global_irradiation'].mean().reset_index()

    # 2. 归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_avg[['global_irradiation']])

    # 3. 构造序列
    X, y = create_sequences(scaled_data, window_size)

    # 4. 划分训练和测试集（80/20）
    split_index = int(len(X) * 0.8)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    # 5. 构建 LSTM 模型
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), units=lstm_units)

    # 6. 训练模型
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_test, y_test), 
                        verbose=1)

    # 7. 预测
    y_pred = model.predict(X_test)

    # 8. 反归一化
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    # 9. 评估模型
    evaluate_model(y_test_inv, y_pred_inv, name="LSTM Model")

    # 10. 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.title('LSTM Prediction vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Global Irradiation')
    plt.legend()
    plt.tight_layout()
    plt.show()


