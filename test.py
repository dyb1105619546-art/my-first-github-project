import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


# 1. 数据加载和预处理
def load_and_preprocess_data(file_path):
    # 加载数据
    data = pd.read_csv(file_path)

    # 将日期时间列转换为datetime类型并设为索引
    data['date_time'] = pd.to_datetime(data['date_time'])
    data.set_index('date_time', inplace=True)

    # 检查缺失值
    print("缺失值统计:")
    print(data.isnull().sum())

    # 填充缺失值
    data.fillna(method='ffill', inplace=True)

    # 选择特征和目标变量
    features = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
                'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection',
                'nwp_pressure', 'lmd_totalirrad', 'lmd_diffuseirrad',
                'lmd_temperature', 'lmd_pressure', 'lmd_winddirection',
                'lmd_windspeed']
    target = 'power'

    X = data[features]
    y = data[[target]]

    # 数据标准化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y


# 2. 创建时间序列数据集
def create_dataset(X, y, look_back=24 * 4, forecast_horizon=4):
    X_data, y_data = [], []
    for i in range(len(X) - look_back - forecast_horizon + 1):
        X_data.append(X[i:(i + look_back)])
        y_data.append(y[i + look_back:i + look_back + forecast_horizon, 0])

    return np.array(X_data), np.array(y_data)


# 3. 构建LSTM模型
def build_lstm_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(16))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mse')
    return model


# 4. 主函数
def main():
    # 参数设置
    look_back = 7*24*4  # 使用6小时的数据(15分钟间隔，24=6小时)
    forecast_horizon =7*24*4  # 预测未来1小时(4个15分钟)

    # 1. 加载和预处理数据
    file_path = r"C:\Users\20586\Desktop\station08.csv"  # 修改为你的文件路径
    try:
        X_scaled, y_scaled, scaler_X, scaler_y = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"加载文件出错: {e}")
        return

    # 2. 创建时间序列数据集
    X, y = create_dataset(X_scaled, y_scaled, look_back, forecast_horizon)

    # 3. 划分训练集和测试集
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

    # 4. 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, forecast_horizon)
    model.summary()

    # 5. 训练模型
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # 6. 评估模型
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"测试集RMSE: {rmse:.4f}")

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 可视化第一个样本的预测结果
    # 创建时间轴
    hours = np.arange(forecast_horizon) / 4

    # 绘制图形
    plt.figure(figsize=(15, 6))
    plt.plot(hours, y_true[0], 'b-', label='真实值', linewidth=2)
    plt.plot(hours, y_pred[0], 'r--', label='预测值', linewidth=2)

    # 添加标签和标题
    plt.title('光伏功率7天→未来7天预测', fontsize=14, pad=20)
    plt.xlabel('时间 (小时)', fontsize=12)
    plt.ylabel('功率 (kW)', fontsize=12)

    # 设置坐标轴
    plt.xticks(np.arange(0, 168.1, 24))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    # 调整布局
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


