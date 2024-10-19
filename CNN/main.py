import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ============================================================================================================


def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))  # 殘差平方和
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # 總變異平方和
    r2 = 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())  # 防止除以 0
    return r2

def preprocess_data(data_path):

    data=pd.read_excel(data_path)

    # 取出特定範圍波段
    # features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220], 
    #                       data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)
    features = pd.concat([data.iloc[:, 2901:3106]], axis=1)

    # 去掉正確答案的部分
    features = features.iloc[:, :-4]

    # 確保所有的特徵名稱都轉換為字串
    features.columns = features.columns.astype(str)

    # 取出目標變量「TOC(%)」
    target = data.loc[:, "TOC(%)"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 將數據重塑為 CNN 所需的三維格式 (樣本數, 特徵長度, 每個觀測點上只有一個數據)
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    return X_train_scaled, X_test_scaled, y_train, y_test

def build_and_train_cnn(X_train, y_train, X_test, y_test, epochs):
    # 構建 CNN 模型
    model = Sequential()
    # 添加一個一維卷積層
    model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
    # 因為它表示每個樣本有 X_train_scaled.shape[1] 個時間步長，每個時間步長有 1 個特徵。樣本數是隱藏的，這是由 Keras 在訓練過程中處理的，因此這仍然是三維的輸入格式。  
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='mse', metrics=['mae', r2_metric]) # 使用 Adam 優化器和均方誤差損失函數（適合迴歸）

    history = model.fit(X_train_scaled, y_train, epochs=epochs, validation_data=(X_test_scaled, y_test), batch_size=64)

    # 保存模型
    model.save('model/my_cnn_model.h5')

    # 評估模型
    train_loss, train_mae, train_r2 = model.evaluate(X_train, y_train)
    test_loss, test_mae, test_r2 = model.evaluate(X_test, y_test)

    print(f'Train Loss: {train_loss}, Train MAE: {train_mae}, Train R²: {train_r2}')
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}, Test R²: {test_r2}')

    return model, history

def plot_r2(history):
    # 提取 R² 數據
    r2_train = history.history['r2_metric']

    # 繪製 R² 趨勢圖
    plt.plot(r2_train, label='Training R²')
    plt.title('R² Trend During Training')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    plt.savefig("chart_result/R2_Trend_During_Training.png", bbox_inches="tight")
    plt.show()

def plot_loss(history):
    # 提取 loss 數據
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']

    # 繪製 Loss 趨勢圖
    plt.plot(loss_train, label='Training Loss')
    plt.plot(loss_val, label='Validation Loss')
    plt.title('Loss Trend During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("chart_result/Loss_Trend_During_Training.png", bbox_inches="tight")
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10,6))
    plt.plot(y_test.values, label='Actual Values', marker='o', linestyle='--')
    plt.plot(y_pred.flatten(), label='Predicted Values', marker='x', linestyle='-')
    plt.title('Actual vs Predicted TOC Values')
    plt.xlabel('Samples')
    plt.ylabel('TOC (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("chart_result/Actual_vs_Predicted_TOC.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data('dataset\FTIR(調基準線)_senior.xlsx')

     # 建立並訓練 CNN 模型
    model, history = build_and_train_cnn(X_train_scaled, y_train, X_test_scaled, y_test, 100000)

    # 繪製 R² 和 Loss 趨勢圖
    plot_r2(history)
    plot_loss(history)

    # 使用測試集進行預測
    y_pred = model.predict(X_test_scaled)

    # 繪製實際值與預測值
    plot_actual_vs_predicted(y_test, y_pred)