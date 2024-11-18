import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, AveragePooling1D, Dropout, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os
import random


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # 如果使用 GPU，設置確定性行為
    tf.config.experimental.enable_op_determinism()

def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))  
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  
    r2 = 1 - ss_res / (ss_tot + tf.keras.backend.epsilon()) 
    return r2

def preprocess_data(data_path):

    data = pd.read_excel(data_path)
    
    # 分離特徵與目標
    features = data.drop(columns=["number", "TC(%)", "TOC(%)", "TN(%)", "TIC(%)"])
    target = data.loc[:, "TOC(%)"]
    
    # 特徵標準化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 使用 PCA 降維，選擇主要的 1 個主成分
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(scaled_features)
    
    # 輸出 PCA 主成分權重
    pc1_contributions = pca.components_[0]
    contributions_df = pd.DataFrame({
        "波段": features.columns,
        "貢獻度": pc1_contributions
    })
    
    # 選擇貢獻度最高的 1000 個波段
    top_bands = contributions_df.nlargest(1000, "貢獻度")
    selected_features = features[top_bands["波段"]]
    
    # 使用選定波段進行分割
    X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=42)
    
    # 將數據重塑為 CNN 所需的三維格式
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test


def build_and_train_cnn(X_train, y_train, X_test, y_test, epochs):
    
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=7, padding='same', activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
    # 因為它表示每個樣本有 X_train_scaled.shape[1] 個時間步長，每個時間步長有 1 個特徵。樣本數是隱藏的，這是由 Keras 在訓練過程中處理的，因此這仍然是三維的輸入格式。  
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae', metrics=['mae', r2_metric]) # 使用 Adam 優化器和均方誤差損失函數（適合迴歸）

    history = model.fit(X_train_scaled, y_train, epochs=epochs, validation_data=(X_test_scaled, y_test), batch_size=256)


    model.save('soil_AI/model/my_cnn_model.h5')

    train_loss, train_mae, train_r2 = model.evaluate(X_train, y_train)
    test_loss, test_mae, test_r2 = model.evaluate(X_test, y_test)

    print(f'Train Loss: {train_loss}, Train MAE: {train_mae}, Train R²: {train_r2}')
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}, Test R²: {test_r2}')

    return model, history

def plot_r2(history):
    r2_train = history.history['r2_metric']

    plt.figure(figsize=(10,6))
    plt.plot(r2_train, label='Training R²')
    plt.title('R² Trend During Training')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    plt.savefig("soil_AI/chart_result/R2_Trend_During_Training.png", bbox_inches="tight")

def plot_loss(history):
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']

    plt.figure(figsize=(10,6))
    plt.plot(loss_train, label='Training Loss')
    plt.plot(loss_val, label='Validation Loss')
    plt.title('Loss Trend During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("soil_AI/chart_result/Loss_Trend_During_Training.png", bbox_inches="tight")

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10,6))
    plt.plot(y_test.values, label='Actual Values', marker='o', linestyle='--')
    plt.plot(y_pred.flatten(), label='Predicted Values', marker='x', linestyle='-')
    plt.title('Actual vs Predicted TOC Values')
    plt.xlabel('Samples')
    plt.ylabel('TOC (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("soil_AI/chart_result/Actual_vs_Predicted_TOC.png", bbox_inches="tight")


if __name__ == "__main__":
    set_seed(2)  # 設定固定隨機種子
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data('soil_AI/dataset/ALL_4000_750_881筆(未調).xlsx')

     # 建立並訓練 CNN 模型
    model, history = build_and_train_cnn(X_train_scaled, y_train, X_test_scaled, y_test, 600)

    plot_r2(history)
    plot_loss(history)
    
    y_pred = model.predict(X_test_scaled)
    plot_actual_vs_predicted(y_test, y_pred)
    
    
# 減少濾波器數量