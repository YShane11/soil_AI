<<<<<<< HEAD
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_data(data_path):
    # 載入數據
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


if __name__ == "__main__":
    # 預處理數據
    X_train_scaled, X_test_scaled, y_train, y_test, contributions = preprocess_data('dataset/FTIR(調基準線)_senior.xlsx')
    
    # 輸出選擇的波段貢獻度
    print("Top PCA Contributions:")
    print(contributions.head(10))
=======
import tensorflow as tf

print(tf.test.is_gpu_available())
>>>>>>> aaf402be51fca190ffabfdfc1fc78f9df1e7fe56
