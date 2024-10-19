import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# ============================================================================================================

data=pd.read_excel('FTIR_dust-main/dataset/FTIR_Data_FCU.xlsx')

# 取出特定範圍波段
# features = pd.concat([data.iloc[:, 650:820], data.iloc[:, 850:1220], 
#                       data.iloc[:, 1250:1750], data.iloc[:, 2800:3000]], axis=1)
features = pd.concat([data.iloc[:, 2900:3101]], axis=1)

# 確保所有的特徵名稱都轉換為字串
features.columns = features.columns.astype(str)

# 取出目標變量「TOC(%)」
target = data.loc[:, "TOC(%)"]

# ============================================================================================================


# 加載最佳模型
best_model = joblib.load("FTIR_dust-main/best_model/PLSRegression_best_model.joblib")

# 使用加載的模型進行預測
predictions = best_model.predict(features)


mse = mean_squared_error(target, predictions)
r2 = r2_score(target, predictions)


print(f'''
    MSE: {mse},
    R2: {r2}
       ''')



plt.figure(figsize=(10, 6))

# 折線圖：實際值
plt.plot(target.values, label='Actual Data', color='blue')

# 折線圖：預測值
plt.plot(predictions, label='Predicted Data', color='red', linestyle='--')

# 添加圖例與標題
plt.title('Actual vs Predicted TOC(%)')
plt.xlabel('Sample Index')
plt.ylabel('TOC(%)')
plt.legend()

# 顯示圖表
plt.show()