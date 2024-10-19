import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

# ============================================================================================================

data=pd.read_excel('FTIR_dust-main\dataset\FTIR(調基準線)_senior.xlsx')

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

# ============================================================================================================

# 初始化保存結果的list
results = []

# 切割資料
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=11)

models = {
    "PLSRegression": PLSRegression(),
}

param_grids = {
    "PLSRegression": {
        'n_components': [2, 3, 5, 7, 10],
        'scale': [True, False],
        'max_iter': [150, 300, 500, 1000, 1500],
        'tol': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    }
}

for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 保存最佳模型
    joblib.dump(best_model, f'./model/{model_name}_best_model.joblib')

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # 保存結果到results列表中
    results.append({
        "Model": model_name,
        "Best Parameters": grid_search.best_params_,
        "Training MSE": train_mse,
        "Training R2": train_r2,
        "Testing MSE": test_mse,
        "Testing R2": test_r2
    })

# 將結果轉換為DataFrame
df_results = pd.DataFrame(results)

# 儲存為xlsx文件
df_results.to_excel('model/model_results.xlsx', index=False)

print("結果已成功保存到 'model_results.xlsx'")
print(f"最佳模型已保存到 '{model_name}_best_model.joblib'")