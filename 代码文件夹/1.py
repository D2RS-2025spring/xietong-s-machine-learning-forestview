# 假设已经有了模型名称和 MSE 值，构建字典
model_results = {
    "Model1": {"MSE": 0.5}
    "Model2": {"MSE": 0.3}
}
model_names = list(model_results.keys())
mse_values = [model_results[model]["MSE"] for model in model_names]
# --- 提取模型名称和 MSE 值 ---
model_names = list(model_results.keys()) 
mse_values = [model_results[model]['MSE'] for model in model_names] 
#创建了一个字典model_results，包含两个模型(Model1和Model2)及其对应的MSE值 提取模型名称到model_names列表 提取MSE值到mse_values列表
# --- 创建并显示条形图 ---
plt.figure(figsize=(10, 6)) 
plt.bar(model_names, mse_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'mediumpurple'])
plt.xlabel("Model") 
plt.ylabel("Mean Squared Error (MSE)") 
plt.title(" Error (MSE)") 
plt.title("Initial Model Performance Comparison (MSE)") 
plt.xticks(rotation=45, ha="right") 
plt.ylim(bottom=0, top=max(mse_values) * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() 
plt.show() 
#创建10x6英寸大小的图形；使用plt.bar()绘制条形图，为不同模型分配不同颜色；设置X轴标签为"Model"，Y轴标签为"MSE"；添加标题"Initial Model Performance Comparison (MSE)"；将X轴标签旋转45度以便更好显示；设置Y轴范围从0到最大MSE值的1.1倍；添加网格线(仅Y轴，虚线，半透明)；使用tight_layout()自动调整子图参数
print("\nThe bar chart visualizes the Mean Squared Error (MSE) for each model trained with default hyperparameters.")
print("Lower MSE indicates better performance on the test set.")
#显示生成的条形图，说明图表展示的是使用默认超参数训练的模型的MSE比较；指出MSE值越低表示模型在测试集上性能越好
# --- 定义 Optuna 的目标函数 ---
def objective(trial, model_name):
    #这是一个目标函数，用于定义优化目标和参数搜索空间
    global X_train, y_train

    if model_name == "DecisionTree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        model = DecisionTreeRegressor(**params, random_state=42)

    elif model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        model = RandomForestRegressor(**params, random_state=42)

    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = XGBRegressor(**params, random_state=42, objective='reg:squarederror', booster='gbtree')

    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", -1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        model = LGBMRegressor(**params, random_state=42, verbose=-1)

    elif model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        }
        model = CatBoostRegressor(**params, random_state=42, verbose=0)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()
    return -score
#这是一个目标函数，用于定义优化目标和参数搜索空间：
#输入参数：trial: Optuna的Trial对象，用于建议参数值 ；model_name: 字符串，指定要优化的模型类型
#功能:根据model_name选择不同的模型和参数空间;使用trial.suggest_*方法定义每个参数的搜索范围
#支持的模型包括：DecisionTree(决策树);RandomForest(随机森林);XGBoost(XGBoost);LightGBM(LightGBM);CatBoost(CatBoost)
#参数优化：对每种模型定义了关键的超参数及其搜索范围；例如，决策树的max_depth(2-10)、随机森林的n_estimators(50-300)等
#评估方法：使用5折交叉验证(cross_val_score)；评估指标为负均方误差(neg_mean_squared_error)
#返回正均方误差(通过-score转换
# --- 定义运行 Optuna 优化的辅助函数 ---
def tune_model(model_name, n_trials=20):
    #这是一个辅助函数，用于执行实际的优化过程：
    print(f"\nStarting hyperparameter tuning for {model_name}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials)
#model_name: 要优化的模型名称；n_trials: 优化尝试次数，默认为20
    print(f"Best parameters found for {model_name}: {study.best_params}")
    print(f"Best score (MSE) achieved during tuning: {-study.best_value:.4f}")
    return study.best_params
#创建Optuna study对象，指定优化方向为"minimize"(最小化MSE)；运行优化过程，尝试指定次数的参数组合；打印最佳参数和对应的MSE值；返回最佳参数组合
# --- 调整每个模型的超参数 ---
dt_params = tune_model("DecisionTree", n_trials=50)
#调优 决策树模型，尝试50组不同的超参数组合（如 max_depth、min_samples_split 等）。返回最佳参数，保存到变量 dt_params
rf_params = tune_model("RandomForest", n_trials=50)
#调优 随机森林模型，尝试50组参数（如 n_estimators、max_depth 等）。返回最佳参数，保存到 rf_params
xgb_params = tune_model("XGBoost", n_trials=50)
#调优 XGBoost模型，尝试50组参数（如 learning_rate、subsample 等）。返回最佳参数，保存到 xgb_params
lgb_params = tune_model("LightGBM", n_trials=50)
#调优 LightGBM模型，尝试50组参数（如 num_leaves、max_depth 等）。返回最佳参数，保存到 lgb_params
cat_params = tune_model("CatBoost", n_trials=50)
#调优 CatBoost模型，尝试50组参数（如 iterations、depth 等）。返回最佳参数，保存到 cat_params
# --- 使用找到的最佳参数重新初始化模型 ---
dt_model = DecisionTreeRegressor(**dt_params, random_state=42)
#dt_params：包含通过 tune_model("DecisionTree") 找到的最佳参数（如 max_depth、min_samples_split 等）。random_state=42：固定随机种子，确保结果可复现。**dt_params：将字典中的参数解包传递给模型构造函数
rf_model = RandomForestRegressor(**rf_params, random_state=42)
#rf_params：包含通过 tune_model("RandomForest") 找到的最佳参数（如 n_estimators、max_depth 等）。同样固定 random_state 保证可复现性
xgb_model = XGBRegressor(**xgb_params, random_state=42, objective='reg:squarederror', booster='gbtree')
#xgb_params：包含通过 tune_model("XGBoost") 找到的最佳参数（如 learning_rate、subsample 等）。额外指定：objective='reg:squarederror'：使用均方误差作为损失函数。booster='gbtree'：使用基于树的模型（默认选项）
lgb_model = LGBMRegressor(**lgb_params, random_state=42, verbose=-1)
#lgb_params：包含通过 tune_model("LightGBM") 找到的最佳参数（如 num_leaves、learning_rate 等）。verbose=-1：关闭训练过程中的日志输出（静默模式）
cat_model = CatBoostRegressor(**cat_params, random_state=42, verbose=0)
#cat_params：包含通过 tune_model("CatBoost") 找到的最佳参数（如 iterations、depth 等）。verbose=0：关闭训练过程中的日志输出。
# --- 重新训练模型 ---
print("\nRe-training models with optimized hyperparameters...")
#输出一条提示信息，表明程序正在使用优化后的超参数重新训练模型。
start_time = time.time()
#调用 time.time() 记录训练开始的时间戳，用于后续计算训练耗时。
dt_model.fit(X_train, y_train)
#调用 fit 方法训练决策树模型：X_train：训练集的特征数据。y_train：训练集的目标值（标签）。模型使用之前通过 Optuna 调优后的超参数（dt_params）进行初始化
end_time = time.time()
#调用 time.time() 记录训练结束的时间戳
print(f"Optimized Decision Tree training time: {end_time - start_time:.2f} seconds")
#计算训练耗时（end_time - start_time），并格式化为保留两位小数的秒数。
start_time = time.time()
#作用：调用Python的time模块，记录模型训练开始的时间点。返回值：start_time保存的是一个时间戳（单位：秒），例如1620000000.123456
rf_model.fit(X_train, y_train)
#作用：训练随机森林回归模型。X_train：训练数据的特征矩阵（二维数组，形状为[n_samples, n_features]）。y_train：训练数据的目标值（一维数组，形状为[n_samples]）。说明：rf_model是通过RandomForestRegressor(**rf_params, random_state=42)初始化的，其中rf_params是Optuna调优得到的最佳参数。训练过程会根据数据和超参数（如n_estimators、max_depth等）构建多棵决策树。
end_time = time.time()
#作用：记录模型训练完成时的时间戳。返回值：与start_time类似，保存的是训练结束的时间点
print(f"Optimized Random Forest training time: {end_time - start_time:.2f} seconds")
#计算：end_time - start_time得到训练耗时（单位：秒）
start_time = time.time()
#这行代码记录了训练开始的时间。time.time() 是 Python 中 time 模块的一个函数，返回当前时间的时间戳（从1970年1月1日00:00:00 UTC到现在的秒数）。start_time 变量保存了训练开始时的时间戳
xgb_model.fit(X_train, y_train)
#这行代码调用了 XGBoost 模型的 fit 方法，用于训练模型。X_train 是训练数据的特征矩阵，通常是一个二维数组，其中每一行是一个样本，每一列是一个特征。y_train 是训练数据的目标变量（标签），通常是一个一维数组，与 X_train 的行数相同。xgb_model 是一个 XGBoost 模型实例，可能通过 XGBClassifier 或 XGBRegressor 创建
end_time = time.time()
#这行代码记录了训练结束的时间。同样使用 time.time() 获取当前时间的时间戳，并将其保存到变量 end_time 中
print(f"Optimized XGBoost training time: {end_time - start_time:.2f} seconds")
#这行代码计算并打印出模型训练所花费的时间。end_time - start_time 计算出训练时间的差值，单位是秒。：.2f 是格式化字符串的语法，表示将时间差值保留两位小数
start_time = time.time()
#这行代码记录了训练开始的时间。time.time() 是 Python 中 time 模块的一个函数，返回当前时间的时间戳（从1970年1月1日00:00:00 UTC到现在的秒数）start_time 变量保存了训练开始时的时间戳
lgb_model.fit(X_train, y_train)
#这行代码调用了 LightGBM 模型的 fit 方法，用于训练模型
end_time = time.time()
#这行代码记录了训练结束的时间。同样使用 time.time() 获取当前时间的时间戳，并将其保存到变量 end_time 中。
print(f"Optimized LightGBM training time: {end_time - start_time:.2f} seconds")
#这行代码计算并打印出模型训练所花费的时间
start_time = time.time()
#记录训练开始的时间。time.time() 返回当前时间的时间戳（从1970年1月1日00:00:00 UTC到现在的秒数）。start_time 变量保存了训练开始时的时间戳
cat_model.fit(X_train, y_train)
#调用CatBoost模型的 fit 方法进行训练。
end_time = time.time()
#记录训练结束的时间。end_time 变量保存了训练结束时的时间戳。
print(f"Optimized CatBoost training time: {end_time - start_time:.2f} seconds")
#计算训练时间：end_time - start_time，单位为秒。
print("\nModels re-trained successfully with optimized hyperparameters.")
#打印一条消息，表示模型已经重新训练完成，并且使用了优化后的超参数。
# --- 使用优化后的模型进行预测 ---
dt_predictions = dt_model.predict(X_test)
#使用决策树模型 (dt_model) 对测试数据集 X_test 进行预测。
rf_predictions = rf_model.predict(X_test)
#使用决策树模型 (dt_model) 对测试数据集 X_test 进行预测。
xgb_predictions = xgb_model.predict(X_test)
#使用决策树模型 (dt_model) 对测试数据集 X_test 进行预测。
lgbm_predictions = lgb_model.predict(X_test)
#使用决策树模型 (dt_model) 对测试数据集 X_test 进行预测。
catb_predictions = cat_model.predict(X_test)
#使用 CatBoost 模型 (cat_model) 对测试数据集 X_test 进行预测。
# --- 重新计算并打印评估指标 ---
print("\nEvaluating models with optimized hyperparameters on the test set:")
#打印一条消息，表示接下来将对优化后的模型在测试集上进行评估
# --- 决策树 (优化后) ---
dt_mse = mean_squared_error(y_test, dt_predictions)
#计算决策树模型的 均方误差 (MSE)。
dt_mae = mean_absolute_error(y_test, dt_predictions)
#计算决策树模型的 平均绝对误差 (MAE)
dt_r2 = r2_score(y_test, dt_predictions)
#计算决策树模型的 R²分数 (R² Score)
print(f"Optimized Decision Tree MSE: {dt_mse:.4f}, MAE: {dt_mae:.4f}, R2: {dt_r2:.4f}")
#打印决策树模型的评估指标。
# --- 随机森林 (优化后) ---
rf_mse = mean_squared_error(y_test, rf_predictions)
#计算随机森林模型的 均方误差 (MSE)。
rf_mae = mean_absolute_error(y_test, rf_predictions)
#计算随机森林模型的 平均绝对误差 (MAE)。mean_absolute_error 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 rf_predictions 之间的平均绝对误差。
rf_r2 = r2_score(y_test, rf_predictions)
#计算随机森林模型的 R²分数 (R² Score)。r2_score 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 rf_predictions 之间的 R² 分数
print(f"Optimized Random Forest MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}, R2: {rf_r2:.4f}")
#打印随机森林模型的评估指标。使用格式化字符串 :.4f 将 MSE、MAE 和 R² 分数保留四位小数。
# --- XGBoost (优化后) ---
xgb_mse = mean_squared_error(y_test, xgb_predictions)
#计算XGBoost模型的 均方误差 (MSE)。mean_squared_error 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 xgb_predictions 之间的均方误差。
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
#计算XGBoost模型的 平均绝对误差 (MAE)。mean_absolute_error 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 xgb_predictions 之间的平均绝对误差
xgb_r2 = r2_score(y_test, xgb_predictions)
#计算XGBoost模型的 R²分数 (R² Score)。r2_score 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 xgb_predictions 之间的 R² 分数。
print(f"Optimized XGBoost MSE: {xgb_mse:.4f}, MAE: {xgb_mae:.4f}, R2: {xgb_r2:.4f}")
#打印XGBoost模型的评估指标。使用格式化字符串 :.4f 将 MSE、MAE 和 R² 分数保留四位小数。
# --- LightGBM (优化后) ---
lgbm_mse = mean_squared_error(y_test, lgbm_predictions)
#计算LightGBM模型的 均方误差 (MSE)。mean_squared_error 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 lgbm_predictions 之间的均方误差。
lgbm_mae = mean_absolute_error(y_test, lgbm_predictions)
#计算LightGBM模型的 平均绝对误差 (MAE)。mean_absolute_error 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 lgbm_predictions 之间的平均绝对误差。
lgbm_r2 = r2_score(y_test, lgbm_predictions)
#计算LightGBM模型的 R²分数 (R² Score)。r2_score 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 lgbm_predictions 之间的 R² 分数
print(f"Optimized LightGBM MSE: {lgbm_mse:.4f}, MAE: {lgbm_mae:.4f}, R2: {lgbm_r2:.4f}")
#打印LightGBM模型的评估指标。使用格式化字符串 :.4f 将 MSE、MAE 和 R² 分数保留四位小数。
# --- CatBoost (优化后) ---
catb_mse = mean_squared_error(y_test, catb_predictions)
#计算CatBoost模型的 均方误差 (MSE)。mean_squared_error 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 catb_predictions 之间的均方误差。
catb_mae = mean_absolute_error(y_test, catb_predictions)
#计算CatBoost模型的 平均绝对误差 (MAE)。mean_absolute_error 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 catb_predictions 之间的平均绝对误差。
catb_r2 = r2_score(y_test, catb_predictions)
#计算CatBoost模型的 R²分数 (R² Score)。r2_score 是 scikit-learn 提供的一个函数，用于计算真实值 y_test 和预测值 catb_predictions 之间的 R² 分数。
print(f"Optimized CatBoost MSE: {catb_mse:.4f}, MAE: {catb_mae:.4f}, R2: {catb_r2:.4f}")
#打印CatBoost模型的评估指标。使用格式化字符串 :.4f 将 MSE、MAE 和 R² 分数保留四位小数
# --- 将优化后的结果汇总到新字典 ---
opti_model_results = {
    'Decision Tree': {'MSE': dt_mse, 'MAE': dt_mae, 'R2': dt_r2},
    'Random Forest': {'MSE': rf_mse, 'MAE': rf_mae, 'R2': rf_r2},
    'XG Boost': {'MSE': xgb_mse, 'MAE': xgb_mae, 'R2': xgb_r2},
    'LGBM Boost': {'MSE': lgbm_mse, 'MAE': lgbm_mae, 'R2': lgbm_r2},
    'Cat Boost': {'MSE': catb_mse, 'MAE': catb_mae, 'R2': catb_r2},
}
#创建一个名为 opti_model_results 的字典，用于存储各个优化后的模型的评估指标。
#字典的键是模型的名称（如 'Decision Tree'、'Random Forest' 等），值是另一个字典，包含该模型的三个评估指标：
#'MSE'：均方误差（Mean Squared Error）。
#'MAE'：平均绝对误差（Mean Absolute Error）。
#'R2'：R²分数（R² Score）。
#每个模型的评估指标值分别从前面计算得到的变量（如 dt_mse、rf_mae 等）中获取。
print("\nOptimized Model Results:")
print(opti_model_results)
#打印一条消息，表示接下来将显示优化后的模型结果。使用 print 函数打印整个 opti_model_results 字典的内容。
# 假设获取到了Model1和Model2优化后的MSE值
opti_model_results["Model1"] = {"MSE": 0.3}
opti_model_results["Model2"] = {"MSE": 0.25}\
#这两行代码将 Model1 和 Model2 的优化后的MSE值分别添加到 opti_model_results 字典中。
model_names = list(model_results.keys())
#这行代码提取 model_results 字典中的所有键（即模型名称），并将它们存储在一个列表 model_names 中
mse_values = [model_results[model]["MSE"] for model in model_names]
#这行代码使用列表推导式，从 model_results 字典中提取每个模型的MSE值。
opti_mse_values = [opti_model_results[model]["MSE"] for model in model_names]
#这行代码使用列表推导式，从 opti_model_results 字典中提取每个模型的优化后的MSE值
# 后续绘图代码保持不变
# ...
model_names = list(model_results.keys())
#提取 model_results 字典中的所有键（即模型名称），并将它们存储在列表 model_names 中。
mse_values = [model_results[model]["MSE"] for model in model_names]
#使用列表推导式，从 model_results 字典中提取每个模型的MSE值。
# 新增检查逻辑
for model in model_names:
    if model not in opti_model_results:
        print(f"键 {model} 不存在于 opti_model_results 字典中")
opti_mse_values = [opti_model_results[model]["MSE"] for model in model_names]
#遍历 model_names 列表中的每个模型名称。检查每个模型名称是否存在于 opti_model_results 字典中。如果某个模型名称不存在于 opti_model_results 中，则打印一条警告消息，指出该模型名称缺失。这个检查逻辑的作用是确保在提取优化后的MSE值之前，所有模型都有对应的优化结果。
# --- 提取优化前后的 MSE 值 ---
model_names = list(model_results.keys()) 
#使用列表推导式，从 opti_model_results 字典中提取每个模型的优化后的MSE值
mse_values = [model_results[model]['MSE'] for model in model_names] 
opti_mse_values = [opti_model_results[model]['MSE'] for model in model_names]
#这部分代码与前面的代码功能相同，但重复了两次。它再次提取了 model_results 和 opti_model_results 中的模型名称和MSE值。这种重复可能是多余的，除非在代码的其他部分对 model_names、mse_values 或 opti_mse_values 进行了修改
# --- 创建并显示对比条形图 ---
plt.figure(figsize=(12, 7)) 
#使用 matplotlib.pyplot（通常简写为 plt）创建一个新的图形窗口，并设置图形的大小为宽度 12 英寸、高度 7 英寸
bar_width = 0.4
index = np.arange(len(model_names))
#定义条形图的宽度为 0.4。
#使用 numpy.arange 创建一个从 0 到 len(model_names) - 1 的数组，表示每个模型在图中的位置索引
plt.bar(index - bar_width/2, mse_values, bar_width, label='Initial MSE', color='skyblue')
plt.bar(index + bar_width/2, opti_mse_values, bar_width, label='Optimized MSE', color='lightgreen')
#使用 plt.bar 绘制两组条形图：第一组条形图表示原始MSE值（mse_values），位置为 index - bar_width/2，宽度为 bar_width，标签为 'Initial MSE'，颜色为 'skyblue'。第二组条形图表示优化后的MSE值（opti_mse_values），位置为 index + bar_width/2，宽度为 bar_width，标签为 'Optimized MSE'，颜色为 'lightgreen'。通过调整条形图的位置（index - bar_width/2 和 index + bar_width/2），确保两组条形图并排显示，而不是重叠。
plt.xlabel("Model") 
plt.ylabel("Mean Squared Error (MSE) ↓") 
plt.title("Model Performance Comparison: Initial vs. Optimized (MSE)") 
#设置 x 轴标签为 "Model"。设置 y 轴标签为 "Mean Squared Error (MSE) ↓"，其中 ↓ 表示MSE越低越好。设置图形标题为 "Model Performance Comparison: Initial vs. Optimized (MSE)"
plt.xticks(index, model_names, rotation=45, ha="right")
#设置 x 轴的刻度位置为 index，刻度标签为 model_names。将刻度标签旋转 45 度，并设置水平对齐方式为右对齐（ha="right"），以避免标签之间相互重叠。
plt.legend()
#添加图例，显示每组条形图对应的标签（'Initial MSE' 和 'Optimized MSE'）
plt.grid(axis='y', linestyle='--', alpha=0.7) 
#添加 y 轴方向的网格线，使用虚线样式（linestyle='--'），透明度为 0.7（alpha=0.7）
plt.tight_layout() 
#调整图形的布局，确保所有元素（如标题、标签、图例等）都能完整显示，避免相互重叠
plt.show() 
#显示图像
print("\nThe bar chart compares the Mean Squared Error (MSE) for each model before (Initial) and after (Optimized) hyperparameter tuning.")
print("A lower bar indicates better performance. Comparing the height difference for each model shows the impact of tuning.")
#打印一段说明文字，解释条形图的内容：条形图比较了每个模型在超参数优化之前（Initial）和之后（Optimized）的均方误差（MSE）。条形图的高度越低，表示模型性能越好。通过比较每个模型的条形图高度差异，可以直观地看到超参数优化的效果。
# --- 导入 SHAP 和其他必要的库 ---
import shap
import matplotlib.pyplot as plt
import numpy as np
#导入 shap 库，用于解释机器学习模型的输出。
#导入 matplotlib.pyplot（简写为 plt），用于绘图。
#导入 numpy（简写为 np），用于数值计算
# --- 为 XGBoost 模型创建 SHAP 解释器 ---
explainer = shap.Explainer(xgb_model)
#使用 shap.Explainer 创建一个SHAP解释器对象
# --- 计算 SHAP 值 ---
shap_values = explainer(X_test)
#使用创建好的解释器对象 explainer，对测试数据集 X_test 计算SHAP值。
# --- 摘要图 - 显示特征重要性和影响方向 (条形图模式) ---
shap.summary_plot(shap_values, X_test, plot_type="bar")
#使用 shap.summary_plot 函数生成摘要图。shap_values 是之前计算得到的SHAP值。X_test 是测试数据集的特征矩阵，用于提供特征名称和数据范围等信息。参数 plot_type="bar" 指定生成条形图模式的摘要图
# --- 图形调整与保存 ---
plt.tight_layout()
#使用 matplotlib.pyplot.tight_layout() 调整图形的布局，确保所有元素（如标题、标签、图例等）都能完整显示，避免相互重叠
plt.savefig("shap_feature_importance_bar.png", dpi=300, bbox_inches='tight')
#使用 matplotlib.pyplot.savefig 将生成的图形保存为文件。
plt.show()
#显示图形
# --- 摘要点图 - 显示每个特征的SHAP值分布 ---
shap.summary_plot(shap_values, X_test, plot_type="dot")
#使用 shap.summary_plot 函数生成摘要点图。shap_values 是之前计算得到的SHAP值。X_test 是测试数据集的特征矩阵，用于提供特征名称和数据范围等信息。参数 plot_type="dot" 指定生成点图模式的摘要图。
# --- 图形调整与保存 ---
plt.tight_layout()
#使用 matplotlib.pyplot.tight_layout() 调整图形的布局，确保所有元素（如标题、标签、图例等）都能完整显示，避免相互重叠。
plt.savefig("shap_feature_importance_dot.png", dpi=300, bbox_inches='tight')
#使用 matplotlib.pyplot.savefig 将生成的图形保存为文件
plt.show()
#显示图形
# --- 打印特征列名 (辅助理解后续图表) ---
print("Features used in SHAP analysis:", X_test.columns)
#打印出用于SHAP分析的特征列名
# --- 决策图 - 显示样本的预测路径 ---
shap.decision_plot(explainer.expected_value, shap_values.values[:50],
                  feature_names=list(X_test.columns),
                  show=False)
#使用 shap.decision_plot 函数生成决策图。
#参数说明：explainer.expected_value：这是SHAP解释器的期望值，表示模型输出的基线值（即在没有任何特征影响时的预测值）。shap_values.values[:50]：这是计算得到的SHAP值，这里选择了前50个样本的SHAP值。shap_values.values 是一个二维数组，每一行对应一个样本，每一列对应一个特征的SHAP值。
# feature_names=list(X_test.columns)：这是特征名称列表，用于在图中显示特征名称。X_test.columns 是测试数据集的特征列名。show=False：这个参数设置为 False，表示在生成图形时不直接显示，而是等待后续的 plt.show() 调用。调整图形布局
# --- 图形调整与保存 ---
plt.tight_layout()
#使用 matplotlib.pyplot.tight_layout() 调整图形的布局，确保所有元素（如标题、标签、图例等）都能完整显示，避免相互重叠
plt.savefig("shap_decision_plot.png", dpi=300, bbox_inches='tight')
#使用 matplotlib.pyplot.savefig 将生成的图形保存为文件
plt.show()
#显示图形
# --- 瀑布图 - 详细分析多个预测样本 ---
sample_indices = [0, 5, 10] 
#定义一个列表 sample_indices，包含需要生成瀑布图的样本索引。这里选择了索引为 0、5 和 10 的样本。
for i, idx in enumerate(sample_indices):
    exp = shap.Explanation(
        values=shap_values.values[idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[idx].values,
        feature_names=list(X_test.columns)
    )
#使用 shap.Explanation 创建一个SHAP解释对象，用于生成瀑布图。
#参数说明：values=shap_values.values[idx]：指定当前样本的SHAP值。base_values=explainer.expected_value：指定模型的期望值（基线值）。data=X_test.iloc[idx].values：指定当前样本的特征值。feature_names=list(X_test.columns)：指定特征名称列表。#
    plt.figure(figsize=(10, 6))
    #使用 matplotlib.pyplot.figure 设置图形的大小为宽度 10 英寸、高度 6 英寸。
    shap.plots.waterfall(exp, max_display=15, show=False)
    #使用 shap.plots.waterfall 函数生成瀑布图。
    plt.title(f"Waterfall Plot for Sample Index {idx}")
    #使用 matplotlib.pyplot.title 设置图形的标题，显示当前样本的索引
    plt.tight_layout()
    #使用 matplotlib.pyplot.tight_layout 调整图形的布局，确保所有元素（如标题、标签等）都能完整显示，避免相互重叠
    plt.savefig(f"shap_waterfall_sample_{idx}.png", dpi=300, bbox_inches='tight')
    #使用 matplotlib.pyplot.savefig 将生成的图形保存为文件。
    plt.show()
#显示图形
# --- 力图 - 单个样本预测的特征贡献 ---
sample_indices = [7, 15, 25]
#定义一个列表 sample_indices，包含需要生成力图的样本索引。这里选择了索引为 7、15 和 25 的样本。
print("\n--- Force Plot Explanation ---")
print(" - Force plot shows how features push the prediction away from the base value.")
print(" - Red features increase the prediction, Blue features decrease it.")
print("-----------------------------")
#打印一段说明文字，解释力图的作用：力图展示了特征如何将预测值从基线值（平均预测值）推向最终预测值。红色特征表示对预测值有正向贡献（增加预测值），蓝色特征表示负向贡献（减少预测值）

for sample_idx in sample_indices:
    #使用 for 循环遍历 sample_indices 列表中的每个样本索引。
    sample_data_display = X_test.iloc[sample_idx].round(2)
    print(f"\nGenerating Force Plot for Sample Index {sample_idx}...")
    #提取当前样本的特征值，并保留两位小数。
    #打印一条消息，表示正在为当前样本生成力图。


    shap.force_plot(
        explainer.expected_value,
        shap_values.values[sample_idx],
        sample_data_display,
        matplotlib=True,
        feature_names=list(X_test.columns),
        show=False
    )
    #使用 shap.force_plot 函数生成力图。
#参数说明：
explainer.expected_value：模型的期望值（基线值）。
shap_values.values[sample_idx]：当前样本的SHAP值。
sample_data_display：当前样本的特征值。
matplotlib=True：使用 matplotlib 渲染图形。
feature_names=list(X_test.columns)：特征名称列表。
#show=False：不直接显示图形，而是等待后续的 plt.show() 调用
    fig = plt.gcf()
    plt.suptitle(f"Force Plot for Sample Index #{sample_idx}", fontsize=15, y=1.02)
    plt.savefig(f"shap_force_plot_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()
    #获取当前图形对象 fig。
#使用 plt.suptitle 设置图形的总标题。
#使用 plt.savefig 将生成的图形保存为文件，文件名格式为 "shap_force_plot_sample_{sample_idx}.png"。
#使用 plt.show 显示生成的图形。
    predicted_value = explainer.expected_value + np.sum(shap_values.values[sample_idx])
    print(f"--- Sample #{sample_idx} Analysis ---")
    print(f"- Model Predicted Value: {predicted_value:.2f}")
    print(f"- Base Value (Average Prediction): {explainer.expected_value:.2f}")
    #计算当前样本的预测值：基线值加上所有特征的SHAP值之和。
#打印当前样本的预测值和基线值
    feature_names = list(X_test.columns)
    print(f"- Top 3 features increasing prediction: {[feature_names[i] for i in np.argsort(-shap_values.values[sample_idx])[:3]]}")
    print(f"- Top 3 features decreasing prediction: {[feature_names[i] for i in np.argsort(shap_values.values[sample_idx])[:3]]}")
#提取特征名称列表。
#使用 np.argsort 排序SHAP值，找出对预测值增加最多的前3个特征和减少最多的前3个特征。
#打印这些特征的名称
    print("-----------------------------")
#打印分隔线，用于区分不同样本的分析结果。
# --- SHAP值的热图 ---
plt.figure(figsize=(10, 8)) 
#使用 matplotlib.pyplot.figure 设置图形的大小为宽度 10 英寸、高度 8 英寸
shap.plots.heatmap(shap_values, max_display=15, show=False) 
#使用 shap.plots.heatmap 函数生成SHAP值的热图。
plt.title("SHAP Heatmap Plot") 
#使用 matplotlib.pyplot.title 设置图形的标题为 "SHAP Heatmap Plot"
plt.tight_layout() 
#使用 matplotlib.pyplot.tight_layout 调整图形的布局，确保所有元素（如标题、标签、图例等）都能完整显示，避免相互重叠
plt.savefig("shap_heatmap.png", dpi=300, bbox_inches='tight') 
#使用 matplotlib.pyplot.savefig 将生成的图形保存为文件
plt.show()
#显示图形