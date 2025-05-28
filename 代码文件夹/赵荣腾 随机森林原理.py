1. 导入必要的库
import numpy as np # 数据处理与科学计算相关库，numpy 是 Python 里进行科学计算的基础库，它能高效地处理多维数组和各种数学运算。
import pandas as pd # pandas 库有助于数据的处理和分析，它提供了像 DataFrame 这样灵活的数据结构。
import matplotlib.pyplot as plt # matplotlib 是一个基础的绘图库，可以用来创建各种图表。
import seaborn as sns # seaborn 是基于 matplotlib 的高级可视化库，能让图表看起来更加美观。
from sklearn.model_selection import train_test_split, GridSearchCV # train_test_split 能够把数据集划分为训练集和测试集，这对于评估模型性能很关键。GridSearchCV 可用于超参数调优，它会在指定的参数网格中进行搜索，找到最优的参数组合。
from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier 是集成学习中的一种模型，它通过组合多个决策树来提高分类的准确性和鲁棒性。
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay # classification_report 会生成包含精确率、召回率、F1 值等指标的分类报告。confusion_matrix 以矩阵形式展示模型的预测结果，能清晰地看到真正类、假正类、真负类和假负类的情况。
# roc_auc_score 用于计算 ROC 曲线下的面积，该面积越大，说明模型的性能越好。RocCurveDisplay 可以绘制 ROC 曲线，直观地展示模型的性能。
# 这段代码导入了机器学习项目（特别是分类任务）中常用的工具和库，涵盖了数据处理、可视化、模型构建、超参数优化以及模型评估等多个方面。后续使用这些工具可以完成从数据准备到模型部署的一系列工作。

2.创建虚拟数据集
np.random.seed(42) # 这里将 NumPy 的随机数生成器种子设定为 42。这样做的目的是保证每次运行代码时，生成的随机数序列都是一样的，从而让实验结果具有可重复性。
data_size = 10000 # 此代码定义了要生成的样本数量为 10000 条。
data = {
    'age': np.random.randint(18, 70, size=data_size), # age（年龄）：使用 randint 函数生成 18 到 70 之间的整数，以此模拟用户的年龄。
    'income': np.random.randint(20000, 120000, size=data_size), # income（收入）：借助 randint 生成 20000 到 120000 之间的整数，用来表示用户的收入。
    'visits': np.random.poisson(5, size=data_size),# visits（访问次数）：通过 poisson 分布生成，其均值为 5，可用于模拟用户访问网站的次数。
    'time_on_site': np.random.exponential(5, size=data_size),# time_on_site（网站停留时间）：利用 exponential 分布生成，均值是 5，适合模拟用户在网站上的停留时间，该分布具有长尾特性。
    'click_through_rate': np.random.uniform(0, 1, size=data_size),# click_through_rate（点击率）：运用 uniform 分布生成 0 到 1 之间的浮点数，用于表示用户的点击率。
    'previous_purchases': np.random.randint(0, 20, size=data_size),# previous_purchases（过往购买次数）：使用 randint 生成 0 到 19 之间的整数，可模拟用户的历史购买数量。
    'purchase': np.random.choice([0, 1], size=data_size, p=[0.7, 0.3]),# purchase（购买行为）：通过 choice 函数进行二项分布抽样，其中 0 代表未购买，1 代表购买，购买的概率设定为 30%。
}

df = pd.DataFrame(data) # 这行代码把前面生成的字典数据转换为 Pandas 的 DataFrame 格式，方便后续进行数据处理和分析。
# 这个模拟数据集可以用于测试分类算法，比如预测用户是否会购买商品，也能用于特征工程、模型评估等机器学习任务。
3.数据分析与可视化
# 2.1 年龄分布直方图
plt.figure(figsize=(10, 6)) # 创建一个大小为 10×6 英寸的图表，方便后续绘制图形。
sns.histplot(df['age'], kde=True, bins=30, color='blue') # df['age']：指定要绘制直方图的数据源为 DataFrame 中的 age 列。
# kde=True：表示在直方图上同时绘制核密度估计曲线，这样可以更清晰地看到数据的分布形态。bins=30：将数据分成 30 个区间（组），区间数量的多少会影响直方图的外观。color='blue'：将直方图的颜色设置为蓝色。
plt.title('Age Distribution') # 为图表添加标题 “Age Distribution”。
plt.xlabel('Age') # 为 x 轴添加标签 “Age”。
plt.ylabel('Frequency') # 为 y 轴添加标签 “Frequency”。
plt.show() # 将绘制好的直方图显示出来。
# 直观地展示用户年龄的分布情况，了解数据的集中趋势和离散程度。通过核密度估计曲线，可以更平滑地观察年龄分布的概率密度。

# 2.2 特征相关性热图
plt.figure(figsize=(10, 6)) # 创建一个 10×6 英寸的图表。
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f') # df.corr()：计算 DataFrame 中各列之间的相关系数（默认使用皮尔逊相关系数）。annot=True：在热图的每个单元格中显示具体的相关系数值。
# cmap='coolwarm'：使用 “coolwarm” 颜色映射，正数用暖色（红色）表示，负数用冷色（蓝色）表示，便于直观判断相关性的正负。fmt='.2f'：将相关系数值保留两位小数后显示。
plt.title('Feature Correlation Heatmap') # 为热图添加标题 “Feature Correlation Heatmap”。
plt.show() # 显示绘制好的热图。
# 直方图用于展示单个特征（年龄）的分布情况。热图用于展示多个特征之间的相关性，这对于理解数据结构和进行特征工程非常有价值。

# 2.3 目标变量分布
plt.figure(figsize=(8, 5)) # 创建一个大小为 8×5 英寸的图表。
sns.countplot(x='purchase', data=df, palette='Set2') # x='purchase'：以 purchase 列的数据为分类依据进行计数统计。data=df：指定数据源为之前创建的 DataFrame。
# palette='Set2'：使用 Seaborn 的 Set2 颜色方案来区分不同的类别。
plt.title('Target Variable Distribution (Purchase)') # 为图表添加标题，表明这是目标变量的分布情况。
plt.xlabel('Purchase (0=No, 1=Yes)') # 为 x 轴添加标签，说明 0 代表未购买，1 代表购买。
plt.ylabel('Count') # 为 y 轴添加标签，表示数量。
plt.show() # 将绘制好的计数图显示出来。
# 统计并直观呈现购买类（1）和未购买类（0）的样本数量。检查目标变量是否存在类别不平衡的问题。从之前的代码可知，购买类和未购买类的比例是 3:7，这个可视化结果会验证这一比例情况。

# 2.4 停留时间与购买关系
plt.figure(figsize=(10, 6)) # 创建一个 10×6 英寸的图表。
sns.boxplot(x='purchase', y='time_on_site', data=df, palette='Set3') # x='purchase'：按 purchase 列的值（0 和 1）对数据进行分组。y='time_on_site'：分析每组中 time_on_site 列的数据分布。palette='Set3'：使用 Set3 颜色方案。
plt.title('Time on Site vs Purchase') # 为图表添加标题，表明要分析的是停留时间和购买行为之间的关系。
plt.xlabel('Purchase (0=No, 1=Yes)') # 为 x 轴添加标签，说明类别含义。
plt.ylabel('Time on Site') # 为 y 轴添加标签，表示停留时间。
plt.show() # 显示绘制好的箱线图。
#目标变量分布的计数图有助于发现数据集中是否存在类别不平衡问题，这对分类模型的训练很关键。停留时间与购买关系的箱线图可以揭示特征与目标变量之间的潜在关系，辅助进行特征选择和模型构建。

4.模型训练与预测
# 准备数据
X = df.drop('purchase', axis=1) # 从 DataFrame 中移除目标变量 purchase，剩余的列作为特征矩阵 X。
y = df['purchase'] # 将 purchase 列单独提取出来作为目标变量 y。

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 将数据集按 7:3 的比例划分为训练集和测试集，测试集占 30%。random_state=42：设置随机种子，确保每次划分的结果相同，保证实验可重复性。

# 初始化并训练模型
rf = RandomForestClassifier(random_state=42, n_estimators=100) # random_state=42:设置随机种子，保证模型训练的可重复性。n_estimators=100：指定随机森林中决策树的数量为 100 棵。
rf.fit(X_train, y_train) # 使用训练集数据对随机森林模型进行训练。

# 预测
y_pred = rf.predict(X_test) # 对测试集数据进行预测，返回预测的类别标签（0 或 1）。
y_prob = rf.predict_proba(X_test)[:, 1] # 返回测试集样本属于各个类别的概率。 [:, 1]：只取属于类别 1（购买）的概率，用于后续的 AUC 计算和 ROC 曲线绘制。

5.模板性能评估 # 这部分代码主要用于评估随机森林模型的性能，使用了多种评估指标和可视化方法。
print("Classification Report:\n", classification_report(y_test, y_pred)) # 借助classification_report函数，能够生成一份全面的分类评估报告，该报告涵盖了精确率（Precision）、召回率（Recall）、F1 值（F1-Score）以及支持度（Support）等指标。
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) # confusion_matrix函数会输出一个矩阵，该矩阵直观地展示了模型预测结果与真实标签之间的对比情况。
print("ROC AUC Score:", roc_auc_score(y_test, y_prob)) # roc_auc_score函数用于计算模型的受试者工作特征曲线（ROC）下的面积（AUC）。

# 绘制ROC曲线 
  # ROC 曲线是评估分类模型性能的重要工具，它展示了模型在不同分类阈值下的真阳性率 (TPR) 和假阳性率 (FPR) 之间的权衡关系
RocCurveDisplay.from_estimator(rf, X_test, y_test) # from_estimator方法会自动使用模型的预测概率来计算 ROC 曲线
plt.title("ROC Curve for Random Forest")
plt.show()

6.超参数优化 # 随机森林的性能高度依赖于其超参数设置，通过网格搜索可以系统地找到最优参数组合。
param_grid = {
    'n_estimators': [50, 100, 200], #  随机森林中树的数量。通常，数量越多模型越稳定，但计算成本也越高
    'max_depth': [None, 10, 20], # 限制树的生长深度，防止过拟合。None表示不限制深度
    'min_samples_split': [2, 5, 10], # 控制节点分裂的最小样本数，较大的值可以防止过拟合
    'min_samples_leaf': [1, 2, 4] # 确保叶子节点包含足够的样本，提高模型泛化能力
}

grid_search = GridSearchCV( # 这段代码使用GridSearchCV对随机森林模型进行超参数优化，是机器学习工作流中的关键步骤。
    estimator=RandomForestClassifier(random_state=42), # 使用随机森林分类器作为基础模型,random_state=42确保每次运行结果一致，便于复现
    param_grid=param_grid, # 使用之前定义的参数网格（包含n_estimators, max_depth等参数）网格搜索会尝试所有参数组合（本例中有 3×3×3×3=81 种组合）
    scoring='roc_auc', # 使用 ROC 曲线下面积 (AUC) 作为评估指标,适合不平衡数据集，能综合衡量模型在不同阈值下的性能
    cv=3, # 3 折交叉验证：将训练数据分为 3 份，轮流使用 2 份训练，1 份验证,最终结果是 3 次验证分数的平均值，减少过拟合风险
    verbose=2, # 输出详细的搜索进度信息,会显示当前正在尝试的参数组合和验证分数
    n_jobs=-1 # 并行计算，加速搜索过程,-1表示使用所有可用 CPU 核心
)

grid_search.fit(X_train, y_train) # 针对每一组参数，都进行 3 折交叉验证。具体来说，就是将训练数据X_train和y_train分成 3 份，用其中 2 份训练模型，1 份验证模型，如此循环 3 次。
 # 性能评估：每次验证都会计算 AUC 分数，并将 3 次的结果取平均，以此作为该参数组合的最终评分。找到最优参数组合后，会使用全部的训练数据X_train和y_train，基于这些最优参数重新训练模型。
print("Best Parameters:", grid_search.best_params_)

7. 使用最优参数重新训练模型
best_rf = grid_search.best_estimator_ # 返回一个已用最优参数初始化的模型
best_rf.fit(X_train, y_train)

y_best_pred = best_rf.predict(X_test)
y_best_prob = best_rf.predict_proba(X_test)[:, 1] # 获取正类的预测概率（用于计算 AUC）,通过对比优化前后的报告和 AUC 分数，可以直接衡量超参数优化的效果

print("Optimized Classification Report:\n", classification_report(y_test, y_best_pred))
print("Optimized ROC AUC Score:", roc_auc_score(y_test, y_best_prob))
