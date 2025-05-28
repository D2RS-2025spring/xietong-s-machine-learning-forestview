导入和配置环境
import pandas as pd # 用于数据处理和分析，提供了DataFrame和Series等数据结构，适合处理表格数据。
import numpy as np # 提供高性能的多维数组和数学函数，是科学计算的基础库。
import matplotlib.pyplot as plt # matplotlib.pyplot：Python 最常用的绘图库，用于创建各种静态图表（如折线图、散点图等）。
from sklearn.model_selection import train_test_split # 将数据集划分为训练集和测试集，用于模型训练和评估。
from sklearn.preprocessing import MinMaxScaler # 数据预处理工具，将特征缩放到指定范围（通常是[0,1]）。
from sklearn.ensemble import RandomForestRegressor # 随机森林回归模型，集成多个决策树进行回归预测。
from sklearn.metrics import r2_score, mean_absolute_error # 评估回归模型的指标（R² 分数和平均绝对误差）。
# 设置图例正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定中文字体为黑体（SimHei），确保图表中的中文文本正常显示。
# 设置黑体显示中文
plt.rcParams['axes.unicode_minus'] = False 
# 解决负号"-"显示方块问题# 清空环境（Python中不需要显式清空，新的运行环境自动清空）
导入数据
res = pd.read_excel('D:\\文件\\Long-Term Deflection of Reinforced Concrete Beams_New.xlsx') # 这行代码借助 pandas 库的read_excel函数，成功读取了指定路径下的 Excel 文件，将其存储为 DataFrame 格式。
Num, Dim = res.shape # 它能获取数据集的基本信息。Num代表数据集的样本数量，Dim则表示特征维度，也就是列数。
划分训练集和测试集
X = res.iloc[:, :Dim -1].values # 将除最后一列外的所有列数据提取出来，作为特征矩阵X。
y = res.iloc[:, Dim -1].values # 把最后一列数据提取出来，作为目标变量y。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100) # 使用train_test_split函数按 8:2 的比例划分训练集和测试集，random_state=100保证了每次划分的结果是一样的，具有可重复性。

数据归一化
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
# 创建了两个MinMaxScaler对象，scaler_X用于对特征矩阵进行归一化，scaler_y用于对目标变量进行归一化，归一化的范围都是 0 到 1。

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
# 对于特征矩阵，先在训练集上使用fit_transform方法，该方法会计算训练集的统计信息并进行归一化，然后在测试集上使用transform方法，利用训练集的统计信息对测试集进行归一化，这样可以避免数据泄露。

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
# 由于目标变量y是一维数组，而MinMaxScaler需要二维数组作为输入，所以使用reshape(-1, 1)将其转换为二维数组，归一化后再使用flatten()将其转换回一维数组。

训练模型
trees = 100 # 模型会构建 100 棵决策树，通常树的数量越多，模型性能越好，但计算成本也会相应增加。
leaf = 5 # 设定每个叶子节点最少需要包含 5 个样本，这样可以防止模型过度拟合。
net = RandomForestRegressor(n_estimators=trees, min_samples_leaf=leaf, oob_score=True) # 启用袋外误差（Out-of-Bag Error）评估，这是随机森林特有的一种交叉验证方法。
net.fit(X_train_scaled, y_train_scaled) # 使用归一化后的训练数据X_train_scaled和y_train_scaled对随机森林回归模型进行训练。
importance = net.feature_importances_ # 能够计算出各个特征在模型中的重要程度，帮助我们了解哪些因素对预测结果的影响更大。
仿真测试与反归一化
# 仿真测试
y_sim1_scaled = net.predict(X_train_scaled)
y_sim2_scaled = net.predict(X_test_scaled)
# y_sim1_scaled是模型对训练集的预测结果，y_sim2_scaled是模型对测试集的预测结果，这两个结果都是归一化后的数值。

# 数据反归一化
y_sim1 = scaler_y.inverse_transform(y_sim1_scaled.reshape(-1, 1)).flatten()
y_sim2 = scaler_y.inverse_transform(y_sim2_scaled.reshape(-1, 1)).flatten()
# 由于预测结果是经过归一化的，所以需要使用之前训练好的scaler_y进行反归一化操作，将预测结果还原到原始数据的尺度。inverse_transform方法要求输入的是二维数组，所以先使用reshape(-1, 1)进行转换，之后再用flatten()将结果变回一维数组。
均方根误差
M = len(y_train) # 反映了模型对训练数据的拟合效果。
N = len(y_test) # 它体现了模型的泛化能力。
error1 = np.sqrt(np.sum((y_sim1 - y_train) ** 2) / M)
error2 = np.sqrt(np.sum((y_sim2 - y_test) ** 2) / N)

绘图：预测结果对比
plt.figure()
plt.plot(range(1, M + 1), y_train,'r-*', range(1, M + 1), y_sim1,'b-o', linewidth=1)
plt.legend([' 真实值','预测值'])
plt.xlabel(' 预测样本')
plt.ylabel(' 预测结果')
plt.title(f' 训练集预测结果对比\nRMSE={error1:.4f}')
plt.xlim([1, M])
plt.grid()
# 绘制训练集的真实值（红色星号）与预测值（蓝色圆点）的对比折线图。横轴为样本索引（1 到 M），纵轴为预测结果。标题显示训练集的 RMSE（保留 4 位小数）。

plt.figure()
plt.plot(range(1, N + 1), y_test,'r-*', range(1, N + 1), y_sim2,'b-o', linewidth=1)
plt.legend([' 真实值','预测值'])
plt.xlabel(' 预测样本')
plt.ylabel(' 预测结果')
plt.title(f' 测试集预测结果对比\nRMSE={error2:.4f}')
plt.xlim([1, N])
plt.grid()
# 与训练集图类似，但展示测试集的真实值与预测值对比。横轴为样本索引（1 到 N），纵轴为预测结果。标题显示测试集的 RMSE。
绘制误差曲线
oob_errors = 1 - net.oob_score_
plt.figure()
plt.plot(range(1, trees + 1), [oob_errors] * trees,'b-', linewidth=1)
plt.legend([' 误差曲线'])
plt.xlabel(' 决策树数目')
plt.ylabel(' 误差')
plt.xlim([1, trees])
plt.grid()
# 绘制随机森林的 OOB（袋外）误差随决策树数量的变化曲线。
绘制特征重要性
plt.figure()
plt.bar(range(len(importance)), importance)
plt.legend([' 重要性'])
plt.xlabel(' 特征')
plt.ylabel(' 重要性')
# 绘制随机森林中各特征的重要性得分
相关指标计算
#R1和R2  
R1 = r2_score(y_train, y_sim1)
R2 = r2_score(y_test, y_sim2)

print(f'训练集数据的R2为：{R1:.4f}')
print(f'测试集数据的R2为：{R2:.4f}')
# 它表示模型对数据的拟合程度，其取值范围在 (-∞, 1] 之间。R² 越接近 1，说明模型的预测值与真实值越接近，模型的拟合效果越好；若 R² 为负值，则意味着模型的预测效果比简单取平均值还要差。

# MAE
mae1 = mean_absolute_error(y_train, y_sim1)
mae2 = mean_absolute_error(y_test, y_sim2)

print(f'训练集数据的MAE为：{mae1:.4f}')
print(f'测试集数据的MAE为：{mae2:.4f}')
# 它是预测值与真实值之间绝对误差的平均值，能够直观地反映模型预测的平均偏差程度，且不受异常值的强烈影响。

# MBE
mbe1 = np.sum(y_sim1 - y_train) / M
mbe2 = np.sum(y_sim2 - y_test) / N

print(f'训练集数据的MBE为：{mbe1:.4f}')
print(f'测试集数据的MBE为：{mbe2:.4f}')
# 它是预测值与真实值之间偏差的平均值，能够反映模型预测的系统性偏差。
绘制散点图
sz = 25 # 散点大小
c ='b' # 散点颜色（蓝色）

plt.figure()
plt.scatter(y_train, y_sim1, sz, c) # 绘制训练集真实值与预测值的散点
plt.plot([np.min(y_train), np.max(y_train)], [np.min(y_sim1), np.max(y_sim1)],'--k') # 绘制对角线
plt.xlabel(' 训练集真实值')
plt.ylabel(' 训练集预测值')
plt.xlim([np.min(y_train), np.max(y_train)]) # 设置x轴范围
plt.ylim([np.min(y_sim1), np.max(y_sim1)]) # 设置y轴范围
plt.title(' 训练集预测值 vs. 训练集真实值')

plt.figure()
plt.scatter(y_test, y_sim2, sz, c) # 绘制测试集真实值与预测值的散点
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_sim2), np.max(y_sim2)],'--k') # 绘制对角线
plt.xlabel(' 测试集真实值')
plt.ylabel(' 测试集预测值')
plt.xlim([np.min(y_test), np.max(y_test)]) # 设置x轴范围
plt.ylim([np.min(y_sim2), np.max(y_sim2)]) # 设置y轴范围
plt.title(' 测试集预测值 vs. 测试集真实值')

plt.show()