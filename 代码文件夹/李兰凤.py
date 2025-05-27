# -*- coding: utf-8 -*-  # 声明文件编码为UTF-8，确保中文字符和注释正常处理
import time  # 导入时间库，可用于测量代码执行时间或处理时间相关操作

# --- 计算与数据处理库 ---
import numpy as np  # 提供强大的N维数组对象和相关数学函数
import pandas as pd  # 提供DataFrame和Series等数据结构，用于高效的数据处理、清洗和分析

# --- 数据可视化库 ---
import seaborn as sns  # 基于Matplotlib，提供更美观、更高级的统计图形绘制功能
import matplotlib.pyplot as plt  # 最常用的绘图库，用于创建各种图表
import holoviews as hv  # 用于创建声明式、可组合、交互式的数据可视化
from holoviews import opts  # 用于配置和定制HoloViews图表的视觉选项

# --- 机器学习辅助与预处理库 ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 回归模型评估指标
from sklearn.model_selection import train_test_split, cross_val_score  # 数据划分和交叉验证

# --- 机器学习模型库 ---
from xgboost import XGBRegressor  # 高效的梯度提升回归模型
from lightgbm import LGBMRegressor  # 速度更快、内存占用更低的梯度提升回归模型
from catboost import CatBoostRegressor  # 能很好处理类别特征的梯度提升回归模型
from sklearn.tree import DecisionTreeRegressor  # 决策树回归器模型
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归器模型，决策树的集成模型

# --- 超参数优化库 ---
import optuna  # 用于自动化超参数搜索的框架

#from sklearn.model_selection：指定了cross_val_score函数所在的模块。
#sklearn.model_selection是Scikit-learn库中的一个模块，包含用于模型选择和评估的工具，如数据划分、交叉验证等。
#import cross_val_score：表示从sklearn.model_selection模块中导入cross_val_score函数。
from sklearn.model_selection import cross_val_score

# -*- coding: utf-8 -*-  # 声明文件编码为UTF-8，确保中文字符和注释正常处理
import time  # 导入时间库，可用于测量代码执行时间或处理时间相关操作

# --- 计算与数据处理库 ---
import numpy as np  # 提供强大的N维数组对象和相关数学函数
import pandas as pd  # 提供DataFrame和Series等数据结构，用于高效的数据处理、清洗和分析

# --- 数据可视化库 ---
import seaborn as sns  # 基于Matplotlib，提供更美观、更高级的统计图形绘制功能
import matplotlib.pyplot as plt  # 最常用的绘图库，用于创建各种图表
import holoviews as hv  # 用于创建声明式、可组合、交互式的数据可视化
from holoviews import opts  # 用于配置和定制HoloViews图表的视觉选项

# --- 机器学习辅助与预处理库 ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 回归模型评估指标
from sklearn.model_selection import train_test_split, cross_val_score  # 数据划分和交叉验证

# --- 机器学习模型库 ---
from xgboost import XGBRegressor  # 高效的梯度提升回归模型
from lightgbm import LGBMRegressor  # 速度更快、内存占用更低的梯度提升回归模型
from catboost import CatBoostRegressor  # 能很好处理类别特征的梯度提升回归模型
from sklearn.tree import DecisionTreeRegressor  # 决策树回归器模型
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归器模型，决策树的集成模型

# --- 超参数优化库 ---
import optuna  # 用于自动化超参数搜索的框架
from sklearn.model_selection import cross_val_score#这行代码的作用是从sklearn库的model_selection模块中导入一个函数，名字叫cross_val_score。这个函数是用来做交叉验证的，它可以帮助我们评估一个模型在不同数据子集上的性能表现，从而使我们能够更准确地衡量模型的好坏。
import pandas as pd

# 假设 df 是已有的 DataFrame

# 清洗 'Amount' 列
df['Amount'] = df['Amount'].str.replace(r'[$,]', '', regex=True).astype(float)

# 转换 'Date' 列为日期时间对象
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# 从 'Date' 列创建新特征
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
# --- 显示 DataFrame 的形状 ---
print("\nShape of the DataFrame:", df.shape)

# --- 显示数据类型信息 ---
print("\nData types of each column:\n")
df.info()

# --- 显示 DataFrame 的内容（通常在交互式环境如 Jupyter Notebook 中自动渲染）---
# 在脚本中使用 print 函数显示数据框内容
print("\nDataFrame content:")
print(df.to_csv(sep='\t', na_rep='nan'))  # 使用制表符分隔，NaN表示缺失值

# --- 生成描述性统计信息 ---
print("\nDescriptive statistics:\n", df.describe(include='all').to_csv(sep='\t', na_rep='nan'))

import matplotlib.pyplot as plt

# --- 分析关键变量的分布 ---
print("\nDistribution of 'Amount' and 'Boxes Shipped':")

# --- 创建图形和子图 ---
plt.figure(figsize=(12, 5))

# --- 绘制 'Amount' 的直方图 ---
plt.subplot(1, 2, 1)#创建一个子图布局，并选择第一个子图。

plt.hist(df['Amount'], bins=20, edgecolor="black")#绘制一个直方图。
plt.xlabel('Amount')#设置x轴的标签。
plt.ylabel('Frequency')#设置y轴的标签。
plt.title('Distribution of Amount')#设置当前子图的标题。

# --- 绘制 'Boxes Shipped' 的直方图 ---
plt.subplot(1, 2, 2)#创建一个子图布局，参数含义与前文相同，这里是在1行2列的布局中选择第2个位置来绘制图表。
plt.hist(df['Boxes Shipped'], bins=20, edgecolor="black")#绘制直方图。
plt.xlabel('Boxes Shipped')#设置x轴的标签为“Boxes Shipped”。
plt.ylabel('Frequency')#设置y轴的标签为“Frequency”。
plt.title('Distribution of Boxes Shipped')#设置图表标题为“Distribution of Boxes Shipped”。

# --- 调整布局并显示图形 ---
plt.tight_layout()#自动调整子图之间的间距和布局。
plt.show()#显示绘制的图表。
import matplotlib.pyplot as plt#导入matplotlib库中的pyplot模块，并给它一个别名plt。
import seaborn as sns#导入seaborn库，并给它一个别名sns。

# --- 计算并可视化相关性矩阵 ---
correlation_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", correlation_matrix)

# --- 绘制相关性热力图 ---
plt.figure(figsize=(8, 6))#指定图形的宽度为8英寸，高度为6英寸。这可以使生成的图表在显示时更加清晰，避免因默认尺寸过小导致的拥挤。
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")#使用seaborn的heatmap函数绘制热力图。
plt.title('Correlation Matrix of Numerical Features')#设置热力图的标题。
plt.show()#设置热力图的标题。

def winsorize_outliers(series, limits=(0.05, 0.95)):
    """
    对输入的Series进行Winsorization处理，限制极端值
    
    参数:
    series (pd.Series): 需要处理的Series
    limits (tuple): 上下限分位数阈值，默认为(0.05, 0.95)
    
    返回:
    pd.Series: 处理后的Series
    """
    lower_limit = series.quantile(limits[0])
    upper_limit = series.quantile(limits[1])
    winsorized_series = series.clip(lower_limit, upper_limit)
    return winsorized_series

# --- 对指定列应用 Winsorization ---
for col in ['Amount', 'Boxes Shipped']:
    df[col] = winsorize_outliers(df[col])
    print(f"Winsorized outliers in '{col}'.")

# --- 显示处理后的描述性统计 ---
print("\n描述性统计 (处理后):\n", df.describe())

# --- 定义 Winsorization 函数 ---
# 定义一个名为 winsorize_outliers 的函数，它接受两个参数：
# - series: 一个 Pandas Series 对象（即 DataFrame 的一列）
# - limits: 一个元组，包含两个介于 0 和 1 之间的小数，分别代表下限和上限的分位数阈值。默认值为 (0.05, 0.95)，即处理低于 5% 分位数和高于 95% 分位数的值。
def winsorize_outliers(series, limits=(0.05, 0.95)):
    # series.quantile(limits[0]) 计算输入 Series 的下限分位数（例如 5% 分位数）
    lower_limit = series.quantile(limits[0])
    # series.quantile(limits[1]) 计算输入 Series 的上限分位数（例如 95% 分位数）
    upper_limit = series.quantile(limits[1])
    # series.clip(lower, upper) 方法将 Series 中所有小于 lower_limit 的值替换为 lower_limit，
    # 所有大于 upper_limit 的值替换为 upper_limit，介于两者之间的值保持不变。
    winsorized_series = series.clip(lower_limit, upper_limit)
    # 返回经过 Winsorization 处理后的新 Series
    return winsorized_series

# --- 对指定列应用 Winsorization ---
# 使用 for 循环遍历一个包含需要处理的列名的列表 ['Amount', 'Boxes Shipped']
for col in ['Amount', 'Boxes Shipped']:
    # 对 DataFrame df 中的当前列 col 调用 winsorize_outliers 函数进行处理
    # 将返回的处理后的 Series 重新赋值给 df[col]，覆盖原始列
    df[col] = winsorize_outliers(df[col])
    # 使用 f-string 打印一条信息，告知用户哪一列的异常值已被 Winsorized 处理
    print(f"Winsorized outliers in '{col}'.")

# --- 显示处理后的描述性统计 ---
# 再次调用 df.describe() 来查看 Winsorization 处理后数据的统计摘要
# 对比处理前的 describe() 结果，可以观察到 min 和 max 值通常会变化，更接近 5% 和 95% 分位数的值，标准差 std 可能也会减小。
print(df.describe())

# --- 计算 'Profit' 列 ---
df['Profit'] = df['Amount'] * 0.20

# --- 显示更新后的 DataFrame (前几行) ---
print(df.head())
 
# --- 销售额分布统计 ---
sales_stats = df['Amount'].describe()
print(sales_stats) 

# --- 识别畅销和滞销产品 ---
product_sales = df.groupby('Product')['Amount'].sum()
print("\nProduct Sales:\n", product_sales) 

# --- 分析随时间变化的销售趋势 ---
# df.groupby('Year') 按 'Year' 列分组
# ['Amount'].sum() 计算每个年份的总销售额
yearly_sales = df.groupby('Year')['Amount'].sum()
print("\nYearly Sales:\n", yearly_sales) # 打印按年汇总的销售额

# df.groupby('Month') 按 'Month' 列分组
# ['Amount'].sum() 计算每个月份的总销售额（跨所有年份）
monthly_sales = df.groupby('Month')['Amount'].sum()
print("\nMonthly Sales:\n", monthly_sales) # 打印按月汇总的销售额（注意这可能混合了不同年份的月份数据，如需看特定年份的月度趋势，需先过滤年份或进行多级分组）

# --- 销售额与发货箱数的关系 ---
# df['Amount'].corr(df['Boxes Shipped']) 计算 'Amount' 列与 'Boxes Shipped' 列之间的皮尔逊相关系数。
# 该系数衡量两个变量线性关系的强度和方向，范围从 -1 (完全负相关) 到 +1 (完全正相关)，0 表示无线性相关。
correlation = df['Amount'].corr(df['Boxes Shipped'])
print(f"\nCorrelation between Sales Amount and Boxes Shipped:{correlation:.2f}")# 使用 f-string 打印相关系数值，并格式化为保留两位小数

# --- 各国的销售贡献 ---
country_sales = df.groupby('Country')['Amount'].sum()
print("\nCountry Sales:\n", country_sales)

# --- 销售人员业绩 ---
salesperson_sales = df.groupby('Sales Person')['Amount'].sum()
print("\nSalesperson Performance:\n", salesperson_sales)

# --- 组合类别分析 - 示例：产品在各国的表现 ---
product_country_sales = df.groupby(['Product', 'Country'])['Amount'].sum().unstack()
print("\nProduct Performance by Country:\n", product_country_sales)

plt.figure(figsize=(12, 5))

# --- 绘制直方图 ---
plt.subplot(1, 2, 1)
plt.hist(df['Amount'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Sales Amount')

# --- 绘制箱线图 ---
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Amount'], color='lightcoral')
plt.ylabel('Sales Amount')
plt.title('Box Plot of Sales Amount')

plt.tight_layout()
plt.show()
# --- 识别需要编码的分类列 ---
categorical_cols = ['Sales Person', 'Country', 'Product']

# --- 对 DataFrame 副本应用 One-Hot Encoding ---
df_copy = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- 显示原始 DataFrame 的前几行 ---
print(df.head())
  