# --- 提取与产品相关的 One-Hot 编码列 ---
product_columns = [col for col in df_copy.columns if 'Product_' in col]

# --- 计算每个产品的估算“总销售额” ---
product_sales = df_copy[product_columns].sum() * df_copy['Amount'].mean() / len(product_columns)

# --- 绘制产品销售额条形图 ---
plt.figure(figsize=(15, 6))

ax = product_sales.plot(kind='bar', color='skyblue')
ax.set_xticklabels([col_name.split('_')[1] for col_name in product_columns], rotation=45)
#使用product_sales（包含每个产品的估算销售额）绘制条形图，条形图的颜色设置为skyblue。设置x轴的刻度标签。通过col_name.split('_')[1]从列名（如Product_A）中提取产品名称（如A），并将这些名称设置为x轴刻度标签。

plt.xlabel('Product')
plt.ylabel('Total Sales (Estimated)')
plt.title('Estimated Total Sales per Product (Based on Count * Avg Amount)')
plt.xticks(ha='right')
plt.tight_layout()
plt.show()
#设置x轴的标签为'Product'。设置y轴的标签为'Total Sales (Estimated)'。设置图形的标题。设置x轴刻度标签的对齐方式为右对齐（ha='right'）。自动调整布局，确保图形中的元素（如标题、标签等）不会重叠。显示绘制的图形。
#这段代码的核心目的是：提取与产品相关的One-Hot编码列。计算每个产品的估算销售额（基于产品的出现次数和平均交易金额）。使用条形图可视化每个产品的估算销售额。需要注意的是，在计算部分（product_sales的计算公式）可能存在逻辑上的问题，需要根据具体数据和需求进行调整。




plt.figure(figsize=(12, 4)) 
#使用plt.figure函数设置整个图形的大小。参数figsize=(12, 4)表示图形的宽度为12英寸，高度为4英寸。

plt.subplot(1, 1, 1)
#plt.subplot用于创建子图。参数(1, 1, 1)表示：总体布局是1行1列（即只有一个子图）。当前正在绘制的是第1个子图.

plt.plot(df.groupby('Month')['Amount'].sum(), marker='o', linestyle='-', color='green')
plt.xlabel('Month') 
plt.ylabel('Total Sales') 
plt.title('Monthly Sales Trend (Aggregated Across Years)') 
plt.xticks(range(1, 13)) 
plt.grid(axis='y', linestyle='--', alpha=0.7) 
#从DataFrame df中按月份对销售金额进行汇总，计算每个月的销售总额。
#使用折线图展示这些数据，折线图的样式为绿色实线，每个数据点用圆圈标记。
#设置x轴和y轴的标签，分别为“Month”和“Total Sales”，让图表更易于理解。
#添加标题，明确说明图表的内容。
#设置x轴的刻度范围为1到12，对应12个月份。
#在y轴方向添加半透明的虚线网格线，帮助更直观地读取y轴的数值。

plt.tight_layout() 
#使用plt.tight_layout()自动调整图形布局，确保各个元素（如标题、标签、刻度等）之间不会相互重叠。
plt.show()
#使用plt.show()显示最终绘制的图形。




plt.figure(figsize=(8, 6))
#设置图形尺寸为宽度8英寸、高度6英寸。

plt.scatter(df['Boxes Shipped'], df['Amount'], color='purple', alpha=0.6)
#绘制散点图，x轴为发货箱数（Boxes Shipped），y轴为销售额（Sales Amount）。点的颜色为紫色，透明度为0.6，便于看清点的分布情况。

plt.xlabel('Boxes Shipped')
plt.ylabel('Sales Amount')
plt.title('Sales Amount vs. Boxes Shipped')
#设置x轴标签为“Boxes Shipped”，y轴标签为“Sales Amount”，并添加标题“Sales Amount vs. Boxes Shipped”。

z = np.polyfit(df['Boxes Shipped'], df['Amount'], 1)
p = np.poly1d(z)
plt.plot(df['Boxes Shipped'], p(df['Boxes Shipped']), color='red', linestyle='--', label='Trendline')
plt.legend()
#使用np.polyfit对发货箱数和销售额进行一阶多项式拟合（线性回归），计算出线性关系的斜率和截距。利用拟合结果生成趋势线。趋势线用红色虚线表示，添加图例标签“Trendline”。

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
#显示最终图形。





# --- 提取与国家相关的 One-Hot 编码列 ---
country_columns = [col for col in df_copy.columns if 'Country_' in col]

# --- 计算每个国家的估算“总销售额” ---
country_sales = df_copy[country_columns].sum() * df_copy['Amount'].mean() / len(country_columns)

# --- 绘制国家销售贡献条形图 ---
plt.figure(figsize=(10, 6))
#设置图形的宽度为10英寸，高度为6英寸。

ax = country_sales.plot(kind='bar', color='orange')
#使用country_sales数据绘制条形图。设置条形图的颜色为橙色。

ax.set_xticklabels([col_name.split('_')[1] for col_name in country_columns], rotation=45)
#通过列表推导式从列名中提取国家名称（假设列名格式为Country_XX）。将提取的国家名称设置为x轴刻度标签，并旋转45度以防止重叠。

plt.xlabel('Country')
plt.ylabel('Total Sales (Estimated)')
plt.title('Estimated Sales Contribution by Country')
#设置x轴标签为“Country”，y轴标签为“Total Sales (Estimated)”，并添加标题。

plt.xticks(rotation=45, ha='right')
#将x轴刻度标签旋转45度，并设置水平对齐方式为右对齐，以确保标签清晰可读。

plt.tight_layout()
#自动调整图形布局，避免元素重叠。

plt.show()
#显示最终的条形图。




# --- 提取与销售人员相关的 One-Hot 编码列 ---
salesperson_columns = [col for col in df_copy.columns if 'Sales Person_' in col]

# --- 计算每个销售人员的估算“总销售额”/业绩 ---
salesperson_performance = df_copy[salesperson_columns].sum() * df_copy['Amount'].mean() / len(salesperson_columns)

# --- 绘制销售人员业绩条形图 ---
plt.figure(figsize=(12, 6))
#设置图形的宽度为12英寸，高度为6英寸，以便有足够的空间显示条形图。

ax = salesperson_performance.plot(kind='bar', color='green')
#使用salsperson_performance数据绘制条形图。设置条形图的颜色为绿色。

ax.set_xticklabels([col_name.split('_')[1] for col_name in salesperson_columns], rotation=45)
#提取salsperson_columns中列名的销售人员名称部分。将这些名称设置为x轴的刻度标签，并旋转45度以防止标签重叠。

plt.xlabel('Salesperson')
plt.ylabel('Total Sales (Estimated)')
plt.title('Estimated Salesperson Performance')
#设置x轴标签为“Salesperson”，y轴标签为“Total Sales (Estimated)”，并添加标题。

plt.xticks(rotation=45, ha='right')
#再次设置x轴刻度标签的旋转角度为45度，并将水平对齐方式设置为右对齐，以确保标签清晰可读。

plt.tight_layout()
#自动调整图形布局，防止元素重叠。

plt.show()
#显示最终的条形图。




# --- 初始化 HoloViews 的 Bokeh 后端 ---
hv.extension('bokeh')
#初始化 HoloViews，并指定使用 Bokeh 作为后端绘图引擎。Bokeh 是一个交互式可视化库，适合绘制复杂的图形。

# --- 准备 Chord 图的数据 ---
unique_labels = list(set(df['Country']).union(set(df['Sales Person'])))
label_map = {label: i for i, label in enumerate(unique_labels)}
links = [(label_map[row['Country']], label_map[row['Sales Person']], row['Amount']) for _, row in df.iterrows()]
nodes = hv.Dataset(pd.DataFrame({'index': list(label_map.values()), 'label': list(label_map.keys())}), 'index')
#unique_labels：提取数据中所有国家和销售人员的唯一标签。label_map：为每个标签（国家或销售人员）分配一个唯一的索引。links：构建连接数据。每行数据表示从国家到销售人员的连接，并携带销售金额（Amount）作为权重。nodes：创建节点数据集，包含索引和标签信息。

# --- 创建并配置 Chord 图 ---
chord = hv.Chord((links, nodes)).opts(
    opts.Chord(
        labels='label', 
        cmap='Category20', 
        edge_cmap='viridis', 
        edge_color='Amount', 
        node_color='index', 
        node_size=20, 
        width=800, 
        height=800
    )
)
#hv.Chord((links, nodes))：创建 Chord 图，输入连接数据和节点数据。
#opts.Chord(...)：配置 Chord 图的样式：
#labels='label'：使用节点的标签作为显示的标签。
#cmap='Category20'：为节点设置颜色映射。
#edge_cmap='viridis'：为连接线设置颜色映射。
#edge_color='Amount'：根据销售金额为连接线着色。
#node_color='index'：根据索引为节点着色。
#node_size=20：设置节点大小。
#width=800, height=800：设置图形的宽度和高度。

# --- 显示 Chord 图 ---
chord
#显示最终的 Chord 图，展示国家和销售人员之间的销售关系。





# --- 识别需要编码的分类列 (包括 Month) ---
categorical_cols = ['Sales Person', 'Country', 'Product', 'Month']
#创建一个列表 categorical_cols，指定需要进行独热编码的列名。这些列包括销售人员、国家、产品和月份。

# --- 对原始 DataFrame 应用 One-Hot Encoding 并覆盖 ---
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)
#使用 pandas 的 get_dummies 函数对指定的分类列进行独热编码。
#columns=categorical_cols：指定需要编码的列。
#drop_first=True：为了避免多重共线性，去掉每个分类变量的第一个哑变量。
#dtype=float：将生成的哑变量列的数据类型设置为浮点型。
#结果会覆盖原始的 df 数据框，生成新的独热编码后的数据框。

# --- 显示最终处理后的 DataFrame ---
df
#显示处理后的数据框，查看独热编码后的结果。




# --- 定义特征 (X) 和目标 (y) ---
X = df.drop(['Amount', 'Year', 'Date', 'Profit'], axis=1)
#df.drop(...)：从数据框 df 中删除指定的列，返回一个新的数据框。
#['Amount', 'Year', 'Date', 'Profit']：指定要删除的列名列表，这些列通常不作为特征使用。
#'Amount'：可能是目标变量（要预测的值）。
#'Year'、'Date'：时间相关的列，可能在当前模型中不作为特征。
#'Profit'：利润列，可能与目标变量相关，但在此处被排除。
#axis=1：表示沿列方向进行操作。
#最终，X 包含了数据框中除上述列之外的所有列，这些列将作为模型的输入特征。

y = df['Amount']
#df['Amount']：提取数据框中 'Amount' 列的值。
#'Amount' 列通常表示销售金额，是模型要预测的目标变量。
#y 是一个一维数组，包含了所有样本的目标值。



# --- 将数据划分为训练集和测试集 ---
from sklearn.model_selection import train_test_split
#从 sklearn.model_selection 模块中导入 train_test_split 函数，该函数用于将数据集划分为训练集和测试集。


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test：分别表示划分后的训练特征集、测试特征集、训练目标集和测试目标集。X, y：分别表示原始的特征矩阵和目标变量。test_size=0.2：表示测试集占总数据集的20%。random_state=42：用于控制数据划分的随机性，设置固定的随机种子可以保证每次运行代码时数据划分的结果一致，便于复现和调试。

# --- 打印结果数据集的形状以验证划分 ---
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
#使用 print 函数打印出划分后各个数据集的形状（行数和列数），验证数据划分是否正确。
#X_train.shape：训练特征集的形状，行数为总样本数的80%，列数与原始特征矩阵相同。
#X_test.shape：测试特征集的形状，行数为总样本数的20%，列数与原始特征矩阵相同。
#y_train.shape：训练目标集的形状，行数为总样本数的80%。
#y_test.shape：测试目标集的形状，行数为总样本数的20%。



X_train shape: (875, 58)
X_test shape: (219, 58)
y_train shape: (875,)
y_test shape: (219,)
#随机状态对数据划分和模型训练的影响可以总结如下：
#设置随机状态：确保数据划分结果一致，提高实验的可复现性和稳定性。
#不设置随机状态：数据划分完全随机，可能导致每次运行结果不同，增加实验的不确定性。




# --- 初始化模型 ---
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
#从不同的库中导入常用的回归模型：
#DecisionTreeRegressor：决策树回归器，基于单一决策树的模型。
#RandomForestRegressor：随机森林回归器，基于多棵决策树的集成模型。
#XGBRegressor：XGBoost 回归器，一种高效的梯度提升模型。
#LGBMRegressor：LightGBM 回归器，一种轻量级的梯度提升模型。
#CatBoostRegressor：CatBoost 回归器，另一种高效的梯度提升模型。

dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
lgb_model = LGBMRegressor(random_state=42)
catb_model = CatBoostRegressor(random_state=42, verbose=0)
#DecisionTreeRegressor(random_state=42)：初始化一个决策树回归器，设置随机种子为42，以确保模型的可复现性。
#RandomForestRegressor(random_state=42)：初始化一个随机森林回归器，设置随机种子为42。
#XGBRegressor(random_state=42)：初始化一个 XGBoost 回归器，设置随机种子为42。
#LGBMRegressor(random_state=42)：初始化一个 LightGBM 回归器，设置随机种子为42。
#CatBoostRegressor(random_state=42, verbose=0)：初始化一个 CatBoost 回归器，设置随机种子为42。
#参数 verbose=0 用于关闭训练过程中的日志输出，避免过多的输出信息干扰。



# --- 训练决策树模型 ---
start_time = time.time()
dt_model.fit(X_train, y_train)
end_time = time.time()
print(f"Decision Tree training time: {end_time - start_time:.2f} seconds")
#start_time = time.time()：记录训练开始时的时间戳。
#dt_model.fit(X_train, y_train)：使用训练集数据 X_train 和目标变量 y_train 训练决策树模型。
#end_time = time.time()：记录训练结束时的时间戳。
#print(...)：计算并打印训练决策树模型所花费的时间，精确到小数点后两位。

# --- 训练随机森林模型 ---
start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time()
print(f"Random Forest training time: {end_time - start_time:.2f} seconds")
#同样的逻辑，记录随机森林模型的训练时间。
#随机森林通常比单一决策树更复杂，因为它是基于多棵决策树的集成模型，因此其训练时间可能会更长。

# --- 训练 XGBoost 模型 ---
start_time = time.time()
xgb_model.fit(X_train, y_train)
end_time = time.time()
print(f"XGBoost training time: {end_time - start_time:.2f} seconds")
#记录 XGBoost 模型的训练时间。
#XGBoost 是一种高效的梯度提升模型，通常在训练过程中会利用多线程或其他优化技术，因此训练速度可能相对较快。

# --- 训练 LightGBM 模型 ---
start_time = time.time()
lgb_model.fit(X_train, y_train)
end_time = time.time()
print(f"LightGBM training time: {end_time - start_time:.2f} seconds")
#记录 LightGBM 模型的训练时间。
#LightGBM 是另一种高效的梯度提升模型，特别优化了大规模数据集的训练速度和内存使用效率，因此通常训练速度较快。

# --- 训练 CatBoost 模型 ---
start_time = time.time()
catb_model.fit(X_train, y_train)
end_time = time.time()
print(f"CatBoost training time: {end_time - start_time:.2f} seconds")
#记录 CatBoost 模型的训练时间。
#CatBoost 也是一种梯度提升模型，但在处理分类特征时具有独特的优势。其训练时间可能因数据集的大小和复杂性而有所不同。

print("Models trained successfully.")
#在所有模型训练完成后，打印一条消息，确认模型训练过程已成功完成。





# --- 训练决策树模型 ---
start_time = time.time()
dt_model.fit(X_train, y_train)
end_time = time.time()
print(f"Decision Tree training time: {end_time - start_time:.2f} seconds")
#记录训练决策树模型所需的时间。决策树模型通常训练速度较快，但可能容易过拟合。

# --- 训练随机森林模型 ---
start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time()
print(f"Random Forest training time: {end_time - start_time:.2f} seconds")
#记录训练随机森林模型所需的时间。
#随机森林由多棵决策树组成，通常比单棵决策树表现更好，但训练时间更长。

# --- 训练 XGBoost 模型 ---
start_time = time.time()
xgb_model.fit(X_train, y_train)
end_time = time.time()
print(f"XGBoost training time: {end_time - start_time:.2f} seconds")
#记录训练 XGBoost 模型所需的时间。
#XGBoost 是一种高效的梯度提升模型，通常训练速度较快，性能优异。

# --- 训练 LightGBM 模型 ---
start_time = time.time()
lgb_model.fit(X_train, y_train)
end_time = time.time()
print(f"LightGBM training time: {end_time - start_time:.2f} seconds")
#记录训练 LightGBM 模型所需的时间。
#LightGBM 是一种优化的梯度提升模型，通常在大规模数据集上训练速度更快，内存占用更少

# --- 训练 CatBoost 模型 ---
start_time = time.time()
catb_model.fit(X_train, y_train)
end_time = time.time()
print(f"CatBoost training time: {end_time - start_time:.2f} seconds")
#记录训练 CatBoost 模型所需的时间。
#CatBoost 是一种高效的梯度提升模型，对分类特征的处理有独特优势，但训练时间可能因数据复杂性而有所不同。

print("Models trained successfully.")
#在所有模型训练完成后，确认训练过程成功完成。

#这段代码通过逐一训练五种模型并记录训练时间，帮助评估不同模型在相同数据集上的训练效率。这种比较对于选择适合特定任务的模型非常有帮助，尤其是在处理大规模数据集时，训练时间是一个重要的考量因素。