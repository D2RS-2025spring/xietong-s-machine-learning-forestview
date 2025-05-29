# 机器学习森林可视化项目

## 项目概述
这是一个基于随机森林算法的端到端机器学习流程实现，集成了数据预处理、模型训练、评估与结果可视化功能。项目使用Python编写，通过清晰的模块化设计提供开箱即用的机器学习解决方案。

## 环境要求
- Python 3.7+（推荐Python 3.8）
- 依赖库：Pandas, Scikit-learn, Matplotlib, Seaborn

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/simplelity/xietong-s-machine-learning-forestview.git
cd xietong-s-machine-learning-forestview
```

### 2. 论文数据

本实验数据采用的是从相关公众号获取的论文数据

### 3.项目结构

```
├── data/               # 数据集目录
│   ├── train.csv       # 训练数据
│   └── test.csv        # 测试数据
│
├── src/                # 源代码
│   ├── load_data.py    # 数据加载模块
│   ├── preprocess.py   # 数据预处理模块
│   ├── train.py        # 模型训练模块
│   └── plot.py         # 可视化模块
│
├── results/            # 输出结果
│   ├── predictions.csv       # 预测结果
│   ├── confusion_matrix.png  # 混淆矩阵
│   └── feature_importance.png # 特征重要性图
│
├── main.py             # 主程序
├── requirements.txt    # 依赖列表
└── README.md           # 项目文档
```

### 4.核心功能

完整机器学习流程：数据加载 → 预处理 → 训练 → 评估 → 可视化

(1)数据预处理：
(2)自动处理缺失值
(3)分类特征编码
(4)随机森林模型：
(5)分类任务支持
(6)特征重要性分析
(7)可视化输出：
(8)混淆矩阵
(9)特征重要性排序图
(10)预测结果CSV文件

### 5.项目总结

(1)优势：
- 模块化设计：清晰分离数据加载、预处理、训练和可视化模块
- 开箱即用：提供完整示例数据和依赖列表
- 可视化导向：自动生成关键模型指标图表
- 代码可读性：结构清晰，注释完善，适合学习
(2)当前限制
- 仅支持CSV格式数据输入
- 预处理流程固定，缺乏灵活性
- 仅实现随机森林单一模型

### 6.项目说明

（1）相关代码解释在代码文件夹
（2）原理解释部分是实例的基础
（3）其他文件主要是对项目中产生的图片进行保存
