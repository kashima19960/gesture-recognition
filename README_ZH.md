# 基于长短期记忆神经网络的手势动作识别

## 数据集说明

项目的数据集来源于[HAND GESTURES RECORDED WITH MM-WAVE FMCW RADAR (AWR1642)](https://ieee-dataport.org/open-access/hand-gestures-recorded-mm-wave-fmcw-radar-awr1642)

![](./assets/clip_image001.png)

数据集包含4600个样本，涵盖12种不同的手部运动手势。这些数据由四位不同的人使用FMCW AWR1642雷达采集。每个样本均以CSV文件形式保存，并与相应的手势类型关联。手势描述如下：(G1) 左臂挥动——手臂从右至左的完整挥动，(G2) 右臂挥动——手臂从左至右的完整挥动，(G3) 手远离——手从雷达前移开，(G4) 手靠近——手向雷达靠近，(G5) 手臂上举——手臂从下至上的运动，(G6) 手臂下放——手臂从上至下的运动，(G7) 手掌朝上——手掌向上旋转，(G8) 手掌朝下——手掌向下旋转，(G9) 手向左——手向左移动（不伴随手臂动作），(G10) 手向右——手向右移动，(G11) 水平握拳，(G12) 垂直握拳。

## 项目目录结构

```
.
├── DataProcessed.py 包含数据预处理相关的类和方法，负责读取和预处理数据集
├── Neural_Networks.py 包含神经网络模型定义、训练、评估和可视化方法，用于分类任务。
├── main.py 主程序，用于加载模型并评估其在测试数据上的准确率。
└── requirements.txt 项目依赖的Python库列表。
└── assets 存储生成的图像   
└── model.pth 训练好的模型文件
└── ConstantDefinition.py 存放网络训练的参数，通过调参能影响网络的训练效果
```

## 使用说明

### 依赖项版本

- `python3.10`
- `matplotlib==3.9.2`
- `numpy==2.1.3`
- `torch==2.5.1+cu124`
  安装不同版本的库，会出现相关依赖报错的问题，因此最好创建一个虚拟环境来下载依赖项。

### 安装依赖

在项目根目录下运行以下命令安装所需的Python库：

```bash
pip install -r requirements.txt
```

> 如果你的电脑没有安装CUDA，请将requirements.txt中的torch改为torch==2.5.1,改为使用CPU进行训练

### 运行项目

model.pth是作者已经训练好的模型，直接运行以下的命令即可查看模型的准确率

```bash
python main.py
```

### 结果输出

- 模型在测试数据上的准确率将打印在控制台。
- 损失和准确率的变化图将保存在 `assets/`目录下，分别为 `loss.png`和 `accuracy.png`。
- 混淆矩阵图将保存在 `assets/`目录下，文件名为 `title.png`。

## 参考资料

- 更多关于数据预处理的详细信息，请参考 `DataProcessed.py`文件。
- 更多关于神经网络模型定义和训练的详细信息，请参考 `Neural_Networks.py`文件。
- 更多关于主程序的详细信息，请参考 `main.py`文件。
