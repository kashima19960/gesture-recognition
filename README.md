https://blog.51cto.com/u_12413309/6243649

https://ieee-dataport.org/open-access/hand-gestures-recorded-mm-wave-fmcw-radar-awr1642

HAND GESTURES RECORDED WITH MM-WAVE FMCW RADAR (AWR1642)

[![https://ieee-dataport.org/sites/default/files/styles/large/public/IEEE_image2.png?itok=DhD7c62M](./assets/clip_image001.png)](https://ieee-dataport.org/sites/default/files/IEEE_image2.png)

Citation Author(s):

The dataset contains 4600 samples of 12 different hand-movement gestures. Data were collected from four different people using the FMCW AWR1642 radar. Each sample is saved as a CSV file associated with its gesture type. Gesture description: (G1) arm to left – full swipe of an arm from right to left, (G2) arm to right – full swipe of an arm from left to right, (G3) hand away – taking a hand away from radar, (G4) hand closer – taking a hand closer to the radar, (G5) arm up – an arm movement from bottom to top, (G6) arm down – an arm movement from top to bottom, (G7) palm up – rotating a palm upwards, (G8) palm down – rotating a palm downwards, (G9) hand to the left – a hand movement to the left (without an arm movement), (G10) hand to the right – a hand movement to the right, (G11) closing a fist horizontally, (G12) closing a fist vertically.

**INSTRUCTIONS:**

Dataset is divided into 12 separate folders associated to different gesture types. Each folder contains gesture samples saved as a CSV file. First line of the CSV file is a headline describing the columns of data: FrameNumber, ObjectNumber, Range, Velocity, PeakValue, x coordinate, y coordinate. In order to read the gestures into matrix representation copy all 12 folders into single folder called “data”. Copy the “read_gesture.py” script to the same folder as “data” and run it. Script will convert CSV files of given gesture type into the numpy matrix.

中文：

该数据集包含12种不同手势的4600个样本。

使用FMCW非流动AWR1642从四个不同的人采集数据，每个示例都保存为与其手势类型关联的CSV文件。

(G1) 手臂向左-手臂从右向左完全滑动

(G2) 手臂向右-手臂从左向右完全滑动

(G3) 手离开-将手从雷达上移开

(G4) 手靠近-将一中手靠近雷达

(G5) 手臂向上——从底部到顶部的手臂运动

(G6) 手臂向下——从顶部到底部的手臂运动

(G7) 手掌向上——旋转手掌向上

(G8) 手掌向下——向下旋转手掌

(G9) 手向左——手向左移动（没有手臂移动）

(G10) 手向右——手向右移动

(G11) 横抱拳

(G12) 竖抱拳

数据集分为12个独立的文件平吴，与不同的手势类型相关联。每个文件夹都包含保存为CSV文件的手势示例

Csv文件的第一行是描述数据列的标题

FrameNumber、ObjectNumber，Range, Velocity, PeakValue, x坐标、y坐标

为了将手势读入Python程序中，将所有12个文件夹复制到名为”data”的单个文件夹下，并将“read_gesture.py”脚本复制到与’data’相同的文件中并运行它，脚本会将给定手势类型的csv文件转换为numpy矩阵。

如果你喜欢使用MATLAB，也可以使用Python将数据格式转为.mat.



以下是根据您提供的混淆矩阵数据制作的表格：

|    | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      | 10     | 11     | 12     |
| -- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 1  | 99.875 | 0.000  | 0.000  | 0.004  | 0.000  | 0.004  | 0.000  | 0.061  | 0.002  | 0.000  | 0.033  | 0.021  |
| 2  | 0.005  | 97.516 | 0.000  | 1.282  | 0.004  | 0.003  | 0.312  | 0.014  | 0.017  | 0.411  | 0.112  | 0.325  |
| 3  | 0.001  | 0.617  | 98.476 | 0.001  | 0.028  | 0.029  | 0.253  | 0.000  | 0.000  | 0.000  | 0.563  | 0.032  |
| 4  | 0.004  | 0.001  | 0.000  | 99.946 | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.007  | 0.041  |
| 5  | 0.000  | 0.000  | 0.014  | 0.000  | 87.154 | 12.507 | 0.178  | 0.027  | 0.006  | 0.112  | 0.001  | 0.001  |
| 6  | 0.001  | 0.000  | 0.007  | 0.000  | 15.160 | 82.157 | 0.052  | 0.277  | 0.319  | 1.000  | 0.713  | 0.315  |
| 7  | 0.000  | 4.357  | 0.002  | 0.001  | 0.512  | 0.010  | 94.990 | 0.000  | 0.003  | 0.056  | 0.023  | 0.047  |
| 8  | 2.802  | 0.008  | 0.000  | 0.304  | 0.512  | 0.120  | 0.008  | 91.919 | 2.067  | 2.148  | 0.095  | 0.017  |
| 9  | 0.000  | 0.009  | 0.000  | 0.002  | 1.444  | 0.025  | 0.002  | 0.047  | 74.676 | 23.639 | 0.108  | 0.045  |
| 10 | 0.000  | 0.001  | 0.000  | 0.001  | 0.808  | 1.165  | 0.004  | 1.194  | 3.916  | 92.871 | 0.031  | 0.011  |
| 11 | 0.001  | 0.014  | 0.001  | 0.042  | 0.003  | 0.000  | 0.000  | 0.000  | 0.000  | 0.001  | 84.138 | 15.799 |
| 12 | 0.014  | 0.042  | 0.000  | 0.821  | 0.005  | 0.000  | 0.000  | 0.001  | 0.016  | 1.266  | 10.372 | 87.462 |

表格中的行和列分别表示预测类别和实际类别。每个单元格中的数值表示预测为该行类别且实际为该列类别的样本比例。最后一列（1.000）表示每一行的总和，确保每一行的比例总和为1。
