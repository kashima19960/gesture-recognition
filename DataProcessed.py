import numpy as np
import os
from ConstantDefinition import *
"""
数据预处理类，用处就是将data文件夹下的所有手势csv文件转换
成ndarray的数据，然后保存成npy文件
"""
class DataProcessed():
    #Directories
    training_data = []
    def read_database(self, dir):
        """
        读取指定目录下的数据集，并将其处理为适合训练的格式。

        参数:
        dir (str): 数据集目录的名称。

        返回:
        tuple: 包含处理后的数据集和样本数量的元组。

        详细说明:
        1. 该函数首先遍历指定目录下的所有手势数据文件。
        2. 对于每个手势数据文件，读取其中的数据并跳过前两行（标题和空点）。
        3. 数据被处理为帧的形式，每帧包含多个点的位置、速度等信息。
        4. 处理后的数据被存储在一个三维数组中，其中每个元素代表一个手势的帧数据。
        5. 最后，函数返回处理后的数据集和样本数量。
        """
        dataset = []
        dirpath = "data/" + dir + "/"
        for gesture in os.listdir(dirpath):
            path = dirpath + gesture
            data = np.loadtxt(path, delimiter=",", skiprows=2)  # skip header and null point

            FrameNumber = 1   # counter for frames
            pointlenght = 80  # maximum number of points in array
            framelenght = 80  # maximum number of frames in array
            datalenght = int(len(data))
            gesturedata = np.zeros((framelenght, frame_parameters, pointlenght))
            counter = 0

            while counter < datalenght:
                velocity = np.zeros(pointlenght)
                peak_val = np.zeros(pointlenght)
                x_pos = np.zeros(pointlenght)
                y_pos = np.zeros(pointlenght)
                object_number = np.zeros(pointlenght)
                iterator = 0

                try:
                    while data[counter][0] == FrameNumber:
                        object_number = data[counter][1]
                        range = data[counter][2]
                        velocity[iterator] = data[counter][3]
                        peak_val[iterator] = data[counter][4]
                        x_pos[iterator] = data[counter][5]
                        y_pos[iterator] = data[counter][6]
                        iterator += 1
                        counter += 1
                except:
                    pass

                #这里取输入特征为x(必须)，y(必须)，多普勒速度(可选)
                framedata = np.array([x_pos, y_pos, velocity])
                try:
                    gesturedata[FrameNumber - 1] = framedata
                except:
                    pass
                FrameNumber += 1

            dataset.append(gesturedata)
            number_of_samples = len(dataset)

        return dataset, number_of_samples

    def load_data(self):
        """
        加载并预处理数据集。

        详细说明:
        1. 该函数遍历所有标签，并调用 `read_database` 方法读取每个标签对应的数据集。
        2. 对于每个数据集，将其处理为适合训练的格式，并附加相应的标签。
        3. 处理后的数据被存储在 `training_data` 列表中。
        4. 最后，函数打印每个标签的数据集大小，并计算总样本数量。
        5. 数据集被随机打乱，并保存为 `training_data.npy` 文件。
        """
        total = 0
        for label in LABELS:
            trainset, number_of_samples = self.read_database(label)

            for data in trainset:
                self.training_data.append([np.array(data),np.array([LABELS[label]])]) #save data and assign label
            total = total + number_of_samples
            print(label,number_of_samples)

        print("Total number:", total)

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", np.array(self.training_data, dtype=object))


#测试预处理功能是否能工作正常
if __name__=='__main__':
    data = DataProcessed()
    data.load_data()