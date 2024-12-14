"""
author: 木人舟
brief: Converts all gesture csv files in the data folder into ndarray data and saves them as npy files.
contact:CodingCV@outlook.com
"""
import numpy as np
import os
from ConstantDefinition import *

"""
Data preprocessing class, used to convert all gesture csv files in the data folder
into ndarray data and save them as npy files.
"""
class DataPreprocessing():
    # Directories
    training_data = []

    def read_database(self, dir):
        """
        Reads the dataset from the specified directory and processes it into a format suitable for training.

        Parameters:
        dir (str): The name of the dataset directory.

        Returns:
        tuple: A tuple containing the processed dataset and the number of samples.

        Detailed Description:
        1. The function first iterates through all gesture data files in the specified directory.
        2. For each gesture data file, it reads the data and skips the first two rows (header and null point).
        3. The data is processed into frame format, where each frame contains multiple points' positions, velocities, etc.
        4. The processed data is stored in a three-dimensional array, where each element represents a gesture's frame data.
        5. Finally, the function returns the processed dataset and the number of samples.
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
        Load and preprocess the dataset.
        Detailed instructions:
        1. The function iterates through all labels and calls the `read_database` method to load the dataset corresponding to each label.
        2. For each dataset, it is processed into a format suitable for training and appended with the corresponding label.
        3. The processed data is stored in the `training_data` list.
        4. Finally, the function prints the size of the dataset for each label and calculates the total number of samples.
        5. The dataset is shuffled randomly and saved as the `training_data.npy` file.
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


#Testing if the preprocessing function works properly
if __name__=='__main__':
    data = DataPreprocessing()
    data.load_data()