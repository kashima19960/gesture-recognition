import numpy as np
import os
from ConstantDefinition import *

class DataProcessed():
    #Directories
    training_data = []
    def read_database(self, dir):
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

                #############Choosing paramaters to extract########################
                framedata = np.array([x_pos, y_pos, velocity])
                ###################################################################

                try:
                    gesturedata[FrameNumber - 1] = framedata
                except:
                    pass
                FrameNumber += 1

            dataset.append(gesturedata)
            number_of_samples = len(dataset)

        return dataset, number_of_samples

    def load_data(self):
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