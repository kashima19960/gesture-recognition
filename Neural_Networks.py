"""
author: 木人舟
brief: This code implements a classification model based on LSTM (Long Short-Term Memory Network)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from DataProcessed import *
import warnings
from ConstantDefinition import *
warnings.filterwarnings("ignore")  # Ignore warning messages, as long as the code runs (just kidding) 😋😋😋

# Default to using GPU for accelerated training. Please download the Pytorch version that supports Cuda, and your computer needs an Nvidia graphics card
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Chinese font. If you are not a Chinese user, you can ignore or comment out these two lines
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class LSTM(nn.Module):
    """
    Define a fully connected layer (linear layer) to map the output of LSTM to the number of classes.

    Parameters:
    - neurons_num: Number of neurons in the LSTM hidden layer.
    - class_number: Number of classes in the classification task.

    The input dimension of this fully connected layer is neurons_num * 80, and the output dimension is class_number.
    """
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.lstm = nn.LSTM(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        # LSTM is followed by a fully connected output layer, with a total of 12 output results (corresponding to 12 gestures)
        self.fc1 = nn.Linear(neurons_num * 80, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out


def train_model(net, train_X, train_y, test_X, test_y, criterion, optimizer, device):
    """
    Function to train the model.

    Parameters:
    - net: Neural network model, type RNN_LSTM.
    - train_X: Input features of the training dataset, type torch.Tensor.
    - train_y: Labels of the training dataset, type torch.Tensor.
    - test_X: Input features of the test dataset, type torch.Tensor.
    - test_y: Labels of the test dataset, type torch.Tensor.
    - criterion: Loss function, type torch.nn.modules.loss.
    - optimizer: Optimizer, type torch.optim.Optimizer.
    - device: Device type, type torch.device.

    Returns:
    - loss_list: List of loss values, recording the loss value for each iteration.
    - accuracy_list: List of accuracy values, recording the accuracy for each iteration.
    - iteration_list: List of iteration counts, recording the count for each iteration.

    This function iterates through the training dataset, calculates the loss, and updates the model parameters. 
    It calculates the accuracy on the test set every 200 iterations and records the loss and accuracy. 
    If the accuracy exceeds the set threshold, it terminates training early.
    """
    loss_list = []
    accuracy_list = []
    iteration_list = []
    total_accuracy = 0
    count = 0

    for epoch in range(epochs):
        for i in range(len(train_X)):
            data = train_X[i].to(device=device)
            targets = train_y[i].to(device=device)
            targets = torch.tensor(targets, dtype=torch.long)
            scores = net(data.view(-1, 80, 80 * frame_parameters))
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

            if count % 200 == 0:
                total_accuracy = calculate_accuracy(net, test_X, test_y, device)
                loss = loss.data.cpu()
                loss_list.append(loss)
                iteration_list.append(count)
                accuracy_list.append(total_accuracy)

        print(f'Iteration: {count}  Loss: {loss}  Accuracy: {total_accuracy*100} %')
        # break_limit can be modified in DataProcessed.py
        if total_accuracy*100 > break_limit:
            break
    return loss_list, accuracy_list, iteration_list

def calculate_accuracy(net, test_X, test_y, device):
    """
    Calculate the accuracy of the model on the test dataset.

    Parameters:
    - net: Neural network model, type RNN_LSTM.
    - test_X: Input features of the test dataset, type torch.Tensor.
    - test_y: Labels of the test dataset, type torch.Tensor.
    - device: Device type, type torch.device.

    Returns:
    - accuracy: Accuracy of the model on the test dataset, type float.

    This function iterates through the test dataset, calculates the proportion of correctly predicted samples 
    to the total number of samples, thus obtaining the accuracy of the model.
    """
    total_correct = 0
    total_number = 0
    with torch.no_grad():
        for label in LABELS:
            correct = 0
            number = 0
            for a in range(len(test_X)):
                if test_y[a] == LABELS[label]:
                    X = test_X[a].to(device=device)
                    y = test_y[a].to(device=device)

                    output = net(X.view(-1, 80, 80 * frame_parameters))
                    for idx, i in enumerate(output):
                        if torch.argmax(i) == y[idx]:
                            total_correct += 1
                            correct += 1
                        number += 1
                        total_number += 1

    return round(total_correct / total_number, 3)

def evaluate_model(net, test_X, test_y, device):
    """
    Evaluate the model's performance on the test dataset.
    Parameters:
    - net: Neural network model, type RNN_LSTM.
    - test_X: Input features of the test dataset, type torch.Tensor.
    - test_y: Labels of the test dataset, type torch.Tensor.
    - device: Device type, type torch.device.

    Returns:
    - total_accuracy: Total accuracy of the model on the test dataset, type float.
    - total_propability: Prediction probability matrix for each class, type numpy.ndarray.
    - total_classified: Classification result matrix for each class, type numpy.ndarray.

    This function iterates through the test dataset, calculates the proportion of correctly predicted samples 
    to the total number of samples, thus obtaining the total accuracy of the model.
    Additionally, it calculates the prediction probabilities and classification results for each class, 
    and generates the corresponding matrices.
    """
    total_number = 0
    total_correct = 0
    total_propability = np.zeros([class_number, class_number+1])
    total_classified = np.zeros([class_number, class_number+1])
    with torch.no_grad():
        for label in LABELS:
            correct = 0
            number = 0
            for a in range(len(test_X)):
                if test_y[a] == LABELS[label]:
                    X = test_X[a].to(device=device)
                    y = test_y[a].to(device=device)
                    output = net(X.view(-1, 80, 80 * frame_parameters))

                    out = F.softmax(output, dim=1)
                    out = out.cpu()
                    prob = out.numpy() * 100
                    prob = np.append(prob, 0)

                    y = y.cpu()
                    it = int(y.numpy())
                    total_propability[it] = total_propability[it] + prob
                    total_propability[it][class_number] = total_propability[it][class_number] + 1

                    for idx, i in enumerate(output):
                        gesture_label_num = torch.argmax(i)
                        total_classified[it][gesture_label_num] += 1
                        if torch.argmax(i) == y[idx]:
                            total_correct += 1
                            correct += 1
                        number += 1
                        total_number += 1

            print(label + " accuracy: ", round(correct / number, 3) * 100)
    total_accuracy = round(total_correct / total_number, 3) * 100
    print("Total accuracy: ", total_accuracy)
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:16.1f}'.format})
    for i in range(class_number):
        gest_num = int(total_propability[i][class_number])
        total_propability[i] = np.round((total_propability[i] / gest_num), 3)
    return total_accuracy,total_propability,total_classified


def visualize_loss_accuracy(loss_list, accuracy_list, iteration_list):
    """
    Plot the change curves of loss and accuracy.

    Parameters:
    - loss_list: List of loss values, recording the loss value for each iteration, type list.
    - accuracy_list: List of accuracy values, recording the accuracy for each iteration, type list.
    - iteration_list: List of iteration counts, recording the count for each iteration, type list.

    This function first plots the change curve of loss values with the number of iterations and saves it as an image file. 
    Then it plots the change curve of accuracy with the number of iterations and saves it as an image file. 
    Each plot is displayed after drawing.
    """
    plt.plot(iteration_list, loss_list)
    plt.xlabel("迭代次数")
    plt.ylabel("损失")
    plt.title("损失变化")
    plt.savefig("assets/loss.png",dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.plot(iteration_list, accuracy_list, color="red")
    plt.xlabel("迭代次数")
    plt.ylabel("准确率")
    plt.title("准确率变化")
    plt.savefig("assets/accuracy.png",dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def visualize_confusion_propability_matrix(matrix, title):
    """
    Visualize the confusion matrix or accuracy matrix.

    Parameters:
    - matrix: The matrix to be visualized, type numpy.ndarray.
    - title: The title of the image, type str.

    This function first takes the first 12 rows of the matrix and generates corresponding class labels. 
    Then it uses matplotlib to plot the heatmap of the matrix, adds a color bar, class labels, 
    and the values in the matrix. Finally, it saves the image and displays it.
    """
    pm = matrix[:12]
    classes = [f'G{i}' for i in range(1,13)]
    plt.imshow(pm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # 在矩阵中添加数值
    threshold = pm.max() / 2
    for i in range(pm.shape[0]):
        for j in range(pm.shape[1]):
            plt.text(j, i, format(pm[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if pm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(f"assets/{title}.png",dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def execute_ml_workflow(epochs, optimizer_step_value, test_percent, break_limit):
    """
    Main function to execute the machine learning workflow.

    Parameters:
    - epochs: Number of epochs for training, type int.
    - optimizer_step_value: Learning rate for the optimizer, type float.
    - test_percent: Percentage of the dataset to be used as the test set, type float.
    - break_limit: Accuracy threshold for early stopping, type float.

    This function first loads the dataset and splits it into training and test sets. 
    Then it initializes the neural network model, loss function, and optimizer.
    Next, it calls the training function to train the model, recording the loss and accuracy during training. 
    After training, it evaluates the model's performance on the test set, 
    and plots the change curves of the loss function and accuracy, the accuracy matrix, and the confusion matrix. 
    If the model's total accuracy exceeds the set threshold, it saves the model.
    """
    net = LSTM().to(device)
    gestures = DataPreprocessing()
    # If the training set does not exist, create one first
    if not os.path.exists(f"training_data.npy"):
        gestures.load_data()
    dataset = np.load("training_data.npy", allow_pickle=True)
    X = torch.Tensor([i[0] for i in dataset])
    y = torch.Tensor([i[1] for i in dataset])

    divide = int(len(dataset) * test_percent)
    train_X = X[:-divide]
    train_y = y[:-divide]
    test_X = X[-divide:]
    test_y = y[-divide:]
    print("Trainingset:", len(train_X))
    print("Testset", len(test_X))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=optimizer_step_value)

    loss_list, accuracy_list, iteration_list = train_model(net, train_X, train_y, test_X, test_y, criterion, optimizer, device)
    total_accuracy,total_propability,total_classified = evaluate_model(net, test_X, test_y, device)
    print("总准确率为",total_accuracy)
    if total_accuracy>break_limit:
        torch.save(net,"./model.pth")
    # Plot the change curves of loss and accuracy
    visualize_loss_accuracy(loss_list, accuracy_list, iteration_list)

    # Plot the accuracy matrix
    visualize_confusion_propability_matrix(total_propability, "Accuracy Matrix")

    # Plot the confusion matrix
    visualize_confusion_propability_matrix(total_classified, "Confusion Matrix")

if __name__ == '__main__':
    #测试是否能正确训练模型
    import time
    start = time.time()
    execute_ml_workflow(epochs, optimizer_step_value, test_percent, break_limit)
    end = time.time()
    print("总共花的时间", end - start)
