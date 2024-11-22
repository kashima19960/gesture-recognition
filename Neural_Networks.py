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
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Pytorch version --",torch.__version__)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#训练数据集不同，导致训练出来的差异很大
class RNN_LSTM(nn.Module):
    def __init__(self):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.lstm = nn.LSTM(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
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
        if total_accuracy*100 > break_limit:
            break
    return loss_list, accuracy_list, iteration_list

def calculate_accuracy(net, test_X, test_y, device):
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
    # save_results(total_accuracy, total_propability, total_classified)
    for i in range(class_number):
        gest_num = int(total_propability[i][class_number])
        total_propability[i] = np.round((total_propability[i] / gest_num), 3)
    return total_accuracy,total_propability,total_classified

#这个函数后面考虑弃用
# def save_results(total_accuracy, total_propability, total_classified):
    with open("results.txt", 'a') as f:
        print(total_accuracy, file=f)

    for i in range(class_number):
        gest_num = int(total_propability[i][class_number])
        total_propability[i] = np.round((total_propability[i] / gest_num), 3)

    np.savetxt("propability matrix.txt", total_propability, fmt='%f')
    np.savetxt("confusion matrix.txt", total_classified, fmt='%f')

def visualize_loss_accuracy(loss_list, accuracy_list, iteration_list):
    plt.plot(iteration_list, loss_list)
    plt.xlabel("迭代次数")
    plt.ylabel("损失")
    plt.title("循环神经网络：损失 vs 迭代次数")
    plt.savefig("assets/loss.png",dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.plot(iteration_list, accuracy_list, color="red")
    plt.xlabel("迭代次数")
    plt.ylabel("准确率")
    plt.title("循环神经网络：准确率 vs 迭代次数")
    plt.savefig("assets/accuracy.png",dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def visualize_confusion_propability_matrix(matrix,title):
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
    net = RNN_LSTM().to(device)
    gestures = DataProcessed()
    #训练集不存在的话，要先创建一个训练集
    if not os.path.exists(f"training_data.npy"):
        #调用DataProcessed类的load_data函数，会生成处理后的数据集training_data.npy
        gestures.load_data()
    dataset = np.load("training_data.npy", allow_pickle=True)
    X = torch.Tensor([i[0] for i in dataset])
    y = torch.Tensor([i[1] for i in dataset])

    val_size = int(len(dataset) * test_percent)
    train_X = X[:-val_size]
    train_y = y[:-val_size]
    test_X = X[-val_size:]
    test_y = y[-val_size:]

    print("Trainingset:", len(train_X))
    print("Testset", len(test_X))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=optimizer_step_value)

    loss_list, accuracy_list, iteration_list = train_model(net, train_X, train_y, test_X, test_y, criterion, optimizer, device)
    total_accuracy,total_propability,total_classified = evaluate_model(net, test_X, test_y, device)
    print("总准确率为",total_accuracy)
    if total_accuracy>break_limit:
        torch.save(net,"./model.pth")
    #绘制损失函数与准确率的变化曲线
    visualize_loss_accuracy(loss_list, accuracy_list, iteration_list)

    #绘制准确率矩阵
    visualize_confusion_propability_matrix(total_propability,"准确率矩阵")

    # 绘制混淆矩阵
    visualize_confusion_propability_matrix(total_classified, "混淆矩阵")

if __name__ == '__main__':
    #测试函数
    import time
    start = time.time()
    execute_ml_workflow(epochs, optimizer_step_value, test_percent, break_limit)
    end = time.time()
    print("总共花的时间", end - start)
