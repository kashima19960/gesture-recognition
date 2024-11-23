import torch
from Neural_Networks import RNN_LSTM, evaluate_model
from DataProcessed import DataProcessed
import numpy as np
import os

import torch
from Neural_Networks import RNN_LSTM, evaluate_model
from DataProcessed import DataProcessed
import numpy as np
import os
def main():
    """
    主函数，用于加载模型并评估其在测试数据上的准确率。

    步骤如下：
    1. 加载预训练的RNN_LSTM模型。
    2. 检查是否存在预处理后的训练数据文件。如果存在，则直接加载；否则，重新处理数据并保存。
    3. 将数据转换为PyTorch张量。
    4. 使用GPU（如果可用）或CPU评估模型在测试数据上的准确率。
    5. 打印模型的准确率。
    """
    model = RNN_LSTM()
    model = torch.load("./model.pth", weights_only=False)
    model.eval()

    if not os.path.exists("training_data.npy"):
        data_processed = DataProcessed()
        dataset= data_processed.load_data()
    else:
        dataset = np.load("training_data.npy", allow_pickle=True)
    test_X = torch.Tensor([i[0] for i in dataset])
    test_y = torch.Tensor([i[1] for i in dataset])
    # 评估模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracy = evaluate_model(model, test_X, test_y, device)

    print(f"模型在测试数据上的准确率为: {accuracy * 100}%")

if __name__=='__main__':
    main()