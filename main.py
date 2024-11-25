"""
author: 木人舟
brief: Running this file will start the training of the model, evaluate the network performance, and save the model parameters to the file model.pth.
"""
# Running this code directly will show you the results! 🤠🤠🤠

import torch
from Neural_Networks import *
from DataProcessed import DataPreprocessing
import numpy as np
import os
import torch
from Neural_Networks import *
from DataProcessed import DataPreprocessing
import numpy as np
import os

"""
Main function to load the model and evaluate its accuracy on the test data.
Steps:
1. Load the pre-trained LSTM model.
2. Check if the preprocessed training data file exists. If it exists, load it directly; otherwise, process the data again and save it.
3. Convert the data to PyTorch tensors.
4. Evaluate the model's accuracy on the test data using GPU (if available) or CPU.
5. Print the model's accuracy.
"""
def main():
    model = LSTM()
    """
    Only when the model evaluation accuracy is greater than break_limit, the trained model parameters will be saved to the model.pth file.
    If you do not want to keep training the model, you can set break_limit to a smaller value.
    """
    while not os.path.exists("./model.pth"):
        execute_ml_workflow(epochs, optimizer_step_value, test_percent, break_limit)
    model = torch.load("./model.pth", weights_only=False)
    model.eval()

    if not os.path.exists("training_data.npy"):
        data_processed = DataPreprocessing()
        dataset = data_processed.load_data()
    else:
        dataset = np.load("training_data.npy", allow_pickle=True)
    test_X = torch.Tensor([i[0] for i in dataset])
    test_y = torch.Tensor([i[1] for i in dataset])
    # Evaluate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracy = evaluate_model(model, test_X, test_y, device)

    print(f"The model's accuracy on the test data is: {accuracy * 100}%")
if __name__ == '__main__':
    main()