import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Neural_Networks import *
from ConstantDefinition import *

model=RNN_LSTM()
model=torch.load("./model.pth",weights_only=False)
model.eval()
print(model)