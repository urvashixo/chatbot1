import os
import random
import json
import nltk
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
nltk.download('punkt_tab')

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128) #128 neurons
        self.fc2 = nn.Linear(128, 64) #64 neurons
        self.fc3 = nn.Linear(64, output_size) #output_size neurons
        self.relu = nn.ReLU()