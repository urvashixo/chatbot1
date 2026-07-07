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
        self.relu = nn.ReLU() #Breaks the linearity of the model and allows it to learn complex patterns
        self.dropout = nn.Dropout(0.5) # Dropout layer with 50% probability for regularization

    def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.dropout(x) # Apply dropout after the first layer
        x=self.relu(self.fc2(x))
        x=self.dropout(x) # Apply dropout after the second layer
        x=self.fc3(x) # No activation function for the output layer
    
class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings = None):
        self.model=None
        self.intents_path=intents_path
        self.documents=[]
        self.vocabulary=[]
        self.intents= [] #probabilities for each intent
        self.intents_responses=[] #responses for each intent

        self.function_mappings = function_mappings
        self.X= None #X is matrix
        self.y= None #y is a vector

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)