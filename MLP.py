import os 
import torch 
import torch.nn as nn
import tdqm 
import torch.optim as optim 

# Input size menyesuaikan tempat ditaruhnya MLP pada Block YOLO
# num_classes = seperti output, hasilnya akan mengeluarkan 3 jenis

class MLP(nn.module):
    def __init__ (self, input_size=0*0*3, hidden_size=[], num_classes = 3):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten(),
        self.fc1 = nn.Linear(input_size, hidden_size[0]), 
        self.fc2= nn.Linear(hidden_size[0], hidden_size[1]),
        self.fc3= nn.Linear(hidden_size, num_classes)
        
    def forward (self,x):
        x = self.flatten(x),
        x = torch.relu(self.fc1(x))
        x= torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x 