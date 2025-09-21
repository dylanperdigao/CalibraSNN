import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.networks.cnn.BaseNet_CNN import BaseNet_CNN

class Net_CNN_3conv(nn.Module, BaseNet_CNN):
    """
    Convolutional Neural Network with 3 convolutional layers and 1 fully connected layer.

    Parameters
    ----------
    topology : list
        List of tuples with the topology of the model
    """
    def __init__(self, topology):
        super().__init__()
        # Layer1
        self.conv1 = nn.Conv1d(topology[0][0], topology[0][1], topology[0][2])
        self.mp1 = nn.MaxPool1d(2)
        # Layer2
        self.conv2 = nn.Conv1d(topology[1][0], topology[1][1], topology[1][2])
        self.mp2 = nn.MaxPool1d(2)
        # Layer3
        self.conv3 = nn.Conv1d(topology[2][0], topology[2][1], topology[2][2])
        self.mp3 = nn.MaxPool1d(2)
        # Layer4
        self.fc1 = nn.Linear(topology[3][0], topology[3][1])
        # Architecture
        self.architecture = f"""
        =====================
        Conv1d({topology[0]})
        MaxPool1d(2)
        ---
        Conv1d({topology[1]})
        MaxPool1d(2)
        ---
        Conv1d({topology[2]})
        MaxPool1d(2)
        ---
        Linear({topology[3]})
        =====================
        """

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = self.mp3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x
    
    
if __name__ == "__main__":
    CNN_2H = [(1, 64, 2), (64, 128, 2), (128, 256, 2), (768, 2)]
    TOPOLOGIES = [CNN_2H]
    for topology in TOPOLOGIES:
        model = Net_CNN_3conv(
            topology=topology,
        )
        print(model.get_num_params())
        print(model.get_architecture())