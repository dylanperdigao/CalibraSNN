import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.networks.cnn.BaseNet_CNN import BaseNet_CNN
#from BaseNet_CNN import BaseNet_CNN

class Net_CNN_2conv(nn.Module, BaseNet_CNN):
    """
    Convolutional Neural Network with 2 convolutional layers and 1 fully connected layer.

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
        self.fc1 = nn.Linear(topology[2][0], topology[2][1])
        # Architecture
        self.architecture = f"""
        =====================
        Conv1d({topology[0]})
        MaxPool1d(2)
        ---
        Conv1d({topology[1]})
        MaxPool1d(2)
        ---
        Linear({topology[2]})
        =====================
        """

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x
    
    
if __name__ == "__main__":
    CNN_1H = [(1, 64, 2), (64, 128, 2), (896, 2)]
    TOPOLOGIES = [CNN_1H]
    for topology in TOPOLOGIES:
        model = Net_CNN_2conv(
            topology=topology,
        )
        print(model.get_num_params())
        print(model.get_architecture())