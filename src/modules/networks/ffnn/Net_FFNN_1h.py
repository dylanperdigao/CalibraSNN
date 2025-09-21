import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.networks.ffnn.BaseNet_FFNN import BaseNet_FFNN
#from BaseNet_FFNN import BaseNet_FFNN

class Net_FFNN_1h(nn.Module, BaseNet_FFNN):
    def __init__(self, topology, input=None):
        super().__init__()
        self.input = input if not None else topology[0]
        # Layer1/2
        self.fc1 = nn.Linear(self.input, topology[1])
        # Layer2/3
        self.fc2 = nn.Linear(topology[1], topology[2])
        # Architecture
        self.architecture = f"""
        =====================
        Linear({self.input}, {topology[1]})
        ---
        Linear({topology[1]}, {topology[2]})
        =====================
        """

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = torch.flatten(self.fc2(x), 1)
        return x
    
if __name__ == "__main__":
    topology = (31, 64, 2)
    model = Net_FFNN_1h(topology)
    print(model.get_architecture())
    #print(model.get_parameters())
    print(f"Number of parameters for {topology}: {model.get_num_params()}")