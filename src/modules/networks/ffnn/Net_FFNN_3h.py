import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.networks.ffnn.BaseNet_FFNN import BaseNet_FFNN
#from BaseNet_FFNN import BaseNet_FFNN

class Net_FFNN_3h(nn.Module, BaseNet_FFNN):
    def __init__(self, topology, input=None):
        super().__init__()
        self.input = input if not None else topology[0]
        # Layer1/2
        self.fc1 = nn.Linear(self.input, topology[1])
        # Layer2/3
        self.fc2 = nn.Linear(topology[1], topology[2])
        # Layer3/4
        self.fc3 = nn.Linear(topology[2], topology[3])
        # Layer4/5
        self.fc4 = nn.Linear(topology[3], topology[4])
        # Architecture
        self.architecture = f"""
        =====================
        Linear({self.input}, {topology[1]})
        ---
        Linear({topology[1]}, {topology[2]})
        ---
        Linear({topology[2]}, {topology[3]})
        ---
        Linear({topology[3]}, {topology[4]})
        =====================
        """

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = torch.flatten(self.fc4(x), 1)
        return x
    
if __name__ == "__main__":
    topology = (31, 64, 64, 64, 2)
    model = Net_FFNN_3h(topology)
    print(model.get_architecture())
    #print(model.get_parameters())
    print(f"Number of parameters for {topology}: {model.get_num_params()}")