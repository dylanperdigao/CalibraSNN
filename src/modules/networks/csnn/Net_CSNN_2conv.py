import snntorch as snn
import torch
import torch.nn as nn

from modules.networks.csnn.BaseNet_CSNN import BaseNet_CSNN

class Net_CSNN_2conv(nn.Module, BaseNet_CSNN):
    """
    Convolutional Spiking Neural Network with 2 convolutional layers and 1 fully connected layer.

    Parameters
    ----------
    topology : list
        List of tuples with the topology of the model
    params : dict
        Dictionary with the parameters of the model such as beta, slope, step and threshold
    """
    def __init__(self, topology, params):
        super().__init__()
        self.betas = params['beta']
        self.spike_grad = params['slope']
        self.num_steps = params['step']
        self.thresholds = params['threshold']
        # Layer1
        self.conv1 = nn.Conv1d(topology[0][0], topology[0][1], topology[0][2])
        self.mp1 = nn.MaxPool1d(2)
        self.lif1 = snn.Leaky(beta=self.betas[0], spike_grad=self.spike_grad, threshold=self.thresholds[0], learn_beta=True, learn_threshold=True)
        # Layer2
        self.conv2 = nn.Conv1d(topology[1][0], topology[1][1], topology[1][2])
        self.mp2 = nn.MaxPool1d(2)
        self.lif2 = snn.Leaky(beta=self.betas[1], spike_grad=self.spike_grad, threshold=self.thresholds[1], learn_beta=True, learn_threshold=True)
        # Layer3
        self.fc1 = nn.Linear(topology[2][0], topology[2][1])
        self.lif3 = snn.Leaky(beta=self.betas[2], spike_grad=self.spike_grad, threshold=self.thresholds[2], learn_beta=True, learn_threshold=True, output=True)
        # Architecture
        self.architecture = f"""
        =====================
        Conv1d({topology[0]})
        MaxPool1d(2)
        Leaky({self.betas[0]}, {self.thresholds[0]})
        ---
        Conv1d({topology[1]})
        MaxPool1d(2)
        Leaky({self.betas[1]}, {self.thresholds[1]})
        ---
        Linear({topology[2]})
        Leaky({self.betas[2]}, {self.thresholds[2]}, output=True)
        =====================
        """

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        tuple
            Tuple with the current, spike and membrane potential of the last layer
        """
        cur_last_rec = []
        spk_last_rec = []
        mem_last_rec = []
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()
        for _ in range(self.num_steps):
            # Layer1
            cur1 = self.mp1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            # Layer2
            cur2 = self.mp2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)   
            # Layer3      
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)
            # Record the final layer
            cur_last_rec.append(cur3)
            spk_last_rec.append(spk3)
            mem_last_rec.append(mem3)
        return torch.stack(cur_last_rec), torch.stack(spk_last_rec), torch.stack(mem_last_rec)