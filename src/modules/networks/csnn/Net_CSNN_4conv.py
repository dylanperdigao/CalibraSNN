import snntorch as snn
import torch
import torch.nn as nn

from modules.networks.csnn.BaseNet_CSNN import BaseNet_CSNN

class Net_CSNN_4conv(nn.Module, BaseNet_CSNN):
    """
    Convolutional Spiking Neural Network with 4 convolutional layers and 1 fully connected layer.

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
        self.conv3 = nn.Conv1d(topology[2][0], topology[2][1], topology[2][2])
        self.mp3 = nn.MaxPool1d(2)
        self.lif3 = snn.Leaky(beta=self.betas[2], spike_grad=self.spike_grad, threshold=self.thresholds[2], learn_beta=True, learn_threshold=True)
        # Layer4
        self.conv4 = nn.Conv1d(topology[3][0], topology[3][1], topology[3][2])
        self.mp4 = nn.MaxPool1d(2)
        self.lif4 = snn.Leaky(beta=self.betas[3], spike_grad=self.spike_grad, threshold=self.thresholds[3], learn_beta=True, learn_threshold=True)
        # Layer5
        self.fc1 = nn.Linear(topology[4][0], topology[4][1])
        self.lif5 = snn.Leaky(beta=self.betas[4], spike_grad=self.spike_grad, threshold=self.thresholds[4], learn_beta=True, learn_threshold=True, output=True)
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
        Conv1d({topology[2]})
        MaxPool1d(2)
        Leaky({self.betas[2]}, {self.thresholds[2]})
        ---
        Conv1d({topology[3]})
        MaxPool1d(2)
        Leaky({self.betas[3]}, {self.thresholds[3]})
        ---
        Linear({topology[4]})
        Leaky({self.betas[4]}, {self.thresholds[4]}, output=True)
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
        mem4 = self.lif4.reset_mem()
        mem5 = self.lif5.reset_mem()
        for _ in range(self.num_steps):
            # Layer1
            cur1 = self.mp1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            # Layer2
            cur2 = self.mp2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)   
            # Layer3 
            cur3 = self.mp3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            # Layer4
            cur4 = self.mp4(self.conv4(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)
            # Layer5
            cur5 = self.fc1(spk4.flatten(1))
            spk5, mem5 = self.lif5(cur5, mem5)
            # Record the final layer
            cur_last_rec.append(cur5)
            spk_last_rec.append(spk5)
            mem_last_rec.append(mem5)
        return torch.stack(cur_last_rec), torch.stack(spk_last_rec), torch.stack(mem_last_rec)