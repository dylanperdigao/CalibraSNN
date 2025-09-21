import torch
import torch.nn as nn
import snntorch as snn
from snntorch.surrogate import fast_sigmoid

from modules.networks.ffsnn.BaseNet_FFSNN import BaseNet_FFSNN

class Net_FFSNN_3h(nn.Module, BaseNet_FFSNN):
    def __init__(self, topology, params, input=None):
        super().__init__()
        self.betas = params['beta']
        self.spike_grad = params['slope']
        self.num_steps = params['step']
        self.thresholds = params['threshold']
        self.input = input if not None else topology[0]
        # Layer1/2
        self.fc1 = nn.Linear(self.input, topology[1])
        self.lif1 = snn.Leaky(beta=self.betas[0], threshold=self.thresholds[0], spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        # Layer2/3
        self.fc2 = nn.Linear(topology[1], topology[2])
        self.lif2 = snn.Leaky(beta=self.betas[1], threshold=self.thresholds[1], spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        # Layer3/4
        self.fc3 = nn.Linear(topology[2], topology[3])
        self.lif3 = snn.Leaky(beta=self.betas[2], threshold=self.thresholds[2], spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        # Layer4/5
        self.fc4 = nn.Linear(topology[3], topology[4])
        self.lif4 = snn.Leaky(beta=self.betas[3], threshold=self.thresholds[3], spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True, output=True)
        # Architecture
        self.architecture = f"""
        =====================
        Linear({self.input}, {topology[1]})
        Leaky({self.betas[0]}, {self.thresholds[0]})
        ---
        Linear({topology[1]}, {topology[2]})
        Leaky({self.betas[1]}, {self.thresholds[1]})
        ---
        Linear({topology[2]}, {topology[3]})
        Leaky({self.betas[2]}, {self.thresholds[2]})
        ---
        Linear({topology[3]}, {topology[4]})
        Leaky({self.betas[3]}, {self.thresholds[3]})
        =====================
        """
    def forward(self, x):
        cur_last_rec = []
        spk_last_rec = []
        mem_last_rec = []
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem4 = self.lif4.reset_mem()
        for _ in range(self.num_steps):
            # Layer1/2
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            # Layer2/3
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            # Layer3/4
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            # Layer4/5
            cur4 = self.fc4(spk3.flatten(1))
            spk4, mem4 = self.lif4(cur4, mem4)
            # Record the final layer
            cur_last_rec.append(cur4)
            spk_last_rec.append(spk4)
            mem_last_rec.append(mem4)
        return torch.stack(cur_last_rec,dim=0), torch.stack(spk_last_rec, dim=0), torch.stack(mem_last_rec, dim=0)
    
if __name__ == "__main__":
    input_size = 31
    output_size = 2
    topology = (input_size, 64, 64, 64, output_size)
    params = {
        'beta': (0.9, 0.8, 0.7, 0.6),
        'slope': fast_sigmoid(25),
        'step': 10,
        'threshold': (0.5, 0.5, 0.5, 0.5)
    }
    model = Net_FFSNN_3h(topology, params)
    print(model.get_architecture())
    #print(model.get_parameters())
    print(f"Number of parameters for {topology}: {model.get_num_params()}")