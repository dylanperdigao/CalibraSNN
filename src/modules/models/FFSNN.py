from torch.optim import Adam
from modules.models import BaseModelSNN
from modules.networks import Net_FFSNN_1h
from modules.networks import Net_FFSNN_2h
from modules.networks import Net_FFSNN_3h

class FFSNN(BaseModelSNN):
    def __init__(self, num_features, num_classes, hyperparameters, **kwargs):
        super().__init__(num_features, num_classes, hyperparameters, **kwargs)
        self.network = self.load_network()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, betas=self.adam_betas, weight_decay=0)

    def load_network(self):
        if len(self.topology) == 3:
            network = Net_FFSNN_1h(self.topology, self.snn_params, input=self.num_features)
        elif len(self.topology) == 4:
            network = Net_FFSNN_2h(self.topology, self.snn_params, input=self.num_features)
        elif len(self.topology) == 5:
            network = Net_FFSNN_3h(self.topology, self.snn_params, input=self.num_features)
        else:
            raise ValueError("Fully Connected SNN architecture not found.")
        return network.to(self.device)
