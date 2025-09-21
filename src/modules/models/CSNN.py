from torch.optim import Adam
from modules.models import BaseModelSNN
from modules.networks import Net_CSNN_2conv
from modules.networks import Net_CSNN_3conv
from modules.networks import Net_CSNN_4conv

class CSNN(BaseModelSNN):
    def __init__(self, num_features, num_classes, hyperparameters, **kwargs):
        super().__init__(num_features, num_classes, hyperparameters, **kwargs)
        self.network = self.load_network()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, betas=self.adam_betas, weight_decay=0)

    def load_network(self):
        if len(self.topology)-1 == 2:
            network = Net_CSNN_2conv(self.topology, self.snn_params)
        elif len(self.topology)-1 == 3:
            network = Net_CSNN_3conv(self.topology, self.snn_params)
        elif len(self.topology)-1 == 4:
            network = Net_CSNN_4conv(self.topology, self.snn_params)
        else:
            raise ValueError("Convolutional SNN architecture not found.")
        return network.to(self.device)
