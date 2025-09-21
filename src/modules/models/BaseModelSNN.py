import numpy as np
import torch

from torch.nn import CrossEntropyLoss
from snntorch.surrogate import fast_sigmoid
from modules.other.utils import experimental_print
from modules.models.BaseModel import BaseModel

class BaseModelSNN(BaseModel):
    def __init__(self, num_features, num_classes, hyperparameters, **kwargs):
        super().__init__(num_features, num_classes, hyperparameters, **kwargs) 
        self.snn_params = {
            'threshold': hyperparameters['threshold'],
            'step': hyperparameters['step'],
            'beta': hyperparameters['beta'],
            'slope':fast_sigmoid(hyperparameters['slope'])
        }
        self.adam_betas=hyperparameters['adam_beta']
        self.learning_rate=hyperparameters['learning_rate']
        self.class_weights = torch.tensor((1-hyperparameters['weight'], hyperparameters['weight']), dtype=self.dtype, device=self.device)
        self.loss_fn = CrossEntropyLoss(weight=self.class_weights)
        print_str = [
            "=======================================",
            "BaseModelSNN",
            "=======================================",
            f"- Device: {self.device}",
            f"- Batch size: {self.hyperparameters['batch']}",
            f"- Epochs: {self.hyperparameters['epoch']}",
            f"- Steps: {self.snn_params['step']}",
            f"- Betas: {self.snn_params['beta']}",
            f"- Spike grad slope: {self.hyperparameters['slope']}",
            f"- Thresholds: {self.snn_params['threshold']}",
            f"- Class weights: {self.class_weights}",
            f"- Adam betas: {self.adam_betas}",
            f"- Learning rate: {self.learning_rate}",
            "======================================="
        ]
        experimental_print("\n".join(print_str)) if self.verbose >= 1 else None

    def fit(self, x_train, y_train):
        """Training loop for the network.
        >>> spk_rec.shape 
        >>> # (num_steps, batch_size, num_outputs)
        --
        >>> spk_rec.sum(0) 
        >>> # efetua a soma dos spikes de cada neurônio da ultima camada
        >>> # (batch_size, num_outputs)
        --
        >>> spk_rec.sum(0).argmax(1) # retorna o indice do spike mais prevalente para dizer se é 0 ou 1
        """
        self._train_loader = self.load_data(x_train, y_train)
        for epoch in range(self.hyperparameters['epoch']):
            experimental_print(f"Epoch - {epoch}") if self.verbose >= 2 else None
            train_batch = iter(self._train_loader)
            for data, targets in train_batch:
                data = data.to(self.device)
                targets = targets.to(self.device, dtype=torch.long)
                self.network.train()
                _, _, mem_rec = self.network(data) 
                loss_val = torch.zeros((1), dtype=self.dtype, device=self.device)
                for step in range(self.snn_params['step']):
                    loss_val += self.loss_fn(mem_rec[step], targets)
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()
                experimental_print(f"Loss: {loss_val.item()}") if self.verbose >= 3 else None
               


    def predict(self, x_test, y_test):
        experimental_print("Predicting...") if self.verbose >= 2 else None
        self._test_loader = self.load_data(x_test, y_test)
        predictions = np.array([])
        test_targets = np.array([])
        with torch.no_grad():
            self.network.eval()
            for data, targets in iter(self._test_loader):
                data = data.to(self.device)
                targets = targets.to(self.device, dtype=torch.long)
                _, spk_rec, _ = self.network(data)
                spike_count = spk_rec.sum(0)
                _, max_spike = spike_count.max(1)
                predictions = np.append(predictions, max_spike.cpu().numpy())
                test_targets = np.append(test_targets, targets.cpu().numpy())
        return predictions, test_targets