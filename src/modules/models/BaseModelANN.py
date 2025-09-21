import numpy as np
import torch

from torch.nn import CrossEntropyLoss
from modules.other.utils import experimental_print
from modules.models.BaseModel import BaseModel

class BaseModelANN(BaseModel):
    def __init__(self, num_features, num_classes, hyperparameters, **kwargs):
        super().__init__(num_features, num_classes, hyperparameters, **kwargs) 
        self.adam_betas=hyperparameters['adam_beta']
        self.learning_rate=hyperparameters['learning_rate']
        self.class_weights = torch.tensor((1-hyperparameters['weight'], hyperparameters['weight']), dtype=self.dtype, device=self.device)
        self.loss_fn = CrossEntropyLoss(weight=self.class_weights)
        print_str = [
            "=======================================",
            "BaseModelANN",
            "=======================================",
            f"- Device: {self.device}",
            f"- Batch size: {self.hyperparameters['batch']}",
            f"- Epochs: {self.hyperparameters['epoch']}",
            f"- Class weights: {self.class_weights}",
            f"- Adam betas: {self.adam_betas}",
            f"- Learning rate: {self.learning_rate}",
            "======================================="
        ]
        experimental_print("\n".join(print_str)) if self.verbose >= 1 else None

    def fit(self, x_train, y_train):
        experimental_print("Training...") if self.verbose >= 2 else None
        self._train_loader = self.load_data(x_train, y_train)
        # Outer training loop
        for epoch in range(self.hyperparameters['epoch']):
            experimental_print(f"Epoch - {epoch}") if self.verbose >= 2 else None
            train_batch = iter(self._train_loader)
            running_loss = 0.0
            for i, (data, targets) in enumerate(train_batch):
                data = data.to(self.device).to(self.dtype)
                targets = targets.to(self.device).long()
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.network(data)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    experimental_print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}') if self.verbose >= 2 else None
                    running_loss = 0.0
               
    def predict(self, x_test, y_test):
        experimental_print("Predicting...") if self.verbose >= 2 else None
        self.test_loader = self.load_data(x_test, y_test)
        test_batch = iter(self.test_loader)
        predictions = np.array([])
        test_targets = np.array([])
        with torch.no_grad():
            for i, (data, targets) in enumerate(test_batch):
                data = data.to(self.device).to(self.dtype)
                targets = targets.to(self.device).long()
                # calculate outputs by running images through the network
                outputs = self.network(data)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                # store the predictions
                predictions = np.append(predictions, predicted.cpu().numpy())
                test_targets = np.append(test_targets, targets.cpu().numpy())
        return predictions, test_targets