from torch import Tensor
from torch.nn import LogSoftmax, NLLLoss

class ce_count_loss:
    """
    Cross-entropy loss for count rate coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Classes].
        targets (Tensor): Ground truth labels of shape [Batch].
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, weight=None):
        self.__name__ = "ce_count_loss"
        self.weight = weight

    def __call__(self, spk_out: Tensor, targets: Tensor):
        log_softmax_fn = LogSoftmax(dim=-1)
        loss_fn = NLLLoss(weight=self.weight)
        # Compute rates by summing spikes over time steps
        rates = self._encode(spk_out)
        # Convert rates to probabilities
        probabilities = log_softmax_fn(rates)  
        # Compute cross-entropy loss
        loss = loss_fn(probabilities, targets)
        return loss
    
    def _encode(self, spk_out: Tensor):
        # [Batch, Classes]
        return spk_out.sum(dim=0)
    
    def spike_code(self, spk_out: Tensor):
        # [Batch]
        return self._encode(spk_out).argmax(dim=1)


class ce_population_loss:
    """
    Cross-entropy loss for population rate coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Neurons].
        targets (Tensor): Ground truth labels of shape [Batch].
        num_neurons (int): Number of neurons per class population.
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, population: int, weight=None):
        self.__name__ = "ce_pop_loss"
        self.weight = weight
        self.population = population

    def __call__(self, spk_out: Tensor, targets: Tensor):
        log_softmax_fn = LogSoftmax(dim=-1)
        loss_fn = NLLLoss(weight=self.weight)
        rates = self._encode(spk_out)
        # Convert rates to probabilities
        probabilities = log_softmax_fn(rates)  
        # Compute cross-entropy loss
        loss = loss_fn(probabilities, targets)
        return loss

    def _encode(self, spk_out: Tensor):
        # Reshape to group neurons by class populations
        time_steps, batch_size, _ = spk_out.shape
        # [Steps, Batch, Classes, Neurons]
        spk_out = spk_out.view(time_steps, batch_size, -1, self.population)
        # [Batch, Classes]
        return spk_out.sum(dim=3).sum(dim=0)
    
    def spike_code(self, spk_out: Tensor):
        # [Batch]
        return self._encode(spk_out).argmax(dim=1)
    