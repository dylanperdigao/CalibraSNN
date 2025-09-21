from torch import Tensor, zeros, nonzero, tensor
from torch.nn import LogSoftmax, NLLLoss

class ce_ttfs_loss:
    """
    Cross-entropy loss for time-to-first-spike coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Classes].
        targets (Tensor): Ground truth labels of shape [Batch].
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, weight=None):
        super().__init__()
        self.__name__ = "ce_ttfs_loss"
        self.weight = weight

    def __call__(self, spk_out: Tensor, targets: Tensor):
        log_softmax_fn = LogSoftmax(dim=-1)
        loss_fn = NLLLoss(weight=self.weight)
        # Compute rates by summing spikes over time steps
        rates = self._encode(spk_out)
        # Convert rates to probabilities
        probabilities: Tensor = log_softmax_fn(rates.float())
        probabilities.requires_grad_(True)
        # Compute cross-entropy loss
        loss = loss_fn(probabilities, targets)
        return loss
    
    def _encode(self, spk_out: Tensor):
        # [Batch, Classes]
        time_steps, _, _ = spk_out.shape
        # last element of steps is 1
        spk_out[-1] = 1
        # select the first spike time
        spk_out = spk_out.argmax(dim=0)
        # compute the distance from the first spike to the last time step
        spk_out = -(spk_out - time_steps)
        return spk_out
    
    def spike_code(self, spk_out: Tensor):
        # [Batch]
        return self._encode(spk_out).argmax(dim=1)
    
class ce_phase_loss:
    """
    Cross-entropy loss for phase rate coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Classes].
        targets (Tensor): Ground truth labels of shape [Batch].
        phase (int): Number of steps to sum over.
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, phase, weight=None):
        super().__init__()
        self.__name__ = "ce_phase_loss"
        self.phase = phase
        self.weight = weight

    def __call__(self, spk_out: Tensor, targets: Tensor):
        log_softmax_fn = LogSoftmax(dim=-1)
        loss_fn = NLLLoss(weight=self.weight)
        # Compute rates by summing spikes over time steps
        rates = self._encode(spk_out)
        rates.requires_grad_(True)
        # Convert rates to probabilities
        probabilities = log_softmax_fn(rates)  
        # Compute cross-entropy loss
        loss = loss_fn(probabilities, targets)
        return loss
    
    def _encode(self, spk_out: Tensor):
        # [Batch, Classes]
        steps, batch, classes = spk_out.shape
        assert steps % self.phase == 0, "Number of steps must be divisible by the phase."
        spk_out = spk_out.view(steps//self.phase, -1, batch, classes)
        return spk_out.sum(dim=0).argmax(dim=0).float()
    
    def spike_code(self, spk_out: Tensor):
        # [Batch]
        return self._encode(spk_out).argmax(dim=1)

class ce_phase2_loss:
    """
    Cross-entropy loss for phase rate coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Classes].
        targets (Tensor): Ground truth labels of shape [Batch].
        phase (int): Number of steps to sum over.
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, phase, weight=None):
        super().__init__()
        self.__name__ = "ce_phase_loss"
        self.phase = phase
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
        steps, batch, classes = spk_out.shape
        assert steps % self.phase == 0, "Number of steps must be divisible by the phase."
        spk_out = spk_out.view(steps//self.phase, -1, batch, classes)
        spk_out = spk_out.sum(dim=0)
        return spk_out.sum(dim=0)
    
    def spike_code(self, spk_out: Tensor):
        # [Batch]
        return self._encode(spk_out).argmax(dim=1)


class ce_isi_loss:
    """
    NOT WORKING
    Cross-entropy loss for inter-spike interval (ISI) coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Classes].
        targets (Tensor): Ground truth labels of shape [Batch].
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, weight=None):
        super().__init__()
        self.__name__ = "ce_isi_loss"
        self.weight = weight

    def __call__(self, spk_out: Tensor, targets: Tensor):
        log_softmax_fn = LogSoftmax(dim=-1)
        loss_fn = NLLLoss(weight=self.weight)
        # Compute rates by summing spikes over time steps
        print("-----------------------------")
        rates = self._encode(spk_out)
        rates.requires_grad_(True)
        # Convert rates to probabilities
        probabilities = log_softmax_fn(rates) 
        # Compute cross-entropy loss
        loss = loss_fn(probabilities, targets)
        return loss
    
    def _encode(self, spk_out: Tensor):
        steps, batch, classes = spk_out.shape
        # Reshape to [Batch, Classes, Steps]
        spk_out = spk_out.permute(1, 2, 0)
        new_out = zeros(batch, classes, device=spk_out.device)
        for b in range(batch):
            for c in range(classes):
                spike_times = nonzero(spk_out[b, c]).squeeze()
                if spike_times.numel() > 1:
                    isis = spike_times[1:] - spike_times[:-1]
                    new_out[b, c] = isis.max()
        return new_out
    
    def spike_code(self, spk_out: Tensor):
        # [Batch]
        return self._encode(spk_out).argmax(dim=1)  


class ce_cisi_loss:
    """
    Cross-entropy loss for consistent inter-spike interval (CISI) coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Classes].
        targets (Tensor): Ground truth labels of shape [Batch].
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, weight=None):
        super().__init__()
        self.__name__ = "ce_cisi_loss"
        self.weight = weight

    def __call__(self, spk_out: Tensor, targets: Tensor):
        log_softmax_fn = LogSoftmax(dim=-1)
        loss_fn = NLLLoss(weight=self.weight)
        # Compute rates by summing spikes over time steps
        rates = self._encode(spk_out)
        rates.requires_grad_(True)
        # Convert rates to probabilities
        probabilities = log_softmax_fn(rates) 
        # Compute cross-entropy loss
        loss = loss_fn(probabilities, targets)
        return loss
    
    def _encode(self, spk_out: Tensor):
        steps, batch, classes = spk_out.shape
        # Reshape to [Batch, Classes, Steps]
        new_out = zeros(batch, classes, device=spk_out.device)
        for b in range(batch):
            for c in range(classes):
                # compute the distrance between 1's for the steps dimension
                indices = nonzero(spk_out[:, b, c]).squeeze()
                if indices.numel() >= 2:
                    intervals = indices[1:] - indices[:-1]
                else:
                    intervals = tensor([])
                # count the number of unique values
                unique_values, counts = intervals.unique(return_counts=True)
                # select the most frequent value
                if counts.numel() > 0:
                    freq_value = unique_values[counts.argmax()]
                    new_out[b, c] = freq_value
        return new_out

    
    def spike_code(self, spk_out: Tensor):
        # [Batch]
        return self._encode(spk_out).argmin(dim=1)  



if __name__ == '__main__':
    # Test the loss functions
    """
    import torch
    spk_out = torch.tensor([
        [[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
    ], dtype=torch.float32)
    print(spk_out.shape)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    loss_fn = ce_isi_loss()
    loss = loss_fn(spk_out, targets)
    print(loss)
    """
    import torch

    tensor = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0])
    # compute the distrance between 1's
    intervals = torch.nonzero(tensor).squeeze()[1:] - torch.nonzero(tensor).squeeze()[:-1]
    print(intervals)
    # count the number of unique values
    print(intervals.unique(return_counts=True))
    # SELECT THE ONE WHICH IS THE MOST FREQUENT
    print(intervals.unique(return_counts=True)[0][intervals.unique(return_counts=True)[1].argmax()])

