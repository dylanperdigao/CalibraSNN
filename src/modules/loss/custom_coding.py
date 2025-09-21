from torch import Tensor
from torch.nn import LogSoftmax, NLLLoss

class ce_hybrid_loss:
    """
    Cross-entropy loss for inter-spike interval (ISI) coding.

    Args:
        spk_out (Tensor): Spiking output of shape [Steps, Batch, Classes].
        targets (Tensor): Ground truth labels of shape [Batch].
        phase (int): Number of steps to sum over.
        weight (Tensor, optional): Class weights for imbalanced datasets.
    """
    def __init__(self, weight=None):
        self.__name__ = "ce_hybrid_loss"
        self.phase = 2
        self.alpha = 0.5
        self.weight = weight

    def __call__(self, spk_out: Tensor, targets: Tensor):
        log_softmax_fn = LogSoftmax(dim=-1)
        loss_fn = NLLLoss(weight=self.weight)
        # Compute rates by summing spikes over time steps
        rates1 = self._encode1(spk_out)
        probabilities1 = log_softmax_fn(rates1)  
        loss1 = loss_fn(probabilities1, targets)
        rates2 = self._encode2(spk_out)
        probabilities2 = log_softmax_fn(rates2)
        loss2 = loss_fn(probabilities2, targets)
        return self.alpha * loss1 + (1 - self.alpha) * loss2
    
    def _encode1(self, spk_out: Tensor):
        # [Batch, Classes]
        return spk_out.sum(dim=0)
    
    def _encode2(self, spk_out: Tensor):
        # [Batch, Classes]
        steps, batch, classes = spk_out.shape
        assert steps % self.phase == 0, "Number of steps must be divisible by the phase."
        spk_out = spk_out.view(steps//2, -1, batch, classes)
        spk_out = spk_out.sum(dim=0)
        return spk_out.sum(dim=0)
    
    def spike_code(self, spk_out: Tensor):
        # [Batch]
        return (self.alpha*self._encode1(spk_out)+(1-self.alpha)*self._encode2(spk_out)).argmax(dim=1)
    
if __name__ == '__main__':
    # Test the loss functions
    import torch
    spk_out = torch.tensor([
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[1, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[1, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    ], dtype=torch.float32)
    print(spk_out.shape)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    loss_fn = ce_hybrid_loss()
    loss = loss_fn(spk_out, targets)
    print(loss)