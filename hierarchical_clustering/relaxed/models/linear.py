import torch.nn as nn

from hierarchical_clustering.relaxed.models.model import HypHCModel


class HypHCLinear(HypHCModel):
    """ Hyperbolic linear model for hierarchical clustering. """

    def __init__(self, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-3,
                 sequence_length=128, device='cpu'):
        super(HypHCLinear, self).__init__(temperature=temperature, init_size=init_size, max_scale=max_scale)
        self.device = device
        self.linear = nn.Linear(sequence_length, rank)

    def encode(self, triple_ids=None, sequences=None):
        e = self.linear(sequences)
        return e
