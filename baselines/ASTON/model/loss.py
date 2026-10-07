import torch
from torch import nn

class SoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss"""

    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label):
        self.reduction = 'none'
        unweighted_loss = super(SoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = unweighted_loss.mean(dim=1)
        return weighted_loss