"""
Loss function and its components SEP-LSTM benchmark. SEP-LSTM, in 
contrast to all other implementations, is only trained for one-step ahead 
prediction and not for suffix generation. The metrics for next activity 
and next timestamp prediction are stored here. 
"""

import torch
import torch.nn as nn

#####################################
##    Individual loss functions    ##
#####################################

class ActCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(ActCrossEntropyLoss, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        self.cross_entropy_crit = nn.CrossEntropyLoss()

        
    def forward(self, inputs, targets):
        """Compute the CrossEntropyLoss of the next activity prediction 
        head.

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the unnormalized logits for each 
            activity class. Shape (batch_size, num_classes) 
            and dtype torch.float32.
        targets : torch.Tensor
            The activity labels, containing the indices. Shape 
            (batch_size, window_size), dtype torch.int64. 

        Returns
        -------
        loss: torch.Tensor
            The cross entropy loss for the activity prediction head. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Slice out only the first activity label 
        targets = targets[:, 0] # (batch_size,)

        # Compute loss 
        loss = self.cross_entropy_crit(inputs, targets) # scalar tensor

        return loss
    
class MeanAbsoluteErrorLoss(nn.Module):
    def __init__(self):
        super(MeanAbsoluteErrorLoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss in which the 
        target values of -100.0, corresponding to padded event tokens, 
        are ignored / masked and hence do not contribute to the input 
        gradient. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for the  
            Time Till Next Event (TTNE) target. Shape (batch_size, 1) and 
            dtype torch.float32.
        targets : torch.Tensor
            The continuous time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. 

        Returns
        -------
        loss: torch.Tensor
            The MAE loss for one of the time prediction heads. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """

        # Reshape 
        inputs = inputs[:, 0] # (batch_size,)

        # Slice out only the first TTNE label 
        targets = targets[:, 0, 0] # (batch_size,)

        absolute_errors = torch.abs(inputs-targets) # (batch_size,)

        # Compute masked loss 
        return torch.mean(absolute_errors) # scalar tensor
    

class MultiOutputLoss(nn.Module):
    def __init__(self, num_classes):
        super(MultiOutputLoss, self).__init__()
        self.cat_loss_fn = ActCrossEntropyLoss(num_classes)
        self.cont_loss_fn_ttne = MeanAbsoluteErrorLoss()
        self.concurrent_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels):
        """Labels: [ttne_labels, act_labels, concurrent_labels]
        Outputs: (act_probs, ttne_pred, concurrent_pred)
        """
        # Loss next activity prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[1])

        # Loss TTNE prediction
        cont_loss = self.cont_loss_fn_ttne(outputs[1], labels[0])

        # Loss concurrent prediction (BCE; logits in, binary float target)
        conc_loss = self.concurrent_loss_fn(outputs[2].squeeze(-1), labels[2])

        loss = cat_loss + cont_loss + conc_loss

        return loss, cat_loss.item(), cont_loss.item()