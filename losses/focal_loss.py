"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from focal_loss.py!")


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return: tensor containing the weights for each class
    """

    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    per_cls_weights = None
        # Convert list to float tensor
    cls_num = torch.tensor(cls_num_list, dtype=torch.float32)

    # Compute effective number: 1 - beta^n
    effective_num = 1.0 - torch.pow(beta, cls_num)

    # Compute raw weights
    per_cls_weights = (1.0 - beta) / effective_num

    # Normalize so average weight = 1
    per_cls_weights = per_cls_weights / per_cls_weights.sum() * len(cls_num)

    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        # reshape target for gather
        target = target.view(-1, 1)

        # log softmax for numerical stability
        log_probs = F.log_softmax(input, dim=1)

        # log(p_t)
        log_pt = log_probs.gather(1, target) # Pluck out the log-probability of the *actual* ground truth class for each sample.

        # p_t
        # Convert log-probabilities back to standard probabilities (p_t)
        pt = log_pt.exp()

        # focal term (1 - p_t)^gamma
        # Compute the Focal Loss modulating factor: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma


        # If class weights were provided during initialization, apply them dynamically

        if self.weight is not None:
            weight = self.weight.to(input.device)
            alpha = weight.gather(0, target.view(-1)) # Use the target indices to look up the specific weight (alpha) for each sample's class
            alpha = alpha.view(-1, 1)
            loss = -alpha * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        # Reduce to a scalar to match the docstring
        loss = loss.mean()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
