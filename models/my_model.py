"""
MyModel model.  (c) 2021 Georgia Tech

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

from pickle import TRUE
import torch
import torch.nn as nn


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        ### Convolutional block1 ###
        self.bc1 = nn.Sequential(
          nn.Conv2d(3,32, kernel_size=3, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(inplace= TRUE),
          nn.Conv2d(32, 32, kernel_size=3, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2) 
        )


        ### Convolutional block2 ###
        self.bc2 = nn.Sequential(
          nn.Conv2d(32,64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace= TRUE),
          nn.Conv2d(64, 64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2) 
        )


        ### Global averaging pooling ###
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ### Fully connected layer (Linear)
        self.fc = nn.Linear(64, 10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = self.bc1(x)
        x = self.bc2(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        outs = self.fc(x)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
