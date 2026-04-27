"""
SGD Optimizer.  (c) 2021 Georgia Tech

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
import numpy as np
from ._base_optimizer import _BaseOptimizer

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from sgd.py!")

class SGD(_BaseOptimizer): # Define SGD optimizer class inheriting from base optimizer
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg) 
        self.momentum = momentum  
        self.velocity = {} # Dictionary to store velocity for each module


    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model) # Apply regularization to gradients before updating weights

        for idx, m in enumerate(model.modules): # Iterate through each layer/module in the model
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                if  idx not in self.velocity: 
                    self.velocity[idx] = {} # Create a dictionary entry for this layer
                if 'w_v' not in self.velocity[idx]:  # If weight velocity has not been initialized
                    self.velocity[idx]['w_v'] = np.zeros_like(m.weight)  # Initialize velocity with zeros (same shape as weight)
                w_v = self.velocity[idx]['w_v']   # Retrieve previous velocity for weights
                w_v = self.momentum * w_v - self.learning_rate * m.dw  
                # Compute new velocity:
                # v_t = beta * v_(t-1) - eta * gradient
                m.weight += w_v # Update weights using the new velocity  w=w+w_v
                self.velocity[idx]['w_v'] = w_v # Store updated velocity back into dictionary



                
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                if 'b_v' not in self.velocity[idx]:  # If bias velocity has not been initialized
                    self.velocity[idx]['b_v'] = np.zeros_like(m.bias)  ## initialize bias velocity with zeros
                b_v = self.velocity[idx]['b_v'] # Retrieve previous bias velocity
                b_v = self.momentum * b_v - self.learning_rate * m.db  
                 # Compute new bias velocity:
                 # v_t = beta * v_(t-1) - eta * gradient
                m.bias += b_v  # Update bias using the new bias velocity  b=b+vt​
                self.velocity[idx]['b_v'] = b_v ## store updated bias velocity
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
