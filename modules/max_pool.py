"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        ## Extract input dimensions:
        N, C, H, W = x.shape ### tuple unpacking
        # Pooling parameters
        K = self.kernel_size  
        S = self.stride 
        # Compute output spatial dimensions
        H_out = (H-K)//S + 1 
        W_out = (W-K)//S + 1 

        # Initialize output tensor
        out = np.zeros((N, C, H_out, W_out))
        # Slide pooling window
        for n in range(N): # each image in batch
           for c in range(C):  # each channel
               for i in range(H_out): # vertical 
                  for j in range(W_out): #horizontal

                      ### compute window boundaries
                      h_start  = i * S
                      h_end = h_start + K
                      w_start = j * S
                      w_end = w_start + K

                      # Extract local window
                      window = x[n,c, h_start:h_end, w_start:w_end]

                      # Take maximum value in the window
                      out[n,c,i,j] = np.max(window)
        


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        dx = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out): 
                      ### compute window boundaries
                      h_start  = i * S
                      h_end = h_start + K
                      w_start = j * S
                      w_end = w_start + K

                      # Extract pooling window
                      window = x[n, c, h_start:h_end, w_start:w_end]
                      # Find index of maximum value in this window
                      max_idx = np.argmax(window)
                      # Convert flat index to 2D coordinates
                      max_pos = np.unravel_index(max_idx, window.shape)
                      # Only the max element receives gradient
                      dx[n, c, h_start + max_pos[0], w_start + max_pos[1]] += dout[n, c, i, j]

        # Store gradient
        self.dx = dx                      


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
