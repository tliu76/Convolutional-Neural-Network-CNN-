"""
2d Convolution Module.  (c) 2021 Georgia Tech

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
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, C, H, W = x.shape 
        C_out = self.out_channels
        K = self.kernel_size
        S = self.stride
        P = self.padding
        # Compute output spatial size
        H_out = 1 + (H + 2 * P - K) // S
        W_out = 1 + (W + 2 * P - K) // S

        # Pad input (only height and width dimensions)
        x_padded = np.pad( x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
        # Initialize output
        out = np.zeros((N, C_out, H_out, W_out))
        # Perform convolution
        for n in range(N):              # each image
            for f in range(C_out):          # each filter
                for i in range(H_out):  # vertical sliding
                    for j in range(W_out):  # horizontal sliding
                        # Compute window boundaries
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        # Extract local region (all input channels)
                        window = x_padded[n, :, h_start:h_end, w_start:w_end]

                        # Convolution = elementwise multiply + sum + bias
                        out[n, f, i, j] = np.sum( window * self.weight[f]) + self.bias[f]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        self.cache = (x, x_padded, H_out, W_out)
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x, x_padded, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape
        C_out = self.out_channels
        K = self.kernel_size
        S = self.stride
        P = self.padding

        # Initialize gradients
        dx_padded = np.zeros_like(x_padded)
        dw = np.zeros_like(self.weight)
        db = np.zeros_like(self.bias)

        # Bias gradient: sum over batch and spatial dimensions
        for f in range(C_out):
          db[f] = np.sum(dout[:, f, :, :])

        # Compute gradients
        for n in range(N):
            for f in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):

                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        window = x_padded[n, :, h_start:h_end, w_start:w_end]

                        # Gradient for weights
                        dw[f] += window * dout[n, f, i, j]

                        # Gradient for input
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += \
                            self.weight[f] * dout[n, f, i, j]

        # Remove padding from dx
        if P > 0:
          dx = dx_padded[:, :, P:-P, P:-P]
        else:
          dx = dx_padded

        # Store gradients
        self.dx = dx
        self.dw = dw
        self.db = db
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
