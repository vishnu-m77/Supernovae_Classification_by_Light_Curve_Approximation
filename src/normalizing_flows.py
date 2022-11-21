# imports
import torch
import numpy as np
from torch import nn

class utils():
    def __init__(self) -> None:
        pass

    def mask_inputs(nn_input, var, layer):
        """
        May need to modify multiplications depending on dimensions
        """
        if (layer % 2 == 0):
            nn_masked_input = nn_input*[1,0,1,1]
            var_masked = var*[1,0]
            var_masked_prime = var*[0,1]
        else:
            nn_masked_input = nn_input*[0,1,1,1]
            var_masked = var*[0,1]
            var_masked_prime = var*[1,0]
        return nn_masked_input, var_masked, var_masked_prime



class RealNVPtransforms():

    def __init__(self):
        super(RealNVPtransforms, self).__init__()
        self.s = 3 # must change
        self.t = 4 # must change

    def forward_transform(self, layer, x, y):
        """
        x has dimension 2
        y has dimension 2
        """
        nn_input = torch.cat(x,y)
        nn_masked_input, x_masked, x_masked_prime = utils.mask_inputs(nn_input, x, layer)
        x_forward = x_masked_prime*(np.exp(self.s(nn_masked_input))+self.t(nn_masked_input))+x_masked
        """
        need to compute determinant
        """
        return x_forward

    def inverse_transform(self, layer, z, x):
        """
        x has dimension 2
        y has dimension 2
        """
        nn_input = torch.cat(z,x)
        nn_masked_input, z_masked, z_masked_prime = utils.mask_inputs(nn_input, x, layer)
        x_backward = (z_masked_prime-self.t(nn_masked_input))*np.exp(-self.s(nn_masked_input))+z_masked
        return x_backward

class NormalizingFlowsBase(RealNVPtransforms):
    def __init__(self, num_layers):
        super(NormalizingFlowsBase, self).__init__()
        self.num_layers = num_layers

    def full_forward_transform(self, x, y):
        for layer in range(self.num_layers):
            x = self.forward_transform(layer, x, y)
        z = x
        return x

    def full_backward_transform(self, z, x):
        for layer in range(self.num_layers):
            z = self.inverse_transform(layer, z, x)
        y = z
        return y
    
    def sample_data(self, x):
        z = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample()
        y = self.full_backward_transform(z,x)
        print(y[0]) # flux



if __name__ == '__main__':
    """
    # run normalizing flow directly for testng
    """