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
        self.s = 3
        self.t = 4

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

    def inverse_transforms(self, layer, z, x):
        """
        x has dimension 2
        y has dimension 2
        """
        nn_input = torch.cat(z,x)
        nn_masked_input, z_masked, z_masked_prime = utils.mask_inputs(nn_input, x, layer)
        x_backward = (z_masked_prime-self.t(nn_masked_input))*np.exp(-self.s(nn_masked_input))+z_masked
        return x_backward




if __name__ == '__main__':
    """
    # run normalizing flow directly for testng
    """