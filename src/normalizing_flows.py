# imports
import torch
import numpy as np
from torch import nn
import torch.functional as func

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


class Net(nn.Module):
    def __init__(self, hidden_units=10):
        super(Net, self).__init__()
        self.input_units = 4
        self.hidden_units = hidden_units # need to make it a hyper-parameter
        self.output_units = 2

        self.fc1 = nn.Linear(self.input_units, self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, self.output_units)

    def forward(self):
        h = torch.tanh(self.fc1)
        y = nn.Linear(h)
        return y


class RealNVPtransforms():

    def __init__(self):
        super(RealNVPtransforms, self).__init__()
        self.s = Net(hidden_units=10)
        self.t = Net(hidden_units=10)

    def forward_transform(self, layer, x, y):
        """
        Forward transform of flux data y = [flux,flux_err] to latent z conditioned on x = [time_stamp, passband]
        """
        nn_input = torch.cat(y,x)
        nn_masked_input, x_masked, x_masked_prime = utils.mask_inputs(nn_input, x, layer)
        s_forward = self.s.forward(nn_masked_input)
        x_forward = x_masked_prime*(np.exp(s_forward)+self.t.forward(nn_masked_input))+x_masked
        """
        need to compute determinant
        """
        return x_forward, s_forward # use this s_forard to compute the determinant

    def inverse_transform(self, layer, z, x):
        """
        Inverse transform of latent z to flux data y = [flux,flux_err] conditioned on x = [time_stamp, passband]
        """
        nn_input = torch.cat(z,x)
        nn_masked_input, z_masked, z_masked_prime = utils.mask_inputs(nn_input, x, layer)
        x_backward = (z_masked_prime-self.t.forward(nn_masked_input))*np.exp(-self.s.forward(nn_masked_input))+z_masked
        return x_backward

class NormalizingFlowsBase(RealNVPtransforms):
    def __init__(self, num_layers):
        super(NormalizingFlowsBase, self).__init__()
        self.num_layers = num_layers

    def full_forward_transform(self, x, y):
        for layer in range(self.num_layers):
            x = self.forward_transform(layer, x, y)
        z = x
        return z

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