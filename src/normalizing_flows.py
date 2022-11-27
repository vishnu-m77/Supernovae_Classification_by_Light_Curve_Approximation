# imports
import torch
import numpy as np
from torch import nn
import torch.functional as func
import pandas as pd

class utils():
    def __init__(self) -> None:
        pass

    def mask_inputs(nn_input, var, layer):
        """
        May need to modify multiplications depending on dimensions
        """
        if (layer % 2 == 0):
            nn_mask_mat = torch.from_numpy(np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])).to(torch.float32)
            var_mask_mat = torch.from_numpy(np.array([[1.,0.],[0.,0.]])).to(torch.float32)
            mask_prime = torch.tensor([0.,1.]).to(torch.float32)
            #nn_masked_input = nn_input*[:,[1,0,1,1]]
            nn_masked_input = torch.matmul(nn_input, nn_mask_mat)
            #var_masked = var*[1,0]
            var_masked = torch.matmul(var, var_mask_mat)
            #var_masked_prime = var*[0,1]
            var_masked_prime = torch.matmul(var, torch.eye(2)-var_mask_mat)
        else:
            nn_mask_mat = torch.from_numpy(np.array([[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])).to(torch.float32)
            var_mask_mat = torch.from_numpy(np.array([[0.,0.],[0.,1.]])).to(torch.float32)
            mask_prime = torch.tensor([1.,0.]).to(torch.float32)
            nn_masked_input = torch.matmul(nn_input, nn_mask_mat)
            var_masked = torch.matmul(var, var_mask_mat)
            var_masked_prime = torch.matmul(var, torch.eye(2)-var_mask_mat)
        return nn_masked_input, var_masked, var_masked_prime,mask_prime # torch.eye(2)-var_mask_mat is mask_prime


class Net(nn.Module):
    def __init__(self, hidden_units=10):
        super(Net, self).__init__()
        self.input_units = 4
        self.hidden_units = hidden_units # need to make it a hyper-parameter
        self.output_units = 2

        self.fc1 = nn.Linear(self.input_units, self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, self.output_units)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        #print("x in FORQARD IS {0}".format(x))
        y = self.fc2(h)
        return y


class RealNVPtransforms(Net):

    def __init__(self):
        super(RealNVPtransforms, self).__init__()
        self.s = Net(hidden_units=10)
        self.t = Net(hidden_units=10)

    def forward_transform(self, layer, x, y):
        """
        Forward transform of flux data y = [flux,flux_err] to latent z conditioned on x = [time_stamp, passband]
        """
        nn_input = torch.cat((y,x),dim=1)
        nn_masked_input, y_masked, y_masked_prime, mask_prime = utils.mask_inputs(nn_input, y, layer)
        s_forward = self.s.forward(nn_masked_input)
        #print("s_forward is {0}".format(s_forward))
        #y_forward = y_masked_prime*(torch.exp(s_forward)+self.t.forward(nn_masked_input))+y_masked
        y_forward = y_masked_prime*(torch.exp(s_forward))+mask_prime*self.t.forward(nn_masked_input)+y_masked # masking fixed :: can be improved
        """
        need to compute determinant
        """
        det_comp = torch.exp(torch.sum(mask_prime*torch.exp(s_forward), dim=1))
        #print("det_comp is : {0}".format(det_comp))
        return y_forward, det_comp # use this s_forard to compute the determinant

    def inverse_transform(self, layer, z, x):
        """
        Inverse transform of latent z to flux data y = [flux,flux_err] conditioned on x = [time_stamp, passband]
        """
        nn_input = torch.cat((z,x), dim=0)
        nn_masked_input, z_masked, z_masked_prime, mask_prime = utils.mask_inputs(nn_input, x, layer)
        x_backward = (z_masked_prime-self.t.forward(nn_masked_input))*torch.exp(-self.s.forward(nn_masked_input))+z_masked
        return x_backward

class NormalizingFlowsBase(RealNVPtransforms):
    def __init__(self, num_layers):
        super(NormalizingFlowsBase, self).__init__()
        self.num_layers = num_layers
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    def full_forward_transform(self, x, y):
        det = 1
        for layer in range(self.num_layers):
            y, det = self.forward_transform(layer, x, y)
            log_likelihood = torch.log(det) + self.prior.log_prob(y)
        z = y
        return z, log_likelihood.mean()
    
    # def compute_transform_determinant(self, s_list):


    def full_backward_transform(self, z, x):
        for layer in range(self.num_layers):
            z = self.inverse_transform(layer, z, x)
            print("y in layer {0} is {1}".format(layer, z))
        y = z
        return y
    
    def sample_data(self, x):
        z = torch.from_numpy(np.asarray(self.prior.sample()))
        print(z)
        y = self.full_backward_transform(z,x)
        print(y[0]) # flux
        return y



if __name__ == '__main__':
    """
    # run normalizing flow directly for testng
    """
    data_dir = 'data/ANTARES_NEW.csv'
    df = pd.read_csv(data_dir)
    #print(df['object_id']=='ZTF21abwxaht')
    object_name = 'ZTF20aahbamv'
    df_obj = df.loc[df['object_id']==object_name]
    timestamp = np.asarray(df_obj['mjd'])
    passbands = np.asarray(df_obj['passband'])
    wavelength_arr = []
    for pb in passbands:
        if pb==0:
            wavelength_arr.append(np.log10(3751.36))
        elif pb==1:
            wavelength_arr.append(np.log10(4741.64))
        else:
            print("Passband invalid")
    flux = np.asarray(df['flux'])
    flux_err = np.asarray(df['flux_err'])

    X = []
    y = []
    for i in range(len(passbands)):
        X.append(np.array([timestamp[i], wavelength_arr[i]]))
        y.append(np.array([flux[i], flux_err[i]]))
    X = torch.from_numpy(np.array(X)).to(torch.float32)
    y = torch.from_numpy(np.array(y)).to(torch.float32)
    #print(df[0])
    RealNVP = RealNVPtransforms()
    x_out, s_out = RealNVP.forward_transform(1,X, y)

    NF = NormalizingFlowsBase(num_layers=8)
    out, log_likelihood = NF.full_forward_transform(X,y)
    optimizer = torch.optim.Adam(NF.parameters(), lr=0.001)
    # print("S FORWARDS")
    #print("log_like is {0}".format(log_likelihood))
    num_epochs = 50
    for epoch in range(num_epochs):
        _ , loss = NF.full_forward_transform(X,y)
        loss = -loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        print("Epoch: {0} and Loss: {1}".format(epoch, loss))
    inp = torch.from_numpy(np.asarray([58871., np.log10(3751.36)])).to(torch.float32)
    print(inp)
    out = NF.sample_data(inp)
    print(out)

    