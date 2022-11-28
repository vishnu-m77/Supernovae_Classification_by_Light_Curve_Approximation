# imports
import torch
import numpy as np
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class utils():
    """
    class containing utility functions
    """
    def __init__(self) -> None:
        pass

    def mask_inputs(nn_input, layer):
        """
        This is used to mask variables in the flow. When layer is even,
        variables of the normalizing flow are masked by [0.,1.] and when
        layer is odd, variable are masked by [1.,0.]
        mask_prime is the reverse masking of each var_mask
        """
        if (layer % 2 != 0):
            nn_masked_mat = torch.from_numpy(np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])).to(torch.float32)
            var_mask = torch.tensor([1.,0.]).to(torch.float32)
            mask_prime = torch.tensor([0.,1.]).to(torch.float32)
        else:
            nn_masked_mat = torch.from_numpy(np.array([[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])).to(torch.float32)
            var_mask = torch.tensor([0.,1.]).to(torch.float32)
            mask_prime = torch.tensor([1.,0.]).to(torch.float32)
        return nn_masked_mat, var_mask,mask_prime

class Net(nn.Module):
    """
    Contains neural network architecture for 
    implementing functions s (scale) and t (translation)
    """
    def __init__(self, hidden_units=10):
        super(Net, self).__init__()
        self.input_units = 4
        self.hidden_units = hidden_units
        self.output_units = 2

        self.fc1 = nn.Linear(self.input_units, self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, self.output_units)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        y = self.fc2(h)
        return y

class RealNVPtransforms(Net):
    """
    This class contains the functions which are used for the realNVP implementation
    of normalizing flows.
    """
    def __init__(self):
        super(RealNVPtransforms, self).__init__()
        self.s = Net(hidden_units=10)
        self.t = Net(hidden_units=10)

    def forward_transform(self, layer, x, y):
        """
        Forward transform of flux data y = [flux,flux_err] to latent z conditioned on x = [time_stamp, passband]
        """
        nn_input = torch.cat((y,x),dim=1)
        nn_mask_mat, var_mask, mask_prime = utils.mask_inputs(nn_input, layer)
        nn_masked_input = torch.matmul(nn_input, nn_mask_mat)
        s_forward = self.s.forward(nn_masked_input)
        t_forward = self.t.forward(nn_masked_input)
        y_forward = (y*torch.exp(s_forward)+t_forward)*mask_prime+y*var_mask
        log_det = torch.sum(s_forward*mask_prime, dim=1) # log determinant
        return y_forward, log_det

    def inverse_transform(self, layer, z, x):
        """
        Inverse transform of latent z to flux data y = [flux,flux_err] conditioned on x = [time_stamp, passband]
        """
        nn_input = torch.cat((z,x), dim=0)
        nn_mask_mat, var_mask, mask_prime = utils.mask_inputs(nn_input, layer)
        #x_backward = (z-self.t.forward(nn_masked_input))*torch.exp(-self.s.forward(nn_masked_input))*mask_prime+z_masked
        nn_masked_input = torch.matmul(nn_input, nn_mask_mat)
        s_forward = self.s.forward(nn_masked_input)
        t_forward = self.t.forward(nn_masked_input)
        z_backward = (z - t_forward)*torch.exp(-s_forward)*mask_prime+z*var_mask
        return z_backward

class NormalizingFlowsBase(RealNVPtransforms):
    def __init__(self, num_layers):
        super(NormalizingFlowsBase, self).__init__()
        self.num_layers = num_layers
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    def full_forward_transform(self, x, y):
        log_likelihood = 0
        for layer in range(self.num_layers):
            y, det = self.forward_transform(layer, x, y)
            log_likelihood = log_likelihood + det
        prior_prob = self.prior.log_prob(y)
        log_likelihood = log_likelihood + prior_prob
        z = y
        return z, log_likelihood.mean()

    def full_backward_transform(self, z, x):
        for layer in range(self.num_layers):
            z = self.inverse_transform(layer, z, x)
        y = z
        return y
    
    def sample_data(self, x):
        z = torch.from_numpy(np.asarray(self.prior.sample()))
        y = self.full_backward_transform(z,x)
        return y

class FitNF():
    def __init__(self, obj_name):
        super(FitNF, self).__init__()
        self.object_name = obj_name # string containing object name
        data_dir = 'data/ANTARES_NEW.csv' # define data directory
        df = pd.read_csv(data_dir) # define pandas datadrame for while data
        df_obj = df.loc[df['object_id']==self.object_name] # select data for object=object_name
        self.timestamp = np.asarray(df_obj['mjd']) # timestamp
        passbands = np.asarray(df_obj['passband']) # define passband
        # process passband to log(wavelength) [wavelegnth_arr]
        self.wavelength_arr = [] 
        for pb in passbands:
            if pb==0:
                self.wavelength_arr.append(np.log10(3751.36))
            elif pb==1:
                self.wavelength_arr.append(np.log10(4741.64))
            else:
                print("Passband invalid")
        self.flux = np.asarray(df_obj['flux'])
        self.flux_err = np.asarray(df_obj['flux_err'])

        self.X = []
        self.y = []
        for i in range(len(self.flux)):
            self.X.append(np.array([self.timestamp[i], self.wavelength_arr[i]]))
            self.y.append(np.array([self.flux[i], self.flux_err[i]]))
        self.X = torch.from_numpy(np.array(self.X)).to(torch.float32)
        self.y = torch.from_numpy(np.array(self.y)).to(torch.float32)

        self.NF = NormalizingFlowsBase(num_layers=8) # make hyperparam

    def predict_mean_flux(self, X_pred = None):
        optimizer = torch.optim.Adam(self.NF.parameters(), lr=0.0001) # make hyper param
        num_epochs = 8000 # make hyperparam
        X = StandardScaler().fit_transform(self.X)
        X = torch.from_numpy(X).to(torch.float32)
        for epoch in range(num_epochs):
            _ , log_likelihood = self.NF.full_forward_transform(X,self.y)
            loss = -log_likelihood
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            if ((epoch+1) % 200 ==0): # make hyperparam
                print("Epoch: {0} and Loss: {1}".format(epoch+1, loss))
        # prediction
        """
        format of X_pred = {
            [[timestamp_1, pb_1],
            [timestamp_2, pb_1],
            [timestamp_3, pb_1]...
            [timestamp_256, pb_1],
            [timestamp_1, pb_2],
            [timestamp_2, pb_2]...
            [timestamp_256, pb_2]]
        }
        """
        if (X_pred!=None):
            X = StandardScaler().fit_transform(X_pred)
            X = torch.from_numpy(X).to(torch.float32)
        pred_flux = []
        orig_flux = []
        for i in range(len(self.flux)): # length of x_pred (256*2)
            if (i+1)%5==0: # remove this 
                num_samples = 1500 # hyperparam
                flux_approx = []
                for j in range(num_samples):
                    flux_approx.append(self.NF.sample_data(X[i])[0])
                mean_flux = sum(flux_approx)/len(flux_approx)
                pred_flux.append(mean_flux)
                orig_flux.append(self.flux[i])
                print("Original flux is {0} and predicted flux is {1} for flux {2}".format(self.flux[i],mean_flux,i+1))
        return pred_flux, orig_flux

if __name__ == '__main__':
    """
    # run normalizing flow directly for testng
    """

    object_name =  'ZTF20adaduxg'
    pred_flux, orig_flux = FitNF(object_name).predict_mean_flux()
    # print("pred_flux is {0}".format(pred_flux))
    # print("orig_flux is {0}".format(orig_flux))
    """
    data_dir = 'data/ANTARES_NEW.csv'
    df = pd.read_csv(data_dir)
    #print(df['object_id']=='ZTF21abwxaht')
    object_name =  'ZTF18accnmri'#'ZTF21abvicne'#'' #'ZTF20adadlqv' #'ZTF20adaduxg' #-> works VERY well #'ZTF20aahbamv'-> paper_object
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
    flux = np.asarray(df_obj['flux'])
    flux_err = np.asarray(df_obj['flux_err'])

    X = []
    y = []
    for i in range(len(passbands)):
        X.append(np.array([timestamp[i], wavelength_arr[i]]))
        y.append(np.array([flux[i], flux_err[i]]))
    X = torch.from_numpy(np.array(X)).to(torch.float32)
    y = torch.from_numpy(np.array(y)).to(torch.float32)

# rewrite training loop
    NF = NormalizingFlowsBase(num_layers=8)
    optimizer = torch.optim.Adam(NF.parameters(), lr=0.0001)
    num_epochs = 8000
    X = StandardScaler().fit_transform(X)
    X = torch.from_numpy(X).to(torch.float32)
    for epoch in range(num_epochs):
        _ , loss = NF.full_forward_transform(X,y)
        loss = -loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if ((epoch+1) % 200 ==0):
            print("Epoch: {0} and Loss: {1}".format(epoch+1, loss))
    
    pred_flux = []
    orig_flux = []
    for i in range(len(flux)):
        if (i+1)%5==0:
            inp = X[i] # 119, 160 works
            num_samples = 1500
            flux_approx = []
            for j in range(num_samples):
                flux_approx.append(NF.sample_data(inp)[0])
            mean_flux = sum(flux_approx)/len(flux_approx)
            pred_flux.append(mean_flux)
            orig_flux.append(flux[i])
            print("Original flux is {0} and predicted flux is {1} for flux {2}".format(flux[i],mean_flux,i+1))
    original_flux = orig_flux
    """