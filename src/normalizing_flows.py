# imports
import torch
import numpy as np
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

"""
utility functions
"""

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

def augmentation(timestamps, wavelengths=np.array([np.log10(3751.36), np.log10(4741.64)]), num_timestamps=128):
    """
    augments the data for flux interpolation
    """
    augmented_timestamps = np.linspace(min(timestamps), max(timestamps), num=num_timestamps)
    X_pred = []
    for wavelength in wavelengths:
        for timestamp in augmented_timestamps:
            X_pred.append([timestamp, wavelength])
    return X_pred, augmented_timestamps

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
        nn_mask_mat, var_mask, mask_prime = mask_inputs(nn_input, layer)
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
        nn_mask_mat, var_mask, mask_prime = mask_inputs(nn_input, layer)
        #x_backward = (z-self.t.forward(nn_masked_input))df = pd.read_csv(data_dir) # define pandas datadrame for while data*torch.exp(-self.s.forward(nn_masked_input))*mask_prime+z_masked
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
    def __init__(self, data_dir, param):
        super(FitNF, self).__init__()
        num_objects = param["num_objects"]

        self.lr = param["lr"]
        self.num_epochs = param["num_epochs"]
        self.display_epochs = param["display_epochs"]
        self.num_samples = param["num_samples"]
        self.num_ts = param["num_ts"]

        df = pd.read_csv(data_dir) # define pandas datadrame for while data

        objects = df['object_id'].unique()
        np.random.shuffle(objects)

        if num_objects < len(objects):
            objects = objects[:num_objects]

        pred_fluxes = []
        aug_timestamps = []

        # df_obj = df.loc[df['object_id'] == 'ZTF20adaduxg'] # select data for object=object_name
        # pred_flux, aug_timestamp = self.one_object_pred(df_obj)
        # pred_fluxes.append(pred_flux)
        # aug_timestamps.append(aug_timestamp)
        X_test = []
        y_test = []

        # df.loc[df['obj_type'] == 'SN Ia', 'obj_type'] = 1
        # df.loc[df['obj_type'] != 1, 'obj_type'] = 0
        df.loc[df['obj_type'] == 'SN Ia', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Ia-91T', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Ia-pec', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Iax', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Ia-91bg', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Ia-CSM', 'obj_type'] = 1
        df.loc[df['obj_type'] != 1, 'obj_type'] = 0

        for object in objects:
            print(object)
            df_obj = df.loc[df['object_id'] == object] # select data for object=object_name
            true_value = int(df_obj['obj_type'].to_numpy()[0])
            y_test.append(true_value)
            pred_flux, aug_timestamp = self.one_object_pred(df_obj)
            pred_fluxes.append(pred_flux)
            # pred_flux.reshape((2, self.num_ts))
            mid = int(len(pred_flux)/2)
            temp = []
            temp.append(pred_flux[:mid])
            temp.append(pred_flux[mid: ])
            X_test.append(temp)
            aug_timestamps.append(aug_timestamp)
        
        X_test = np.array(X_test)
        X_test = torch.from_numpy(np.array(X_test)).to(torch.float32)
        y_test = np.array(y_test)
        y_test = torch.from_numpy(np.array(y_test)).to(torch.float32)

        self.X_test = X_test
        self.y_test = y_test
        self.pred_fluxes = pred_fluxes
        self.aug_timestamps = aug_timestamps
    
    def one_object_pred(self, df_obj):
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

        self.NF = NormalizingFlowsBase(num_layers = 8)
        
        optimizer = torch.optim.Adam(self.NF.parameters(), self.lr) 
        
        X = StandardScaler().fit_transform(self.X)
        X = torch.from_numpy(X).to(torch.float32)
        y_transform = StandardScaler()
        processed_flux = y_transform.fit_transform(self.y)
        self.y = torch.from_numpy(processed_flux).to(torch.float32)
        loss_vals = []
        for epoch in range(self.num_epochs):
            _ , log_likelihood = self.NF.full_forward_transform(X,self.y)
            loss = -log_likelihood
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            loss_vals.append(float(loss))
            if ((epoch+1) % self.display_epochs == 0): 
                print ('Epoch [{}/{}]\tTrain Loss : {:.4f}'.format(epoch+1, self.num_epochs, loss))
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
        print("\nSampling...\n")
        X_pred, aug_timestamps = augmentation(timestamps=self.timestamp, num_timestamps = self.num_ts)
        if (X_pred!=None):
            X = StandardScaler().fit_transform(X_pred)
            X = torch.from_numpy(X).to(torch.float32)
        pred_flux = []
        for i in range(len(X_pred)): # length of x_pred (256*2)
            
            flux_approx = []
            for j in range(self.num_samples):
                flux_approx.append(y_transform.inverse_transform(np.expand_dims(self.NF.sample_data(X[i]).detach().numpy(), axis=0))[0][0])
            mean_flux = sum(flux_approx)/len(flux_approx)
            pred_flux.append(mean_flux)
            if (i+1)%32 == 0:
                print("For observation {0}, predicted flux is : {1}, [{2}/512]".format(X_pred[i], pred_flux[i], i+1))
        return pred_flux, list(aug_timestamps)

