# imports
import torch
import numpy as np
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
from joblib import Parallel, delayed
import sys

"""
utility functions for normalizing flows
"""

def mask_inputs(layer):
    """
    This is used to mask variables in the flow. When layer is even,
    variables of the normalizing flow are masked by [1.,0.] and when
    layer is odd, variable are masked by [0.,1.]
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

def augmentation(timestamps, wavelengths=np.array([np.log10(4741.64), np.log10(6173.23)]), num_timestamps=256):
    """
    Augments the data for flux interpolation
    Divided the timestamps into num_timestamps equally spaced datapoints.
    Flux is then approximated at each timestamp using the inverse transform of
    normalising flows. 
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
    4 input unit are used for flux, flux_error, timetamp and transformed passband
    Note that timetamp and transformed passband are conitional variables. This is why
    we have only 2 outputs for functions s and t. On these 2 outputs for s an t,
    we apply the proper masking depening on which layer we are in. 
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
    of normalizing flows. The function s and t are designed be neural networks.
    This class contains the forward and inverse transform of each layer.
    We then call forward transforms 8 times to stack multiple layers together for training.
    We then call backward transforms 8 times to stack multiple layers together for sampling.
    """
    def __init__(self):
        super(RealNVPtransforms, self).__init__()
        self.s = Net(hidden_units=10)
        self.t = Net(hidden_units=10)

    def forward_transform(self, layer, x, y):
        """
        Forward transform of flux data y = [flux,flux_err] to 
        latent z conditioned on x = [time_stamp, transformed passband]. 
        As x is the conditional variable, it does not get modified in the forward transform
        """
        nn_input = torch.cat((y,x),dim=1)
        nn_mask_mat, var_mask, mask_prime = mask_inputs(layer) # nn_input contains 4 elements (2 for y and then 2 for x)
        nn_masked_input = torch.matmul(nn_input, nn_mask_mat) # mask first or second component of y depending on layer
        s_forward = self.s.forward(nn_masked_input)
        t_forward = self.t.forward(nn_masked_input)
        y_forward = (y*torch.exp(s_forward)+t_forward)*mask_prime+y*var_mask # forward transform
        log_det = torch.sum(s_forward*mask_prime, dim=1) # log determinant (for finding determinant)
        return y_forward, log_det

    def inverse_transform(self, layer, z, x):
        """
        Inverse transform of latent z to flux data y = [flux,flux_err] 
        conditioned on x = [time_stamp, trnormed passband]
        As x is the conditional variable, it does not get modified in the forward transform
        """
        nn_input = torch.cat((z,x), dim=0) # nn_input contains 4 elements (2 for z and then 2 for x)
        nn_mask_mat, var_mask, mask_prime = mask_inputs(layer) # mask first or second component of z depending on layer
        nn_masked_input = torch.matmul(nn_input, nn_mask_mat)
        s_forward = self.s.forward(nn_masked_input)
        t_forward = self.t.forward(nn_masked_input)
        z_backward = (z - t_forward)*torch.exp(-s_forward)*mask_prime+z*var_mask # backward transform
        return z_backward

class NormalizingFlowsBase(RealNVPtransforms):
    """
    This class contains the full forward and full inverse transform functions. 
    full_forward_transform is for training and full_backward_transform is for sampling.
    sampling function is also present. We sample from the prior where out prior is just the
    standard Multivariate Gaussian Distribution
    """
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
        log_likelihood = log_likelihood + prior_prob # for loss function
        z = y
        return z, log_likelihood.mean()

    def full_backward_transform(self, z, x):
        for layer in range(self.num_layers):
            z = self.inverse_transform(layer, z, x)
        y = z
        return y
    
    def sample_data(self, x):
        z = torch.from_numpy(np.asarray(self.prior.sample())) # latent sample
        y = self.full_backward_transform(z,x) # y[0] is predicted flux
        return y

class FitNF():
    """
    This is the main class for normalizing flows which finds predicted flux for
    arbitrary objects. It contains one_object_pred fucntion which predicts flux for only 
    one object.
    """
    def __init__(self, data_dir, shuffle, num_objects, param, report_file, verbose = 1):
        super(FitNF, self).__init__()

        self.lr = param["lr"]
        self.num_epochs = param["num_epochs"]
        self.display_epochs = param["display_epochs"]
        self.num_samples = param["num_samples"]
        self.num_ts = 256 # The augmented timestamps for flux interpolation

        df = pd.read_csv(data_dir) # define pandas datadrame for while data

        objects = df['object_id'].unique()
        if shuffle:
            np.random.shuffle(objects)

        if num_objects < len(objects):
            objects = objects[:num_objects]

        flux_pred = []
        aug_timestamps = []

        df.loc[df['obj_type'] == 'SN Ia', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Ia-91T', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Ia-pec', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Iax', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Ia-91bg', 'obj_type'] = 1
        df.loc[df['obj_type'] == 'SN Ia-CSM', 'obj_type'] = 1
        df.loc[df['obj_type'] != 1, 'obj_type'] = 0

        outputs = Parallel(n_jobs=-1)(delayed(self.one_object_pred)(df.loc[df['object_id'] == object], object, report_file, verbose) for object in objects)

        flux_pred = [obj[0] for obj in outputs]
        aug_timestamps = [obj[1] for obj in outputs]
        flux = [obj[2] for obj in outputs]
        flux_err = [obj[3] for obj in outputs]
        flux_err_pred = [obj[4] for obj in outputs]
        flux_pred_metrics = [obj[5] for obj in outputs]
        flux_err_pred_metrics = [obj[6] for obj in outputs]
        
        X_matrix = []
        y_vector = []

        for object in objects:
            df_obj = df.loc[df['object_id'] == object] # select data for object=object_name
            true_value = int(df_obj['obj_type'].to_numpy()[0])
            y_vector.append(true_value)
        
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        flux_err_pred = np.array(flux_err_pred)
        flux_pred_metrics = np.array(flux_pred_metrics)
        flux_err_pred_metrics = np.array(flux_err_pred_metrics)
            
        self.flux = flux
        self.flux_err = flux_err
        self.flux_err_pred = flux_err_pred
        self.flux_pred_metrics = flux_pred_metrics
        self.flux_err_pred_metrics = flux_err_pred_metrics
        
        for obj in flux_pred:
            mid = int(len(obj)/2)
            temp = []
            temp.append(obj[:mid])
            temp.append(obj[mid: ])
            X_matrix.append(temp)
        X_matrix = np.array(X_matrix)
        y_vector = np.array(y_vector)
        
        X_matrix_list = X_matrix.tolist()
        y_vector_list = y_vector.tolist()
        
        directory = os.path.dirname(__file__)
        with open(os.path.join(directory, "X_matrix.json"), 'w') as f:
            json.dump(X_matrix_list, f)
        
        with open(os.path.join(directory, "y_vector.json"), 'w') as f:
            json.dump(y_vector_list, f)

        X_matrix = np.array((X_matrix - X_matrix.mean()) / X_matrix.std(), dtype = np.float32)
        X_matrix = torch.from_numpy(np.array(X_matrix)).to(torch.float32)
        y_vector = torch.from_numpy(np.array(y_vector)).to(torch.float32)
        
        self.X_matrix = X_matrix
        self.y_vector = y_vector
        self.flux_pred = flux_pred
        self.aug_timestamps = aug_timestamps

        self.df = df
        self.objects = objects
    
    def one_object_pred(self, df_obj, obj_name, report_file, verbose):
        timestamp = np.asarray(df_obj['mjd']) # timestamp
        passbands = np.asarray(df_obj['passband']) # passband
        # process passband to log(wavelength) [wavelegnth_arr]. This is processed passband
        wavelength_arr = [] 
        for pb in passbands:
            if pb==0:
                wavelength_arr.append(np.log10(4741.64)) # from the paper we are working with
            elif pb==1:
                wavelength_arr.append(np.log10(6173.23)) # from the paper we are working with
            else:
                print("Passband invalid")
        flux = np.asarray(df_obj['flux'])
        flux_err = np.asarray(df_obj['flux_err'])

        X = []
        y = []
        for i in range(len(flux)):
            X.append(np.array([timestamp[i], wavelength_arr[i]]))
            y.append(np.array([flux[i], flux_err[i]]))
        X = torch.from_numpy(np.array(X)).to(torch.float32)
        y = torch.from_numpy(np.array(y)).to(torch.float32)

        NF = NormalizingFlowsBase(num_layers = 8) # Initialize NormalizingFlowsBase with 8 layers
        
        optimizer = torch.optim.Adam(NF.parameters(), self.lr) 
        untransformed_X = X # original timestamp and processed passband
        # Standardize X and y
        X_transform = StandardScaler()
        X = X_transform.fit_transform(X)
        X = torch.from_numpy(X).to(torch.float32)
        y_transform = StandardScaler()
        processed_flux = y_transform.fit_transform(y)
        y = torch.from_numpy(processed_flux).to(torch.float32)
        loss_vals = []
        # training loop
        for epoch in range(self.num_epochs):
            _ , log_likelihood = NF.full_forward_transform(X,y)
            loss = -log_likelihood
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            loss_vals.append(float(loss))
            if ((epoch+1) % self.display_epochs == 0 and verbose):
                print ('Train Loss : {:.4f} for object {}'.format(loss, obj_name))
        
        # prediction
        # In the augmentation(...) function below, we augment the timestamps for each passband
        # In the end we get an X_pred array which has the format as given in the block comment below.
        # pb_1 is one passband and pb_2 is the other passband
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
        if verbose:
            print('\nSampling for object {} \n'.format(obj_name))
        X_pred, aug_timestamps = augmentation(timestamps=timestamp, num_timestamps = self.num_ts)
        """
        First we train and sample with normalizing flows for original datapoints (unaugmented).
        We only do this to generate metrics on how well our model worked. Thus we find metrics using
        the predicted flux and the original flux at each datapoint of the original data.
        """
        flux_pred_metrics = []
        flux_err_pred_metrics = []
        num_datapoints = len(X)
        for i in range(num_datapoints):          
            flux_approx = []
            for j in range(self.num_samples):
                flux_approx.append(y_transform.inverse_transform(np.expand_dims(NF.sample_data(X[i]).detach().numpy(), axis=0))[0][0])
            flux_approx = np.array(flux_approx)
            mean_flux = sum(flux_approx)/len(flux_approx) # flux_approx.std(axis=0)
            std_flux = flux_approx.std(axis=0)
            flux_pred_metrics.append(mean_flux)
            flux_err_pred_metrics.append(std_flux)
            # if (i+1)%32 == 0:
            #    print("For datapoint {0}, predicted flux is : {1}, [{2}/{3}]".format(untransformed_X[i],flux_pred_metrics[i], i+1, num_datapoints))
        """
        Now we train and sample with normalizing flows for the augmented data.
        We do this to generate more data at augmented X. This generated data is then used for
        flux interpolation and to generate flux plots
        """
        if (X_pred!=None):
            X = X_transform.transform(X_pred)
            X = torch.from_numpy(X).to(torch.float32)
        flux_pred = []
        flux_err_pred = []
        for i in range(len(X_pred)): # length of x_pred (256*2)
            
            flux_approx = []
            for j in range(self.num_samples):
                flux_approx.append(y_transform.inverse_transform(np.expand_dims(NF.sample_data(X[i]).detach().numpy(), axis=0))[0][0])
            flux_approx = np.array(flux_approx)
            mean_flux = sum(flux_approx)/len(flux_approx) # flux_approx.std(axis=0)
            std_flux = flux_approx.std(axis=0)
            flux_pred.append(mean_flux)
            flux_err_pred.append(std_flux)

        output = [] # return an output object containing variables needed from normalizing flows for metrics and augmentation
        output.append(flux_pred)
        output.append(list(aug_timestamps))
        output.append(flux)
        output.append(flux_err)
        output.append(flux_err_pred)
        output.append(flux_pred_metrics)
        output.append(flux_err_pred_metrics)
        
        if verbose:
            print("Predicted object " + obj_name)
        original_stdout = sys.stdout
        
        with open(report_file, 'a') as f:
            sys.stdout = f
            print("Predicted object " + obj_name)
            sys.stdout = original_stdout

        return output

