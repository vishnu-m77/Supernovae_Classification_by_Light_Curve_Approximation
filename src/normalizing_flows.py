# imports
import torch
import numpy as np
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.plot_utils import plotLightCurve
import matplotlib.pyplot as plt
import json
import os
from joblib import Parallel, delayed

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

def augmentation(timestamps, wavelengths=np.array([np.log10(4741.64), np.log10(6173.23)]), num_timestamps=256):
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
            
        directory = os.path.dirname(__file__)
            
        with open(directory + "/objects.json", 'w') as f:
            for object in objects:
                json.dump(object, f)
                json.dump("\n", f)

        flux_pred = []
        aug_timestamps = []

        # df_obj = df.loc[df['object_id'] == 'ZTF20adaduxg'] # select data for object=object_name
        # flux_pred, aug_timestamp = self.one_object_pred(df_obj)
        # flux_predes.append(flux_pred)
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

        outputs = Parallel(n_jobs=-1)(delayed(self.one_object_pred)(df.loc[df['object_id'] == object], object) for object in objects)

        flux_pred = [obj[0] for obj in outputs]
        # print("PRED FLUXES:")
        # print(flux_pred)
        aug_timestamps = [obj[1] for obj in outputs]
        flux = [obj[2] for obj in outputs]
        flux_err = [obj[3] for obj in outputs]
        flux_err_pred = [obj[4] for obj in outputs]

        for object in objects:
            # print(object)
            df_obj = df.loc[df['object_id'] == object] # select data for object=object_name
            true_value = int(df_obj['obj_type'].to_numpy()[0])
            y_test.append(true_value)
            # np.asarray .values
            # flux.extend(df_obj['flux'].values)
            # flux_err.extend(df_obj['flux_err'].values)
        
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        flux_err_pred = np.array(flux_err_pred)
        
        # print(flux)
        # print(flux_err)
        
        # flux_list = flux.tolist()
        # flux_err_list = flux_err.tolist()
        
        # print(flux.shape[0])
        # print(flux_err.shape[0])
        
        # with open(directory + "/flux.json", 'w') as f:
        #     json.dump(flux_list, f)
        
        # with open(directory + "/flux_err.json", 'w') as f:
        #     json.dump(flux_err_list, f)
            
        self.flux = flux
        self.flux_err = flux_err
        self.flux_err_pred = flux_err_pred
        
        for obj in flux_pred:
            mid = int(len(obj)/2)
            temp = []
            temp.append(obj[:mid])
            temp.append(obj[mid: ])
            X_test.append(temp)
        X_test = np.array(X_test)
        # print("X_TEST")
        # print(X_test)
        y_test = np.array(y_test)
        
        X_test_list = X_test.tolist()
        y_test_list = y_test.tolist()
        
        # with open(directory + "/X_test.json", 'w') as f:
        #     json.dump(X_test_list, f)
        
        # with open(directory + "/y_test.json", 'w') as f:
        #     json.dump(y_test_list, f)

        X_test = np.array((X_test - X_test.mean()) / X_test.std(), dtype = np.float32)
        X_test = torch.from_numpy(np.array(X_test)).to(torch.float32)
        y_test = torch.from_numpy(np.array(y_test)).to(torch.float32)
        
        self.X_test = X_test
        self.y_test = y_test
        self.flux_pred = flux_pred
        self.aug_timestamps = aug_timestamps
    
    def one_object_pred(self, df_obj, obj_name):
        timestamp = np.asarray(df_obj['mjd']) # timestamp
        passbands = np.asarray(df_obj['passband']) # define passband
        # process passband to log(wavelength) [wavelegnth_arr]
        wavelength_arr = [] 
        for pb in passbands:
            if pb==0:
                wavelength_arr.append(np.log10(4741.64))
            elif pb==1:
                wavelength_arr.append(np.log10(6173.23))
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

        NF = NormalizingFlowsBase(num_layers = 8)
        
        optimizer = torch.optim.Adam(NF.parameters(), self.lr) 
        
        X = StandardScaler().fit_transform(X)
        X = torch.from_numpy(X).to(torch.float32)
        y_transform = StandardScaler()
        processed_flux = y_transform.fit_transform(y)
        y = torch.from_numpy(processed_flux).to(torch.float32)
        loss_vals = []
        for epoch in range(self.num_epochs):
            _ , log_likelihood = NF.full_forward_transform(X,y)
            loss = -log_likelihood
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            loss_vals.append(float(loss))
            # if ((epoch+1) % self.display_epochs == 0): 
            #     print ('Epoch [{}/{}]\tTrain Loss : {:.4f}'.format(epoch+1, self.num_epochs, loss))
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
        # print("\nSampling...\n")
        X_pred, aug_timestamps = augmentation(timestamps=timestamp, num_timestamps = self.num_ts)
        if (X_pred!=None):
            X = StandardScaler().fit_transform(X_pred)
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
            # if (i+1)%32 == 0:
            #     print("For observation {0}, predicted flux is : {1}, [{2}/512]".format(X_pred[i], flux_pred[i], i+1))

        # df_obj_pb_0 = df_obj
        # df_obj_pb_1 = df_obj
        # df_obj_pb_0 = df_obj_pb_0.loc[df_obj['passband']==0]
        # df_obj_pb_1 = df_obj_pb_1.loc[df_obj['passband']==1]
        # pb0_t = df_obj_pb_0['mjd']
        # pb0_flux = df_obj_pb_0['flux']
        # pb1_t = df_obj_pb_1['mjd']
        # pb1_flux = df_obj_pb_1['flux']
        # plt.plot(pb0_t, pb0_flux, 'o', label='DATA: PB=g', color='b')
        # plt.plot(pb1_t, pb1_flux, 'o', label='DATA: PB=r', color='g')

        # plt.plot(aug_timestamps, flux_pred[:self.num_ts], label='NF: PB=g', color='b')
        # plt.plot(aug_timestamps, flux_pred[-self.num_ts:], label='NF: PB=r', color='g')


        # plt.title("Flux against timestamp for " + obj_name)
        # plt.xlabel("timestamp")
        # plt.ylabel("flux")
        # plt.legend(loc="upper right")
        # # num = np.random.randint(-1000,1000)
        # plt.savefig('plots/Light_Flux_NF_'+ obj_name +'.png')
        # plt.clf()
    
        passband2name = {0: 'g', 1: 'r'}
        plotLightCurve(obj_name, df_obj, flux_pred, aug_timestamps, passband2name)
        output = []
        output.append(flux_pred)
        output.append(list(aug_timestamps))
        output.append(flux)
        output.append(flux_err)
        output.append(flux_err_pred)
        
        print("Predicted object " + obj_name)

        return output

