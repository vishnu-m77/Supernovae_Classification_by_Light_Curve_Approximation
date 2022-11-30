import argparse
import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
# import classification
from src.CNN import classification as CNN
import src.normalizing_flows as NF


if __name__ == '__main__':
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Classification of Supernovae Light Curves by NF approximation')
    
    # Default values of parameters are defined
    parser.add_argument('--param', default = 'param/param.json', help='file containing hyperparameters')
    parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
    
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    
    # Load data
    # Run Normalizing Flows to obtain the approximate light curve

    nf_params = param["NF"]
    data_dir = 'data/ANTARES_NEW.csv'
    nf = NF.FitNF(data_dir, nf_params)
    pred_flux = nf.pred_fluxes
    pred_flux = np.array(pred_flux)
    pred_flux = torch.from_numpy(np.array(pred_flux))
    # print(pred_flux.size())
    # print(pred_flux)
    aug_timestamp = nf.aug_timestamps[0]

    # print("for passband 0 flux is {0}\n".format(pred_flux[:35]))
    # print("for passband 1 flux is {0}\n".format(pred_flux[-35:]))
    # print("augmented timestamp is {0}".format(aug_timestamp))
    # pred_flux = nf.pred_fluxes[1]
    # aug_timestamp = nf.aug_timestamps[1]
    # print("for passband 0 flux is {0}\n".format(pred_flux[:35]))
    # print("for passband 1 flux is {0}\n".format(pred_flux[-35:]))
    # print("augmented timestamp is {0}".format(aug_timestamp))
    # Input heat map into CNN for binary classification
    directory = os.path.dirname(__file__)
    img_file = "data/images.json"
    lbl_file = "data/labels.json"

    cnn_params = param["CNN"]
    # nf = 1
    CNN(directory, img_file, lbl_file, cnn_params, nf)
    # Regression and Performance metrics
    # Visualization and Report
    
    print("Hello")