import argparse
import os
import torch
import json
import matplotlib.pyplot as plt
# import classification
from CNN import classification as CNN
import normalizing_flows


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
    # Input heat map into CNN for binary classification
    CNN()
    # Regression and Performance metrics
    # Visualization and Report
    
    print("Hello")