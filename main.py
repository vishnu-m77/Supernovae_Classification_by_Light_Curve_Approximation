import argparse
import os
import pandas as pd
import json
from src.plot_utils import plotLightCurve
from src.CNN import classification as CNN
import src.normalizing_flows as NF
import src.metrics as met
import CNNMetrics
import sys



if __name__ == '__main__':
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Classification of Supernovae Light Curves by NF approximation')
    
    # Default values of parameters are defined
    parser.add_argument('--param', default = 'param/param.json', help='file containing hyperparameters')
    parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
    parser.add_argument('-p', '--plot', help="plotting frequency", type=int, choices=[0, 1, 2])
    
    args = parser.parse_args()
    verbose = args.verbose
    plot = args.plot

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    
    # Load data
    nf_params = param["NF"]
    data_dir = 'data/ANTARES_NEW.csv'

    # Run Normalizing Flows to obtain the approximate light curve
    nf = NF.FitNF(data_dir, nf_params)
    flux_pred = nf.flux_pred
    flux_err = nf.flux_err
    flux = nf.flux
    flux_err_pred = nf.flux_err_pred
    flux_pred_metrics = nf.flux_pred_metrics
    flux_err_pred_metrics = nf.flux_err_pred_metrics

    metrics = met.generate_NF_report(flux, flux_pred_metrics, flux_err, flux_err_pred_metrics)
    metrics.to_csv('nfmetrics.csv')

    passband2name = {0: 'g', 1: 'r'}
    df = nf.df # Accessing the dataframe from NF object
    objects = nf.objects # Accessing unique objects from data

    # Regression and Performance metrics
    for i in range(len(objects)):
        obj_name = objects[i]
        df_obj = df.loc[df['object_id'] == obj_name] # select data for object=object_name
        plotLightCurve(obj_name, df_obj, flux_pred[i], nf.aug_timestamps[i], passband2name)
       
    directory = os.path.dirname(__file__)
    img_file = "data\X_test.json"
    lbl_file = "data\y_test.json"

    cnn_params = param["CNN"]
    
    # Run binary classification
    y_test, y_test_pred = CNN(directory, img_file, lbl_file, cnn_params, nf)

    # Visualization and Report
    report = CNNMetrics.gen_report(y_test, y_test_pred)
    print(report)

    original_stdout = sys.stdout
    with open('out.txt', 'a') as f:
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout
    
    print("Hello")
