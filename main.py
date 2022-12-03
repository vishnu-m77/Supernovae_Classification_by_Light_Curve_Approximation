import argparse
import os
import pandas as pd
import json
from src.plot_utils import plotLightCurve
from src.CNN import classification as CNN
import src.normalizing_flows as NF
import src.nf_metrics as nf_met
import src.cnn_metrics as cnn_met
import sys
import numpy as np

def plots(nf):
    passband2name = {0: 'g', 1: 'r'}
    df = nf.df # Accessing the dataframe from NF object
    objects = nf.objects # Accessing unique objects from data

    # Regression and Performance metrics
    for i in range(len(objects)):
        obj_name = objects[i]
        df_obj = df.loc[df['object_id'] == obj_name] # select data for object=object_name
        plotLightCurve(obj_name, df_obj, nf.flux_pred[i], nf.aug_timestamps[i], passband2name)

# Run binary classification
def run_CNN(cnn_params, report_file, nf = 0, run_nf = 0):
    
    X_matrix = []
    y_vector = []
    
    if run_nf:
        
        X_matrix = nf.X_matrix
        y_vector = nf.y_vector
        
    else:
        
        X_file = os.path.join("src", "X_matrix.json")
        y_file = os.path.join("src", "y_vector.json")
        with open(X_file, 'r') as f:
            X_matrix = json.load(f)
        with open(y_file, 'r') as f:
            y_vector = json.load(f)
            
    X_matrix = np.array(X_matrix)
    y_vector = np.array(y_vector)
    # normalize input data
    X_matrix = np.array((X_matrix - X_matrix.mean()) / X_matrix.std(), dtype = np.float32)
        
    y_test, y_test_pred = CNN(cnn_params, X_matrix, y_vector, report_file)
        
    # Report
    report = cnn_met.gen_report(y_test, y_test_pred)
    print(report)

    original_stdout = sys.stdout
    with open(report_file, 'a') as f:
        sys.stdout = f
        print(report)
        sys.stdout = original_stdout

if __name__ == '__main__':
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Classification of Supernovae Light Curves by NF approximation')
    
    # Default values of parameters are defined
    parser.add_argument('--param', default = 'param/param.json', help='file containing hyperparameters')
    parser.add_argument('-nf', '--nf', help="input number of objects generated by normalizing flows", type = int, default = 0)
    parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
    parser.add_argument('-p', '--plot', help="generate plots", action="store_true" )
    
    args = parser.parse_args()
    num_objects = args.nf
    verbose = args.verbose
    plot = args.plot
    
    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    
    # Creates a report file
    report_file = 'report.txt'
    # report_file = os.path.join(os.path.abspath, report_file)
    
    if os.path.exists(report_file):
        os.remove(report_file)
    f = open(report_file, 'w')
    f.close()
    
    cnn_params = param["CNN"]
    
    # Runs CNN as default
    if (num_objects == 0):
        run_CNN(cnn_params, report_file)
    else:
        # Load data
        nf_params = param["NF"]
        data_dir = 'data/ANTARES_NEW.csv'

        # Run Normalizing Flows to obtain the approximate light curve
        nf = NF.FitNF(data_dir, num_objects, nf_params, report_file, verbose)

        metrics = nf_met.generate_NF_report(nf.flux, nf.flux_pred_metrics, nf.flux_err, nf.flux_err_pred_metrics)
        metrics.to_csv('nfmetrics.csv')
        
        original_stdout = sys.stdout
        with open(report_file, 'a') as f:
            sys.stdout = f
            print(metrics)
            sys.stdout = original_stdout
        
        if plot:
            plots(nf)
            
        run_CNN(cnn_params, report_file, nf, run_nf = 1)
