import argparse
import os
import torch
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
# import classification
from src.plot_utils import plotLightCurve
from src.CNN import classification as CNN
import src.normalizing_flows as NF
import src.metrics as met
import CNNMetrics
from matplotlib.backends.backend_pdf import PdfPages



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
    flux_pred = nf.flux_pred
    flux_err = nf.flux_err
    flux = nf.flux
    flux_err_pred = nf.flux_err_pred
    flux_pred_metrics = nf.flux_pred_metrics
    flux_err_pred_metrics = nf.flux_err_pred_metrics
    #metrics = met.regression_quality_metrics_report(flux, flux_pred_metrics, flux_err, flux_err_pred_metrics)
    metrics = met.generate_NF_report(flux, flux_pred_metrics, flux_err, flux_err_pred_metrics)
    print(metrics)
    metrics.to_csv('nfmetrics.csv')
  
       


    # fig , ax = plt.subplots(figsize=(5, 7))
    # ax.axis('tight')
    # ax.axis('off')
    # the_table = ax.table(cellText=metrics.values, colLabels=metrics.columns, loc = 'center')

    # pp = PdfPages("NFMetrics.pdf")
    # pp.savefig(fig, bbox_inches='tight')
    # pp.close()

   
    
    
    flux_pred = np.array(flux_pred)
    flux_pred = torch.from_numpy(np.array(flux_pred))
    print(flux_pred.size())
    print(flux_pred)
    aug_timestamp = nf.aug_timestamps[0]

    # print("for passband 0 flux is {0}\n".format(flux_pred[:35]))
    # print("for passband 1 flux is {0}\n".format(flux_pred[-35:]))
    # print("augmented timestamp is {0}".format(aug_timestamp))
    # flux_pred = nf.flux_pred[1]
    # aug_timestamp = nf.aug_timestamps[1]
    # print("for passband 0 flux is {0}\n".format(flux_pred[:35]))
    # print("for passband 1 flux is {0}\n".format(flux_pred[-35:]))
    # print("augmented timestamp is {0}".format(aug_timestamp))
    #Input heat map into CNN for binary classification

    passband2name = {0: 'g', 1: 'r'}
    df = nf.df # Accessing the dataframe from NF object
    objects = nf.objects # Accessing unique objects from data

    for i in range(len(objects)):
        obj_name = objects[i]
        df_obj = df.loc[df['object_id'] == obj_name] # select data for object=object_name
        plotLightCurve(obj_name, df_obj, flux_pred[i], nf.aug_timestamps[i], passband2name)
       
    directory = os.path.dirname(__file__)
    img_file = "data\X_test.json"
    lbl_file = "data\y_test.json"

    cnn_params = param["CNN"]
    # nf = 1
    
    y_test, y_test_pred = CNN(directory, img_file, lbl_file, cnn_params, nf)
    report = CNNMetrics.gen_report(y_test, y_test_pred)

    print(report)

    fig , ax = plt.subplots(figsize=(5, 7))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=report.values, colLabels=report.columns, loc = 'center')

    pp = PdfPages("Metrics.pdf")
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    # Regression and Performance metrics
    # Visualization and Report
    
    print("Hello")
