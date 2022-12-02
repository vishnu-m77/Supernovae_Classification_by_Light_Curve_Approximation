import argparse
import os
import torch
import json
import matplotlib.pyplot as plt
# import classification
import CNN
import CNNMetrics
from matplotlib.backends.backend_pdf import PdfPages
import PyPDF2


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
    y_test, y_test_pred = CNN.classification()
    report = CNNMetrics.gen_report( y_test, y_test_pred)

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
