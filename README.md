# Supernovae Classification by Light Curve Approximation

### Collaborators:
- Vishnu Mugundan

- Maimouna Sow

- Sehmimul Hoque

- Siddhardha Penmetsa

## Install Dependencies

- numpy
- matplotlib
- torch
- fulu
- pandas
- sklearn
- joblib
- torchvision

Code to install requirements:
```sh
pip3 install -r requirements.txt
```

## Running `main.py`

To run `main.py`, use

```sh
python3 main.py
```

Get more information about the other arguments by running

```sh
python3 main.py --help
```

An example call of the program is

```sh
python main.py --param param/param.json -nf 10 -v -p -o -s
```

1. `-nf num_objects` runs Normalizing Flows to approximate `num_objects` light curves, and runs the CNN on the objects. In the above example, the code will produce 10 objects using Normalizing Flows, and run CNN on the 10 objects. `num_objects` must be at least 10. Be warned that even with multiprocessing implemented, the code takes a while to run.
2. `-v` increases the verbosity of the program.
3. `-p` produces plots for the light curves of the objects generated by Normalizing Flows.
4. `-s` shuffles the objects generated by Normalizing Flows.
5. `-o` option to be used when running `-nf`. Trains the CNN on the data generated from 600 objects and predicts on the objects generated by the normalizing flows run.

## Remarks

It is suggested that you train the CNN using the data of 600 objects already obtained by us and then run NF for about 15-50 objects depending on the computing resources. NF takes significant amount of time to compile, and can be tracked in either verbose mode or in `report.txt`. The program trains and samples from the NF in parallel. Running `-nf 24` takes around 20 minutes. It took 15 hours to get the output for 600 objects.

It is recommended to use the option `-o` while running `-nf` as the CNN predicts better when there are more objects. Furthermore, for every 10 objects generated by the NF, the CNN trains on 6, validates on 2, and only predicts on 2 objects.

## Hyperparameters in `param/param.json`

### NF
- num_samples: number of samples taken from the Multivariate Gaussian prior for inverse transforms
- lr: Learning rate of the Adam Optimizer
- n_epochs: Number of epochs for updating the parameters of the unknown model
- display_epochs: Number of epochs for printing the loss

### CNN
- n_epochs: Number of epochs for updating the parameters of the unknown model
- display_epochs: Number of epochs for printing the loss
- weight_decay: regularization technique applied to the weights of CNN
- lr: Learning rate of the gradient descent

## Data
### Files
- `main.py` : main running script
- `requirements.txt` : requirements file
- `src/normalizing_flows.py` : normalizing flows module
- `src/CNN.py` : CNN module
- `src/plot_utls.py` : plot utilities for normalizing flows
-  `src/cnn_metrics.py` : cnn metrics utilities
- `src/nf_metrics.py` : normalizing flows metrics utilities
- `data/ANTARES_NEW.csv` : dataset containing relevant data for flux interpolation and flux type classification of 1870 objects
- `data/images.json` : flux and flux error for 600 objects using the normalzing flows algorithm (CNN can be trained on this data)
- `data/labels.json` : expected labels of the 600 objects in `images.json`
- `param/param,json` : hyperparameters
- `plots`: contains plots of several flux interpolations using normalizing flows

### Outputs
- `nfmetrics.csv` : contains metrics of flux approximation using normalizing flows
- `report.txt` : final report containing metrics for normalizing flows, training epochs of the CNN and metrics of CNN
- `src/X_matrix.json` : output of normalizing flows
- `src/y_vector.json` : expected labels of objects in `src/X_matrix.json`
- `nf_run_plots`: generated plots of several flux interpolations using normalizing flows

