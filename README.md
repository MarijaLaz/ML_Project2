# Project 2 : Machine Learning-based Estimation of Cardiac Contractility from Peripheral Pressure Waveform

This is a project realised for the Machine Learning Course [CS-433]. The project was accomplished with collaboration with the Laboratory of Hemodynamics and Cardiovascular Technology, EPFL, Lausanne. Regarding the stake that risk factor assessment for Cardiovascular Diseases is, we were working on predicting the arterial end-systolic elastance (Ees) using brachial blood pressure waveforms. Our work is an extended work from a previous project from our host laboratory. The main goal was to create a model that won't be using the Ejection Fraction (EF) as a feature for predicting the Ees, but will be able to give good predictions.

## Folder Structure

```
.
├── Data
│   ├── brachial_BP_waveforms_HEMODB.csv     # Raw brachial waveforms
│   ├── HEMODB.csv                           # Data set containing all the features 
│   ├── radial_BP_waveforms_HEMODB.csv       # Raw radial waveforms (not used in our implementaion)
│   ├── true_ees.csv                         # True labels for the Ees
│   ├── u1.csv                               # Data set with features: brSBP, brDBP, HR, cfPWV
│   └── u2.csv                               # Data set with features: brSBP, brDBP, HR, cfPWV, EF
├── ML_Project_2_Report.pdf                  # The report
├── notebooks                                # We executed our code in notebooks
│   ├── Exploitation_brachial_waveform.ipynb # Notebook containing the brachial waveforms analysis
│   ├── GridSearchTest_MinMax.ipynb          # Notebook containing the reproduction part of the project
│   ├── NeuralNetworks_2Channels.ipynb       # Notebook containing the training of the CNN model using 2 Channels
│   ├── NeuralNetworks_3Channels.ipynb       # Notebook containing the training of the CNN model using 3 Channels
│   ├── NeuralNetworks.ipynb                 # Notebook containing the training of the CNN model using 1 Channel
│   └── Create_CSV.ipynb                     # Notebook for generating the u1.csv, u2.csv, true_ees.csv files 
│                                              (they are already generated in Data)
├── README.md
└── scripts
    ├── BrachialWaveform_Helpers.py          # Helping functions used for the brachial waveforms analysis
    ├── GridSearch_helpers.py                # Helping functions used for the reproduction and tuning of hyperparameters
    ├── helpers.py                           # Basic helping functions
    └── NN_helpers.py                        # Helping functions used for the NN training

```

## Prerequisite

- Python 3.6.9
- PyTorch 1.7.0+cu101
- pandas 1.1.5
- sklearn 0.22.2.post1
- scipy 1.4.1
- numpy 1.18.5
- enable the usage of GPU (Because our code takes time to execute we used the free GPU on Google Coolab for running the code for a faster execution.)

## Running the code

All of the code is in notebooks, so to run the notebooks, Google Coolab can be used or [jupyter lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

## Final Model

Our best model and the results can be found in the NeuralNetworks_2Channels.ipynb notebook.

## Authors
* Marija Lazaroska     @ marija.lazaroska@epfl.ch   
* Deborah Scherrer Ma  @ deborah.scherrerma@epfl.ch
* Méline Zhao          @ meline.zhao@epfl.ch 
