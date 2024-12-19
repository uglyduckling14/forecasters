'''
========================================================================
module: cs5600_6600_f24_hw05_duts_models.py
author: vladimir kulyukin
description: training and testing discreate univariate ANN, CNN, and 
LSTM forecasters of hive weight and in-hive temperature.

bugs to vladimir kulyukin in canvas.

this source code is part of the open source code from the
supplementary materials to the article 
Kulyukin, V.; Coster, D.; Kulyukin, A. Meikle, W.; Weiss, M. 
"Discrete Time Series Forecasting of Hive Weight, In-Hive Temperature, 
and Hive Entrance Traffic in Non-Invasive Monitoring of Managed Honey 
Bee Colonies: Part I." 
Sensors 2024, 24(19), 6433; https://doi.org/10.3390/s24196433

This research was supported, in part, by the intramural research program of 
the U.S. Department of Agriculture, National Institute of Food and Agriculture 
Program DSFAS A1541 Award 2024-67013-42521. The findings, conclusions, or 
recommendations expressed in this publication have not been formally 
disseminated by the U.S. Department of Agriculture and should not be construed 
to represent any agency determination or policy. 
==========================================================================
'''

import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
from csv_aux import *  # Ensure this module is available
import argparse
import datetime


### VK: This is the directory where the logging files
###     are created. Change it as you see fit.
LOG_DIR = '\esper\Downloads\CS-5600\hw06\CS5600_6600_F24_LOG'

### VK: This is the CSV File pattern string used in
###     the function log_csv_mse() below.
CSV_FILE_PATH_PATTERN = LOG_DIR + 'LOG_{}_{}_WT_TH.csv'

### WH_HID_MHR and TH_HID_MHR are dictionaries
### that are used for hive id lookups.
### MHR mean hour, because we are forecasting
### mean hourly hive weights and in-hive temperatures.
WH_HID_MHR = {}
WH_HID_MHR[2120] = WH2120_MHR
WH_HID_MHR[2137] = WH2137_MHR
WH_HID_MHR[2059] = WH2059_MHR
WH_HID_MHR[2123] = WH2123_MHR
WH_HID_MHR[2141] = WH2141_MHR
WH_HID_MHR[2142] = WH2142_MHR
WH_HID_MHR[2129] = WH2129_MHR
WH_HID_MHR[2158] = WH2158_MHR
WH_HID_MHR[2130] = WH2130_MHR
WH_HID_MHR[2146] = WH2146_MHR

TH_HID_MHR = {}
TH_HID_MHR[2120] = TH2120_MHR
TH_HID_MHR[2137] = TH2137_MHR
TH_HID_MHR[2059] = TH2059_MHR
TH_HID_MHR[2123] = TH2123_MHR
TH_HID_MHR[2141] = TH2141_MHR
TH_HID_MHR[2142] = TH2142_MHR
TH_HID_MHR[2129] = TH2129_MHR
TH_HID_MHR[2158] = TH2158_MHR
TH_HID_MHR[2130] = TH2130_MHR
TH_HID_MHR[2146] = TH2146_MHR

def get_hid_mhr(hid, t_or_w):
    global WH_HID_MHR
    global TH_HID_MHR
    assert t_or_w == 'th' or t_or_w == 'wh'
    if t_or_w == 'th':
        return TH_HID_MHR[hid]
    elif t_or_w == 'wh':
        return WH_HID_MHR[hid]

def plot_y_yhat(plot_title, xlabel, ylabel, y, yhat, model_name,
                save_flag=False, log_dir=LOG_DIR):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(y, 'r--')
    plt.plot(yhat, 'bs--')
    plt.legend(['y', 'yhat'], loc='best')
    plt.title(plot_title)    
    if save_flag == True:
        plt.savefig(log_dir + model_name + '_y_yhat.png')
    else:
        plt.show()

def log_csv_mse(model_name, in_steps, out_steps, hiveid, mse_str):
    global CSV_FILE_PATH_PATTERN
    log_fp = CSV_FILE_PATH_PATTERN.format(in_steps, out_steps)
    with open(log_fp, 'a') as outf:
        out_str = '{}, {}, {}, {}, {}\n'.format(model_name,
                                                in_steps,
                                                out_steps,
                                                hiveid,
                                                mse_str)
        outf.write(out_str)
        outf.flush()

# split a univariate sequence of measurements (time series) into samples
def split_univar_multi_step_sequence(sequence, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

### ======== MODEL 1: Univar ANN WH Forecaster =========

'''
ANN WH multi-step partitioned train/test dataset:
X                                                                                y
[15.76255833 15.76094167 15.75195833 15.75128333 15.72945    15.649275  ] [15.71055833 15.75673333]
[15.76094167 15.75195833 15.75128333 15.72945    15.649275   15.71055833] [15.75673333 15.75889167]
[15.75195833 15.75128333 15.72945    15.649275   15.71055833 15.75673333] [15.75889167 15.75495   ]

real test inputs x_input_2 = [[[13.52895833]
  [13.53903333]
  [13.544975  ]
  [13.538775  ]
  [13.53390833]
  [13.53004167]]]
real test preds  y_hat_2   = [[13.147956 13.454838]]
real test gt               = [13.52428333 13.52319167]
appended preds: 13.454837799072266
appended gts:   13.52319167
Ground Truths = [14.701475, 14.668725, 14.67180833, 14.68540833, 14.71459167]
Preds         = [14.668917, 14.653497, 14.624329, 14.588801, 14.566184]
'''
### num_feats refers to input features.
def run_duts_wh_ann_aux(wh_train_series, wh_test_series,
                        num_in_steps=6, num_out_steps=2, num_epochs=3, verbose=0,
                        hiveid=WH2059_MHR,
                        real_hive_id=2059,
                        save_flag=True):
    print('real_hive_id = {}'.format(real_hive_id))
    assert save_flag == True
    print('run_duts_wh_ann_aux: num_in_steps={}; num_out_steps={}; num_epochs={}'.format(num_in_steps, num_out_steps, num_epochs))
    # 1. Create a test dataset
    X, y = split_univar_multi_step_sequence(wh_train_series, num_in_steps, num_out_steps)
    ## number of input stream
    num_features = 1

    print(wh_train_series[:48])
    print('num_in_steps={}'.format(num_in_steps))
    print('num_features={}'.format(num_features))
    print('num_out_steps={}'.format(num_out_steps))    
    print('ANN UNIV WH multi-step partitioned train dataset:')
    for i in range(3):
        print(X[i], y[i])

    # 2. Construct ANN model
    model = Sequential()
    model.add(Dense(5, input_shape=(num_in_steps, num_features), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_out_steps))    
    model.compile(optimizer='adam', loss='mse')

    # 3. fit model on the train data X, y
    model.fit(X, y, epochs=num_epochs, verbose=verbose)

    # 4. save model
    sfx = str(datetime.datetime.today().timestamp()).replace('.', '_')    
    model_name = 'ann_duts_wh_{}_{}_{}_{}_{}.h5'.format(real_hive_id, num_in_steps, num_out_steps, num_epochs, sfx)
    model.save(LOG_DIR+model_name)
    loaded_model = load_model(LOG_DIR + model_name)
    
    # 5. Construct test dataset
    X2, y2 = split_univar_multi_step_sequence(wh_test_series, num_in_steps, num_out_steps)

    print(wh_test_series[:48])
    print('num_in_steps={}'.format(num_in_steps))        
    print('num_features={}'.format(num_features))
    print('num_out_steps={}'.format(num_out_steps))    
    print('ANN WH multi-step partitioned test dataset:')
    for i in range(3):
        print(X[i], y[i])

    # 6. test the model on new data
    ground_truth, preds = [], []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_in_steps, num_features))
        y_hat_2   = loaded_model.predict(x_input_2)
        print('real test inputs x_input_2 = {}'.format(x_input_2))
        print('real test preds  y_hat_2   = {}'.format(y_hat_2))
        print('real test gt               = {}'.format(y2[i]))
        print('appended preds: {}'.format(y_hat_2[0][num_out_steps-1]))
        print('appended gts:   {}'.format(y2[i][num_out_steps-1]))
        preds.append(y_hat_2[0][num_out_steps-1])
        ground_truth.append(y2[i][num_out_steps-1])

    ### print a few ground truths and predictions
    print('Ground Truths = {}'.format(ground_truth[:5]))
    print('Preds         = {}'.format(preds[:5]))

    # 7. compute mse and plot
    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    mse_str = "%0.4f" % mse
    model_name = model_name[:-3]
    plot_title = '{}: wh; mse={}'.format(model_name, mse_str)
    print('*** MODEL ****: {}'.format(model_name))
    #log_csv_mse(model_name, in_steps, out_steps, hiveid, mse_str):    
    if save_flag:
        log_csv_mse(model_name, num_in_steps, num_out_steps, real_hive_id, mse_str)
    plot_y_yhat(plot_title, 'x', 'y/yhat',
                ground_truth, preds,
                model_name, save_flag=save_flag)

def run_duts_wh_ann(csv_fp, csv_index, train_percent=0.7,
                    hiveid=WH2120_MHR, real_hive_id=2120,
                    num_in_steps=6, num_out_steps=2, num_epochs=3,
                    verbose=0, save_flag=False):
    print('real_hive_id={}'.format(real_hive_id))
    print('verbose={}'.format(verbose))     
    recs = csv_file_to_arys(csv_fp, csv_index)
    wh   = get_weight_measurements_for_hiveid(recs, hiveid=hiveid)
    num_recs = len(wh)
    split_index   = int(num_recs * train_percent)
    wh_train_recs = wh[:split_index]
    wh_test_recs  = wh[split_index:]
    print('num_recs       = {}'.format(num_recs))
    print('split_index    = {}'.format(split_index))
    print('num_train_recs = {}'.format(len(wh_train_recs)))
    print('num_test_recs  = {}'.format(len(wh_test_recs)))
    run_duts_wh_ann_aux(wh_train_recs, wh_test_recs,
                        hiveid=hiveid,
                        real_hive_id=real_hive_id,
                        num_in_steps=num_in_steps,
                        num_out_steps=num_out_steps,
                        num_epochs=num_epochs,
                        verbose=verbose,
                        save_flag=save_flag)

### ======== MODEL 2: Univar ANN TH (Temperature) forecaster =========

def run_duts_th_ann_aux(th_train_series, th_test_series,
                        num_in_steps=12, num_out_steps=2, num_epochs=3,
                        verbose=1, hiveid=TH2120_MHR, real_hive_id=2120,
                        save_flag=True):
    global LOG_DIR
    print('run_duts_th_ann_aux: num_in_steps={}; num_out_steps={}; num_epochs={}'.format(num_in_steps, num_out_steps, num_epochs))
    assert save_flag == True
    # 1. Create a test dataset
    X, y = split_univar_multi_step_sequence(th_train_series, num_in_steps, num_out_steps)
    num_features = 1

    print(th_train_series[:48])
    print('num_in_steps={}'.format(num_in_steps))
    print('num_features={}'.format(num_features))
    print('num_out_steps={}'.format(num_out_steps))
    print('ANN TH multi-step partitioned train dataset:')
    for i in range(3):
        print(X[i], y[i])

    # 2. Construct ANN model
    model = Sequential()
    model.add(Dense(5, input_shape=(num_in_steps, num_features), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_out_steps))
    model.compile(optimizer='adam', loss='mse')

    # 3. fit model
    model.fit(X, y, epochs=num_epochs, verbose=verbose)

    # 4. save model
    sfx = str(datetime.datetime.today().timestamp()).replace('.', '_')
    model_name = 'ann_duts_th_{}_{}_{}_{}_{}.h5'.format(real_hive_id, num_in_steps, num_out_steps, num_epochs, sfx)    
    model.save(LOG_DIR + model_name)
    loaded_model = load_model(LOG_DIR + model_name)
    
    # 5. Construct test dataset
    X2, y2 = split_univar_multi_step_sequence(th_test_series, num_in_steps, num_out_steps)

    print(th_test_series[:48])
    print('num_in_steps={}'.format(num_in_steps))
    print('num_features={}'.format(num_features))
    print('num_out_steps={}'.format(num_out_steps))    
    print('ANN TH multi-step partitioned test dataset:')
    for i in range(3):
        print(X[i], y[i])

    # 6. test the model on new data
    ground_truth, preds = [], []
    for i in range(len(X2)):
        x_input_2 = X2[i].reshape((1, num_in_steps, num_features))
        y_hat_2   = loaded_model.predict(x_input_2)
        #print('y_hat_2 = {}'.format(y_hat_2))
        preds.append(y_hat_2[0][num_out_steps-1])
        ground_truth.append(y2[i][num_out_steps-1])

    ### print a few ground truths and precitions
    print('Ground Truths = {}'.format(ground_truth[:5]))
    print('Preds         = {}'.format(preds[:5]))

    # 7. compute mse and plot
    mse = (np.array(ground_truth) - np.array(preds))**2
    mse = np.mean(mse)
    mse_str = "%0.4f" % mse
    model_name = model_name[:-3]
    plot_title = '{}: wh; mse={}'.format(model_name, mse_str)
    print('*** MODEL ****: {}'.format(model_name))
    if save_flag:
        log_csv_mse(model_name, num_in_steps, num_out_steps, real_hive_id, mse_str)    
    plot_y_yhat(plot_title, 'x', 'y/yhat',
                ground_truth, preds,
                model_name, save_flag=save_flag)

def run_duts_th_ann(csv_fp, csv_index, train_percent=0.7,
                    hiveid=WH2120_MHR,
                    real_hive_id=2120,
                    num_in_steps=12, num_out_steps=2, num_epochs=3,
                    verbose=0, save_flag=False):
   print('verbose={}'.format(verbose))     
   recs = csv_file_to_arys(csv_fp, csv_index)
   th   = get_temp_measurements_for_hiveid(recs, hiveid=hiveid)
   num_recs = len(th)
   split_index   = int(num_recs * train_percent)
   th_train_recs = th[:split_index]
   th_test_recs  = th[split_index:]
   run_duts_th_ann_aux(th_train_recs, th_test_recs,
                       hiveid=hiveid,
                       real_hive_id=real_hive_id,
                       num_in_steps=num_in_steps,
                       num_out_steps=num_out_steps,
                       num_epochs=num_epochs,
                       verbose=verbose,
                       save_flag=save_flag)

### ======== MODEL 3: Univar CNN WH (Weight of Hive) Forecaster =========

def run_duts_wh_cnn_aux(wh_train_series, wh_test_series,
                        num_in_steps=12, num_out_steps=2, num_epochs=3, verbose=1,
                        hiveid=WH2120_MHR, real_hive_id=2120,
                        save_flag=True):
    global LOG_DIR
    print('run_duts_wh_cnn_aux: num_in_steps={}; num_out_steps={}; num_epochs={}'.format(num_in_steps, num_out_steps, num_epochs))
    assert save_flag == True    
    ### your code here
    pass

def run_duts_wh_cnn(csv_fp, csv_index, train_percent=0.7,
                    hiveid=WH2120_MHR, real_hive_id=2120,
                    num_in_steps=12, num_out_steps=2, num_epochs=3, verbose=0,
                    save_flag=False):
   print('verbose={}'.format(verbose))     
   recs = csv_file_to_arys(csv_fp, csv_index)
   wh   = get_weight_measurements_for_hiveid(recs, hiveid=hiveid)
   num_recs = len(wh)
   split_index   = int(num_recs * train_percent)
   wh_train_recs = wh[:split_index]
   wh_test_recs  = wh[split_index:]        
   run_duts_wh_cnn_aux(wh_train_recs, wh_test_recs,
                       hiveid=hiveid,
                       real_hive_id=real_hive_id,
                       num_in_steps=num_in_steps,
                       num_out_steps=num_out_steps,
                       num_epochs=num_epochs,
                       verbose=verbose,
                       save_flag=save_flag)

### ======== MODEL 4: Univar CNN TH (Temperature of Hive) Forecaster =========

def run_duts_th_cnn_aux(th_train_series, th_test_series,
                        num_in_steps=12, num_out_steps=2, num_epochs=3, verbose=1,
                        hiveid=TH2120_MHR, real_hive_id=2120,
                        save_flag=True):
    global LOG_DIR
    print('run_duts_th_cnn_aux: num_in_steps={}; num_out_steps={}; num_epochs={}'.format(num_in_steps, num_out_steps, num_epochs))
    assert save_flag == True        
    ### your code here
    pass

def run_duts_th_cnn(csv_fp, csv_index, train_percent=0.7,
                    hiveid=TH2120_MHR, real_hive_id=2120,
                    num_in_steps=12, num_out_steps=2, num_epochs=3, verbose=0,
                    save_flag=False):
   print('verbose={}'.format(verbose))
   recs = csv_file_to_arys(csv_fp, csv_index)
   th   = get_temp_measurements_for_hiveid(recs, hiveid=hiveid)
   num_recs = len(th)
   print('NUM RECRS = {}'.format(num_recs))      
   split_index   = int(num_recs * train_percent)
   th_train_recs = th[:split_index]
   th_test_recs  = th[split_index:]        
   run_duts_th_cnn_aux(th_train_recs, th_test_recs,
                       hiveid=hiveid,
                       real_hive_id=real_hive_id,
                       num_in_steps=num_in_steps,
                       num_out_steps=num_out_steps,
                       num_epochs=num_epochs,
                       verbose=verbose,
                       save_flag=save_flag)

### ======== MODEL 5: Univar LSTM WH (Weight of Hive) Forecaster =========

def run_duts_wh_lstm_aux(wh_train_series, wh_test_series,
                         num_in_steps=12, num_out_steps=2, num_epochs=3, verbose=1,
                         hiveid=WH2120_MHR, real_hive_id=2120,
                         save_flag=True):
    global LOG_DIR
    print('run_duts_wh_lstm_aux: num_in_steps={}; num_out_steps={}; num_epochs={}'.format(num_in_steps,
                                                                                          num_out_steps,
                                                                                          num_epochs))
    assert save_flag == True        
    ### your code here
    pass

def run_duts_wh_lstm(csv_fp, csv_index, train_percent=0.7,
                     hiveid=WH2120_MHR, real_hive_id=2120,
                     num_in_steps=12, num_out_steps=2, num_epochs=3,
                     verbose=0, save_flag=False):
   print('verbose={}'.format(verbose))     
   recs = csv_file_to_arys(csv_fp, csv_index)
   wh   = get_weight_measurements_for_hiveid(recs, hiveid=hiveid)
   num_recs = len(wh)
   print('NUM RECRS = {}'.format(num_recs))         
   split_index   = int(num_recs * train_percent)
   wh_train_recs = wh[:split_index]
   wh_test_recs  = wh[split_index:]        
   run_duts_wh_lstm_aux(wh_train_recs, wh_test_recs,
                        hiveid=hiveid, real_hive_id=real_hive_id,
                        num_in_steps=num_in_steps,
                        num_out_steps=num_out_steps,
                        num_epochs=num_epochs,
                        verbose=verbose,
                        save_flag=save_flag)
   
### ======== MODEL 6: Univar LSTM TH (Temperature of Hive) Forecaster =========

def run_duts_th_lstm_aux(th_train_series, th_test_series,
                         num_in_steps=12, num_out_steps=2, num_epochs=3, verbose=1,
                         hiveid=TH2120_MHR, real_hive_id=2120,
                         save_flag=True):
    global LOG_DIR
    print('run_duts_th_lstm_aux: num_in_steps={}; num_out_steps={}; num_epochs={}'.format(num_in_steps, num_out_steps, num_epochs))
    assert save_flag == True        
    ### your code here
    pass

def run_duts_th_lstm(csv_fp, csv_index, train_percent=0.7,
                     hiveid=TH2120_MHR, real_hive_id=2120,
                     num_in_steps=12, num_out_steps=2, num_epochs=3, verbose=0,
                     save_flag=False):
   print('verbose={}'.format(verbose))     
   recs = csv_file_to_arys(csv_fp, csv_index)
   th   = get_temp_measurements_for_hiveid(recs, hiveid=hiveid)
   num_recs = len(th)
   print('NUM RECRS = {}'.format(num_recs))
   split_index   = int(num_recs * train_percent)
   th_train_recs = th[:split_index]
   th_test_recs  = th[split_index:]        
   run_duts_th_lstm_aux(th_train_recs, th_test_recs,
                        hiveid=hiveid,
                        real_hive_id=real_hive_id,
                        num_in_steps=num_in_steps,
                        num_out_steps=num_out_steps,
                        num_epochs=num_epochs,
                        verbose=verbose,
                        save_flag=save_flag)

### ===============================================================

### VK: Command line arguments.
parser = argparse.ArgumentParser(
    prog='wh_th',
    description='wh_th',
    epilog='wh_th')

parser.add_argument('-fp',   '--file_path')
parser.add_argument('-net',  '--network')
parser.add_argument('-tp',   '--train_percent', type=float) 
parser.add_argument('-hid',  '--hiveid', type=int)
parser.add_argument('-ne',   '--num_epochs', type=int)
parser.add_argument('-nis',  '--num_in_steps', type=int)
parser.add_argument('-nos',  '--num_out_steps', type=int)
parser.add_argument('-tw',   '--t_or_w') ## -tw wh or -tw th

def run_duts_forecaster(fp, net, train_percent, hid, ne, nis, nos, t_or_w):
    ### this global variable is defined in csv_aux.py.
    global CSV_INDEX_MHR
    real_hive_id = hid
    hid = get_hid_mhr(hid, t_or_w)
    if t_or_w == 'wh':
        if net == 'ann':
            run_duts_wh_ann(fp, CSV_INDEX_MHR, train_percent=train_percent,
                            hiveid=hid, real_hive_id=real_hive_id, num_in_steps=nis, num_out_steps=nos,
                            num_epochs=ne,
                            verbose=0, save_flag=True)
        elif net == 'cnn':
            run_duts_wh_cnn(fp, CSV_INDEX_MHR, train_percent=train_percent,
                            hiveid=hid, real_hive_id=real_hive_id, num_in_steps=nis, num_out_steps=nos,
                            num_epochs=ne,
                            verbose=0, save_flag=True)
        elif net == 'lstm':
            run_duts_wh_lstm(fp, CSV_INDEX_MHR, train_percent=train_percent,
                             hiveid=hid, real_hive_id=real_hive_id, num_in_steps=nis, num_out_steps=nos,
                             num_epochs=ne,
                             verbose=0, save_flag=True)
        else:
            assert 1 == 0
    elif t_or_w == 'th':
        if net == 'ann':
            run_duts_th_ann(fp, CSV_INDEX_MHR, train_percent=train_percent,
                            hiveid=hid, real_hive_id=real_hive_id, num_in_steps=nis, num_out_steps=nos,
                            num_epochs=ne,
                            verbose=0, save_flag=True)
        elif net == 'cnn':
            run_duts_th_cnn(fp, CSV_INDEX_MHR, train_percent=train_percent,
                            hiveid=hid, real_hive_id=real_hive_id, num_in_steps=nis, num_out_steps=nos,
                            num_epochs=ne,
                            verbose=0, save_flag=True)            
        elif net == 'lstm':
            run_duts_th_lstm(fp, CSV_INDEX_MHR, train_percent=train_percent,
                             hiveid=hid, real_hive_id=real_hive_id, num_in_steps=nis, num_out_steps=nos,
                             num_epochs=ne,
                             verbose=0, save_flag=True)
        else:
            assert 1 == 0


if __name__ == '__main__':
    args = parser.parse_args()
    run_duts_forecaster(args.file_path, args.network, args.train_percent,
                        args.hiveid, args.num_epochs,
                        args.num_in_steps,
                        args.num_out_steps,
                        args.t_or_w)
    pass
