
'''
========================================================================
module: csv_aux.py
author: vladimir kulyukin
description: auxiliary csv funcctions to preprocess csv data
to train/test discrete univariate ANN/CNN/LSTM forecasters of 
hive weight and in-hive temperature.

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math

### global constants to processe data in WH_TH_HOURLY_MEANS_USDA_ARS_TUCSON_AZ_2022.csv
MONTH_MHR      = 0
DAYOFMONTH_MHR = 1
HOUROFDAY_MHR  = 2
HOUR_MHR       = 3
WH2123_MHR     = 4
WH2120_MHR     = 5
WH2137_MHR     = 6
WH2059_MHR     = 7
WH2141_MHR     = 8
WH2142_MHR     = 9
WH2129_MHR     = 10
WH2158_MHR     = 11
WH2130_MHR     = 12
WH2146_MHR     = 13
TH2123_MHR     = 14
TH2120_MHR     = 15
TH2137_MHR     = 16
TH2059_MHR     = 17
TH2141_MHR     = 18
TH2142_MHR     = 19
TH2129_MHR     = 20
TH2158_MHR     = 21
TH2130_MHR     = 22
TH2146_MHR     = 23

### This array is used in run_duts_forecaster() in
### cs5600_6600_f24_hw05_duts_models.py.
CSV_INDEX_MHR = [MONTH_MHR,DAYOFMONTH_MHR,HOUROFDAY_MHR,HOUR_MHR,
                 WH2123_MHR, WH2120_MHR, WH2137_MHR, WH2059_MHR, WH2141_MHR,
                 WH2142_MHR, WH2129_MHR, WH2158_MHR, WH2130_MHR, WH2146_MHR,
                 TH2123_MHR, TH2120_MHR, TH2137_MHR, TH2059_MHR, TH2141_MHR,
                 TH2142_MHR, TH2129_MHR, TH2158_MHR, TH2130_MHR, TH2146_MHR]

### the remainder of the file are csv processing tools.

def csv_line_to_ary(ln, col_index):
    ary = []
    for col_index in col_index:
        if col_index == 0:
            ary.append(ln[col_index])
        else:
            ary.append(float(ln[col_index]))
    return ary

def csv_line_to_str_ary(ln, col_index):
    ary = []
    for col_index in col_index:
        ary.append(ln[col_index].strip())
    return ary    

def csv_file_to_arys(in_path, col_index):
    recs = []
    with open(in_path, mode='r') as inf:
        lncnt = 0
        for ln in csv.reader(inf):
            if '' not in ln:
                if lncnt == 0:
                    lncnt = 1
                elif lncnt > 0 and len(ln)>0:
                    recs.append(csv_line_to_ary(ln, col_index))
    return recs

def csv_file_to_str_arys(in_path, col_index):
    recs = []
    with open(in_path, mode='r') as inf:
        lncnt = 0
        for ln in csv.reader(inf):
            if '' not in ln:
                if lncnt == 0:
                    lncnt = 1
                elif lncnt > 0 and len(ln)>0:
                    recs.append(csv_line_to_str_ary(ln, col_index))
    return recs

def parse_date(dt):
    mdy, hm = dt.split()
    m,d,y = mdy.split('/')
    m,d,y = int(m), int(d), int(y)
    if y == 22:
        y += 2000
    hm    = hm.split(':')
    h, mnt  = int(hm[0]), int(hm[1])
    return m,d,y,h,mnt

def is_date_same(date_1, date_2):
    pd1 = parse_date(date_1)
    pd2 = parse_date(date_2)
    assert len(pd1) == len(pd2) == 5
    for i in range(5):
        if pd1[i] != pd2[i]:
            return False
    return True

def get_month_recs(mon, recs):
    mon_recs = []
    for r in recs:
        pdt = parse_date(r[TP_DATE_INDEX])
        if pdt[0] == mon:
            mon_recs.append(r)
    return mon_recs

def get_tw_recs(recs):
    t_recs, w_recs = [], []
    for d, t, w in recs:
        t_recs.append(t)
        w_recs.append(w)
    return t_recs, w_recs

def get_weight_measurements_for_hiveid(recs, hiveid=WH2120_MHR):
    return [r[hiveid] for r in recs]

def get_temp_measurements_for_hiveid(recs, hiveid=TH2120_MHR):
    return [r[hiveid] for r in recs]

def filter_csv_file_to_str_arys(in_path, col_index, col_number, col_val):
    recs = []
    with open(in_path, mode='r') as inf:
        lncnt = 0
        for ln in csv.reader(inf):
            if len(ln) > 0 and not '' in ln:
                s = csv_line_to_str_ary(ln, col_index)
                if s[col_number] == col_val:
                    recs.append(s)
    return recs


