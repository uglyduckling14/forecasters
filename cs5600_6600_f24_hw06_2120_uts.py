'''
========================================================================
module: cs5600_6600_f24_hw05_duts_uts.py
author: vladimir kulyukin
description: unit tests for training and testing discreate univariate ANN, CNN, and 
LSTM forecasters of hive weight and in-hive temperature.

bugs to vladimir kulyukin in canvas.
==========================================================================
'''

import numpy as np
from  cs5600_6600_f24_hw05_duts_models import *
import unittest

class cs5600_6600_f21_hw02_2120_uts(unittest.TestCase):

    ### hive 2120
    '''
    def test_run_duts_ann_hive_weight_forecaster_for_2120(self):
        run_duts_forecaster('WH_TH_HOURLY_MEANS_USDA_ARS_TUCSON_AZ_2022.csv',
                            'ann', 0.7, 2120, 10, 6, 2, 'wh')
    '''

    '''
    def test_run_duts_ann_hive_temp_forecaster_for_2120(self):
        run_duts_forecaster('WH_TH_HOURLY_MEANS_USDA_ARS_TUCSON_AZ_2022.csv',
                            'ann', 0.7, 2120, 10, 6, 2, 'th')        
    '''

    '''
    def test_run_duts_cnn_hive_weight_forecaster_for_2120(self):
        run_duts_forecaster('WH_TH_HOURLY_MEANS_USDA_ARS_TUCSON_AZ_2022.csv',
                            'cnn', 0.7, 2120, 10, 6, 2, 'wh')
    '''

    '''
    def test_run_duts_cnn_hive_temp_forecaster_for_2120(self):
        run_duts_forecaster('WH_TH_HOURLY_MEANS_USDA_ARS_TUCSON_AZ_2022.csv',
                            'cnn', 0.7, 2120, 10, 6, 2, 'th')        
    '''

    '''
    def test_run_duts_lstm_hive_weight_forecaster_for_2120(self):
        run_duts_forecaster('WH_TH_HOURLY_MEANS_USDA_ARS_TUCSON_AZ_2022.csv',
                            'lstm', 0.7, 2120, 10, 6, 2, 'wh')
    '''

    '''
    def test_run_duts_lstm_hive_temp_forecaster_for_2120(self):
        run_duts_forecaster('WH_TH_HOURLY_MEANS_USDA_ARS_TUCSON_AZ_2022.csv',
                            'lstm', 0.7, 2120, 10, 6, 2, 'th')
    '''

    '''
    def test_run_duts_hive_temp_forecasters(self):
        for forecaster_type in ('ann', 'cnn', 'lstm'):
            for hive_id in (2120,2137,2059,2123,2141,2142,2129,2158,2130,2146):
                run_duts_forecaster('WH_TH_HOURLY_MEANS_USDA_ARS_TUCSON_AZ_2022.csv', forecaster_type,
                                    0.7, hive_id, 10, 6, 2, 'th')
    '''
            
### ================ Unit Tests ====================

if __name__ == '__main__':
    unittest.main()


 
    
        
        

    
        





