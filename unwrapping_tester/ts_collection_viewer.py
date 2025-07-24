#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:42:43 2024

@author: xap

    Optimization algorithm based on CPLEX
   
        Linear Model
        Continuous variable: 2 (a and b_slope)
        Discrete variables: #ts_length  (kd(t_k))
        Total: ts_length + 1 variable
        
        Cost function: sum((g[k] - a) ** 2
        kd[0] = 0
    

"""
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.ts_collection import TSCollection, TSSubset




starttime = datetime.now()

generaldata_folder = "/mnt/DATI_PC/AAA_DOTTORATO/PROGETTI/PS_DATA/Real/"
collection_folder = "Toscana_2/"
collection_file = "TOSCANA_ps.tsc"

# generaldata_folder = "/mnt/DATI_PC/AA1_PROGETTI/PS_DATA/Real/"
# collection_folder = "Brennero/"
# # collection_file = "BrenneroNoReUnwr4DIF.mat"
# collection_file = "BrenneroReUnwr4DIF.mat"
# ts_number_list = list(range(37, 50))
# ts_number_list = [37, 44]


tscollection = TSCollection()
tscollection.load(generaldata_folder + collection_folder + collection_file)



# ts_number = 33869 # no phase jump
# ts_number = 27000 # no phase jump
# ts_number = 62158 # no phase jump
# ts_number = 39418 # no phase jump


# ts_number_list = [22303]
# ts_number_list = list(range(0,20))
# ts_number_list = list(range(0,778))

ts_number_list = [77]

# ts_number_list = [77, 2453, 22303, 26966, 27831, 34247, 50477, 11453, 26071, 17560, 34567, 41138, 34348, 35205, 35623, 38047, 56539, 33646, 16476, 16729]
# ts_number_list = [77, 11453, 26071, 17560, 34348, 35205, 35623, 38047, 56539, 33646, 16476, 16729]

# ts_number_list = [50223, 40685, 37327, 65920, 16777, 16081, 16638, 19441, 73687]
# ts_number_list = list(range(0, tscollection.collection_size))

collection_subset = TSSubset(tscollection.get_collection_dict(), ts_number_list)


# # # # ############ PLOT UNWRAPPING ++++++++++++++++++++++++++++
plt.figure(figsize=(6, 3))
for ts in ts_number_list:
# for ts in [56829]:
# for ts in acc1[acc1 < 70].index:  
    wt = collection_subset.get_data('w').loc[ts]
    ut = collection_subset.get_data('u').loc[ts]
    kd_ref = collection_subset.get_data('kd').loc[ts]
    
    
    plt.plot(collection_subset.absolute_timeline, wt, 'r.', label="Original Series")
    plt.plot(collection_subset.absolute_timeline, ut, '.', label="Original Series")
    plt.plot(collection_subset.absolute_timeline, -kd_ref, '.', label="Original Series", color = 'orange')

    plt.title(f"Reference: {ts}")
    # plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()
    
 