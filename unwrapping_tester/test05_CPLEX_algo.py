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
import seaborn as sns
from src.ts_collection import TSCollection, TSSubset
from src.ts_packets import Unwrapping, Seasonality, SNR



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
#print(collection_subset.get_relative_timeline())
# print(len(collection_subset.w_dict))
# print(collection_subset.w_dict[45231])


starttime1 = datetime.now()

# ############ GENERATE NEW PARTIAL UNWRAPPING ++++++++++++++++++++++++++++
# # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {}
# partial_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# partial_unwrapping.new(unwrapping_name = 'partial_unwrap_test_subset',
#                          unwrapping_algo = 'partial_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "Partial unwrapping: b_slope continuous; integers: kd")
# partial_unwrapping.save()

############ GENERATE NEW CPLEXbase_UNWRAPPING ++++++++++++++++++++++++++++
# # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 16, "n_cpu": 4}
# cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplex_unwrapping.new(unwrapping_name = 'cplexbase_test_subset',
#                          unwrapping_algo = 'CPLEXbase_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping, 1 continuous: v, N integers: kd")
# cplex_unwrapping.save()

# ############ GENERATE NEW CPLEXstep_UNWRAPPING ++++++++++++++++++++++++++++
# # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 160}
# cplexstep_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexstep_unwrapping.new(unwrapping_name = 'cplexstep_test_subset',
#                          unwrapping_algo = 'CPLEXstep_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping, 1 continuous: v, N integers: kd")
# cplexstep_unwrapping.save()

############ GENERATE NEW CPLEXnoslope_UNWRAPPING ++++++++++++++++++++++++++++
# # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 16}
# cplexnoslope_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexnoslope_unwrapping.new(unwrapping_name = 'cplexnoslope_test_subset',
#                          unwrapping_algo = 'CPLEXnoslope_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping: N integers: kd")
# cplexnoslope_unwrapping.save()

# # # ############ GENERATE NEW CPLEXmean_UNWRAPPING ++++++++++++++++++++++++++++
# # # # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 200, 'max_slope': 16, "n_cpu": 4}
# cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplex_unwrapping.new(unwrapping_name = 'cplexmean_test_subset',
#                          unwrapping_algo = 'CPLEXmean_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "used an integer slope")
# # cplex_unwrapping.save()

# # # ############ GENERATE NEW CPLEXmeanpoly_UNWRAPPING ++++++++++++++++++++++++++++
# # # # # 1. Define the unwrapping object 2. Create new Unwrapping data
unwrap_param = {'min_t_index': 0, 'max_t_index': 300, 'max_slope': 16, 'polynomial_degree': 3, "n_cpu": 4}
cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
cplex_unwrapping.new(unwrapping_name = 'cplexmeanpoly_test_subset',
                         unwrapping_algo = 'CPLEXmeanpoly_unwrap', unwrap_param = unwrap_param,
                         unwrapping_note = "used an integer slope")
# # cplex_unwrapping.save()

# # ############ GENERATE NEW CPLEXmeanSeason_UNWRAPPING ++++++++++++++++++++++++++++
# # # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 200, 'max_slope': 16}
# cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplex_unwrapping.new(unwrapping_name = 'cplexmean_season02_test_subset',
#                          unwrapping_algo = 'CPLEXmean_season_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "used the index instead of time")
# #cplex_unwrapping.save()

# # ############ GENERATE NEW PYOMOmeanSeason_UNWRAPPING ++++++++++++++++++++++++++++
# # # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 200, 'max_slope': 16}
# cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplex_unwrapping.new(unwrapping_name = 'pyomomean_season_unwrap',
#                          unwrapping_algo = 'PYOMOmean_season_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "pyomo packet with seasonality")
# cplex_unwrapping.save()



# ############ GENERATE NEW CPLEXloess_UNWRAPPING ++++++++++++++++++++++++++++
# # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 16, 'bandwidth': 25}
# cplexloess_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexloess_unwrapping.new(unwrapping_name = 'cplexloess_test_subset',
#                          unwrapping_algo = 'CPLEXloess_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping: N integers: kd")
# cplexloess_unwrapping.save()

# ############ GENERATE NEW CPLEXlocal_UNWRAPPING ++++++++++++++++++++++++++++
# # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 16, 'd': 10}
# cplexlocal_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexlocal_unwrapping.new(unwrapping_name = 'cplexlocal_d10_test_subset',
#                          unwrapping_algo = 'CPLEXlocal_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping: b_slope continuous; 2 * d integers: kd")
# cplexlocal_unwrapping.save()

# # ############ GENERATE NEW CPLEXrandom_UNWRAPPING ++++++++++++++++++++++++++++
# # # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 16, 'd': 20}
# cplexrandom_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexrandom_unwrapping.new(unwrapping_name = 'cplexrandom_d20_test_subset',
#                          unwrapping_algo = 'CPLEXrandom_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping: b_slope continuous; d integers: kd")
# cplexrandom_unwrapping.save()

# # ############ GENERATE NEW CPLEXvariance_UNWRAPPING ++++++++++++++++++++++++++++
# # # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 16, 'd': 20}
# cplexvariance_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexvariance_unwrapping.new(unwrapping_name = 'cplexvariance_test_subset',
#                          unwrapping_algo = 'CPLEXvariance_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping: b_slope continuous; d integers: kd")
# cplexvariance_unwrapping.save()

# # ############ GENERATE NEW CPLEXhybridrandom_UNWRAPPING ++++++++++++++++++++++++++++
# # # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 16, 'd1': 2, 'd2': 10}
# cplexhybridrandom_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexhybridrandom_unwrapping.new(unwrapping_name = 'cplexhybridrandom_d20_test_subset',
#                          unwrapping_algo = 'CPLEXhybridrandom_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping: b_slope continuous; integers: kd")
# cplexhybridrandom_unwrapping.save()

# # ############ GENERATE NEW CPLEXhybridrandomglobal_UNWRAPPING ++++++++++++++++++++++++++++
# # # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 16, 'd1': 20, 'd2': 10}
# cplexhybridrandomglobal_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexhybridrandomglobal_unwrapping.new(unwrapping_name = 'cplexhybridrandomglobal_2_10_test_subset',
#                          unwrapping_algo = 'CPLEXhybridrandomglobal_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping: b_slope continuous; integers: kd")
# cplexhybridrandomglobal_unwrapping.save()

############ GENERATE NEW CPLEXstep_UNWRAPPING ++++++++++++++++++++++++++++
# # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 100, 'max_slope': 160}
# cplexvolatility_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplexvolatility_unwrapping.new(unwrapping_name = 'cplexvolatility_test_subset',
#                          unwrapping_algo = 'CPLEXvolatility_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "CPLEX unwrapping, 1 continuous: v, N integers: kd")
# cplexvolatility_unwrapping.save()

print("Time CPLEXhybridrandomglobal : ", datetime.now()- starttime)


######### SAVE Unwrapping #####################
# cplexlocal_unwrapping.save()

# # # ############ OPEN UNWRAPPING ++++++++++++++++++++++++++++
# cplex_unwrapping = Unwrapping(collection_subset)

# cplex_unwrapping.load('partial_unwrap_test_subset.unw')
# cplex_unwrapping.load('cplexbase_test_subset.unw')
# cplex_unwrapping.load('cplexnoslope_test_subset.unw')
# cplex_unwrapping.load('cplexmean_test_subset.unw')
# cplex_unwrapping.load('cplexloess_test_subset.unw')

# cplex_unwrapping.load('cplexlocal_d10_test_subset.unw')
# cplex_unwrapping.load('cplexlocal_d20_test_subset.unw')
# cplex_unwrapping.load('cplexlocal_d30_test_subset.unw')
# cplex_unwrapping.load('cplexrandom_d20_test_subset.unw')
# cplex_unwrapping.load('cplexvariance_test_subset.unw')
# cplex_unwrapping.load('cplexhybridrandom_d20_test_subset.unw')
# cplex_unwrapping.load('CPLEXhybridrandomglobal_unwrap.unw')
# cplex_unwrapping.load('cplexmean_season2_test_subset.unw')
###################### COMPARE RESULTS ###################################

# kd_coll = collection_subset.get_data('kd')
# diff, acc1 = cplex_unwrapping.compare_by_offset(kd_coll)

# acc1 = cplex_unwrapping.compare(kd_coll, 'BrenneroGAP')

# print(diff.shape)

# # Filtra la Series per ottenere solo i valori < 80
# mask = acc1 < 80
# count_less_than_80 = mask.sum()
# print("Numero di valori < 80:", count_less_than_80)

# # Crea un DataFrame con l'indice e il valore corrispondente per ciascun elemento < 80
# df_less_than_80 = pd.DataFrame({
#     'Index': acc1.index[mask],
#     'Value': acc1[mask]
# })

# print(df_less_than_80)



# # # # ############ PLOT UNWRAPPING ++++++++++++++++++++++++++++
plt.figure(figsize=(6, 3))
for ts in ts_number_list:
# for ts in [56829]:
# for ts in acc1[acc1 < 70].index:  
    wt = collection_subset.get_data('w').loc[ts]
    ut = collection_subset.get_data('u').loc[ts]
    kd_ref = collection_subset.get_data('kd').loc[ts]
    
    wc = cplex_unwrapping.get_data('w').loc[ts]   
    uc = cplex_unwrapping.get_data('u').loc[ts]
    kd_calc = (np.round(wc - uc)).astype(int)
    

    plt.plot(collection_subset.absolute_timeline, wt, 'r.', label="Original Series")
    plt.plot(collection_subset.absolute_timeline, ut, '.', label="Original Series")
    plt.plot(collection_subset.absolute_timeline, -kd_ref, '.', label="Original Series", color = 'orange')

    plt.title(f"Reference: {ts}")
    # plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()
    
    plt.plot(collection_subset.absolute_timeline, wc, 'r.', label="Original Series")
    plt.plot(collection_subset.absolute_timeline, cplex_unwrapping.get_data('m').loc[ts], 'b.', label="Original Series")
    plt.plot(collection_subset.absolute_timeline, uc, '.', color = 'orange', label="Original Series")
    plt.plot(collection_subset.absolute_timeline, -kd_calc, 'g.', label="Original Series")

    plt.title(f"Unwrapping: {ts}")
    # plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()
    
    
    
    
    # plt.plot(collection_subset.absolute_timeline, collection_subset.get_data('w').loc[ts], '.', label="Original Series")
    # plt.plot(collection_subset.absolute_timeline, kd_ref - kd_calc , '.', label="Original Series")

    # plt.title(f"kd differences: {ts}")
    # # plt.legend()
    # plt.xlabel("Date")
    # plt.ylabel("Value")
    # plt.show()  
    
#     plt.plot(collection_subset.absolute_timeline, (collection_subset.get_data('u').loc[ts] - cplex_unwrapping.get_data('u').loc[ts]).astype(int), '.', label="Original Series")
#     # plt.plot(collection_subset.absolute_timeline, kd_ref - kd_calc , '.', label="Original Series")

#     plt.title(f"u differences: {ts}")
#     # plt.legend()
#     plt.xlabel("Date")
#     plt.ylabel("Value")
#     plt.show() 


############## Calculate Seasonality ++++++++++++++++++++++
# max_interval: is the max interval between measurements; 
# season_param = {"max_interval": 12, "averaging_window": 6}
# seasonality = Seasonality(collection_subset)
# seasonality.new(name = 'seasonality_limited') 
# seasonality.seasonality_calc(partial_unwrapping.get_data('u'), season_param)
 

# ############## Calculate SNR ++++++++++++++++++++++
# max_interval: is the max interval between measurements; 
# snr_param = {"cutoff_frequency": 10, "sampling_rate": 500}
# snr_p_unw = SNR(collection_subset)
# snr_p_unw.new(name = 'snr_p_unw') 
# snr_p_unw.snr_calc(partial_unwrapping.get_data('u'), snr_param)
# snr_p_unw.save() 


# ############## Calculate SNR ++++++++++++++++++++++
# max_interval: is the max interval between measurements; 
# snr_param = {"cutoff_frequency": 10, "sampling_rate": 500}
# snr_w = SNR(collection_subset)
# snr_w.new(name = 'snr_w') 
# snr_w.snr_calc(collection_subset.get_data('w'), snr_param)
# snr_w.save() 



# Open SNR
# snr_p_unw = SNR(collection_subset)
# snr_p_unw.load('snr_p_unw.unw')
# snr_w = SNR(collection_subset)
# snr_w.load('snr_w.unw')


############ STATISTICS #################

# # Example DataFrame
# data = {'column': [10, 20, 20, 30, 30, 30, 40, 50, 50, 60, 70, 80, 90, 100]}
# df = pd.DataFrame(data)

# # # Plot the statistical distribution
# # sns.histplot(snr_w.get_data('snr')['snr'], kde=True, bins=20, color='red')
# # sns.histplot(snr_p_unw.get_data('snr')['snr'], kde=True, bins=30, color='blue')

# bin_width = 1
# sns.histplot(snr_w.get_data('snr')['snr'], kde=True, binwidth=bin_width, color='red', label="Wrapped")
# sns.histplot(snr_p_unw.get_data('snr')['snr'], kde=True, binwidth=bin_width, color='blue', label="Partially Unwrapped")

# # Add labels and title
# plt.title('Statistical Distribution of SNR')
# plt.xlabel('SNR')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()







# # # Step 3: Plot seasonality for each series
# plt.figure(figsize=(6, 3))
# # for ts in ts_number_list:
# for ts in [56829]:
    
#     plt.plot(collection_subset.get_data('absolute_timeline'), collection_subset.get_data('w').loc[ts], '.', label="Original Series")

#     plt.plot(collection_subset.get_data('absolute_timeline'), partial_unwrapping.get_data('u').loc[ts], '.', label="Original Series")
#     # plt.plot(collection_subset.get_data('absolute_timeline'), seasonality.get_data('season').loc[ts] , linestyle="--", label="Season")
#     # plt.plot(collection_subset.get_data('absolute_timeline'), partial_unwrapping.get_data('u').loc[ts].to_numpy() - seasonality.get_data('season').loc[ts].to_numpy(), 'r.', label="Seasonal Component")

#     # plt.title(f"Seasonality for {ts}")
#     # plt.legend()
#     plt.xlabel("Date")
#     plt.ylabel("Value")
#     plt.show()
    


    
    
    

print(datetime.now()- starttime)
