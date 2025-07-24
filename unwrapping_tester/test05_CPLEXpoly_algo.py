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
from src.ts_packets import Packet, Unwrapping, Seasonality, SNR



starttime = datetime.now()

# generaldata_folder = "/mnt/DATI_PC/AA1_PROGETTI/PS_DATA/Real/"
# collection_folder = "Toscana_2/"
# collection_file = "TOSCANA_ps.tsc"

# ### -------- TS Cazzaso
generaldata_folder = "/mnt/DATI_PC/AA1_PROGETTI/PS_DATA/Real/"
collection_folder = "Cazzaso_2/"
collection_file = "Cazzaso.tsc"

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


ts_number_list = [533]
# ts_number_list = list(range(0,9))
# ts_number_list = list(np.random.randint(tscollection.collection_size, size=3))
# ts_number_list = list(range(0,778))

# ts_number_list = [35623]

# ts_number_list = [77, 2453, 22303, 26966, 27831, 34247, 50477, 11453, 26071, 17560, 34567, 41138, 34348, 35205, 35623, 38047, 56539, 33646, 16476, 16729]
# ts_number_list = [77, 11453, 26071, 17560, 34348, 35205, 35623, 38047, 56539, 33646, 16476, 16729]

# ts_number_list = [50223, 40685, 37327, 65920, 16777, 16081, 16638, 19441, 73687]
# ts_number_list = list(range(0, tscollection.collection_size))

collection_subset = TSSubset(tscollection.get_collection_dict(), ts_number_list)

## ------------- DEFINE Polynomial Degree -------------------
pol_degree = 11

starttime1 = datetime.now()

# # # ############ GENERATE NEW CPLEXmean_UNWRAPPING ++++++++++++++++++++++++++++
# # # # # 1. Define the unwrapping object 2. Create new Unwrapping data
# unwrap_param = {'min_t_index': 0, 'max_t_index': 200, 'max_slope': 16, "n_cpu": 4}
# cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
# cplex_unwrapping.new(unwrapping_name = 'cplexmean_test_subset',
#                          unwrapping_algo = 'CPLEXmean_unwrap', unwrap_param = unwrap_param,
#                          unwrapping_note = "used an integer slope")
# # cplex_unwrapping.save()

# # # # ############ GENERATE NEW CPLEXmeanpoly_UNWRAPPING ++++++++++++++++++++++++++++
# # # # # # 1. Define the unwrapping object 2. Create new Unwrapping data
unwrap_param = {'min_t_index': 0, 'max_t_index': 300, 'max_slope': 16,
                'polynomial_degree': pol_degree, "n_cpu": 4}
cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
cplex_unwrapping.new(unwrapping_name = f'cplexmeanpoly_degree{pol_degree}_test_subset_reg',
                         unwrapping_algo = 'CPLEXmeanpoly_unwrap', unwrap_param = unwrap_param,
                         unwrapping_note=f"Used polynomial model with degree p = {pol_degree}")
cplex_unwrapping.save()


# print("Time CPLEX : ", datetime.now()- starttime)


# # # ############ OPEN UNWRAPPING ++++++++++++++++++++++++++++
cplex_unwrapping = Unwrapping(collection_subset)
cplex_unwrapping.load(f'cplexmeanpoly_degree{pol_degree}_test_subset_reg.unw')

###################### Calculate  χ² ###################################

ts_length = cplex_unwrapping.get_data('absolute_timeline').shape[0]
# print(ts_length)

residuals = cplex_unwrapping.get_data('u').to_numpy() - cplex_unwrapping.get_data('m').to_numpy()  # shape: (n, ts_length)
chi2 = np.sum(residuals**2, axis=1)                    # sum per riga
dof = ts_length - (pol_degree + 1)
chi2_reduced = chi2 / dof

# -- create and save packet with χ² results ---------
chi2_packets = {}

chi2_packets[f'pol_degree_{pol_degree}'] = Packet(collection_subset)
chi2_packets[f'pol_degree_{pol_degree}'].set_data('name', f'cplexmeanpoly_degree{pol_degree}_chi2')
chi2_packets[f'pol_degree_{pol_degree}'].set_data('note', f'chi2 measured for polynomial degree {pol_degree}')
chi2_packets[f'pol_degree_{pol_degree}'].set_data('chi2', chi2)
chi2_packets[f'pol_degree_{pol_degree}'].set_data('chi2_reduced', chi2_reduced)
chi2_packets[f'pol_degree_{pol_degree}'].save()

# ----- Load chi2 data ---------------------
chi2_poldegree_1_b = Packet(collection_subset)
chi2_poldegree_1_b.load(f'cplexmeanpoly_degree{pol_degree}_chi2')

chi2_poldegree_1_b.info

# df_result = pd.DataFrame({'chi2': chi2, 'chi2_reduced': chi2_reduced})

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



# # # # # ############ PLOT UNWRAPPING ++++++++++++++++++++++++++++
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
    plt.plot(collection_subset.absolute_timeline, -kd_ref, 'g.', label="Original Series")

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
