#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:31:01 2024

@author: xap
"""

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import cplex
import docplex.mp
from docplex.mp.model import Model

from src.unwrapping_algo import UnwrappingAlgo

  

class CPLEXmeanSeasonAlgo(UnwrappingAlgo):
    """Optimization algorithm based on CPLEX
        Linear Model
        Continuous variable: 2 (a and b_slope)
        Discrete variables: #ts_length  (kd(t_k))
        Total: ts_length + 1 variable
        
        Cost function: sum((g[k] - a) ** 2
        kd[0] = 0
    """
    # ts_length1 = 250

    
    def unwrap_plot(self, w, u_result, linear_fit, kd_ts):
        # # Plot wrapped and unwrapped function
        fig, axs = plt.subplots(2, 1, figsize=(5, 5))
        axs[0].plot(self.timeline, w , 'r.', alpha=0.7, label='Wrapped phase')
        axs[0].plot(self.timeline, u_result, 'b.', alpha=1, label='Unwrapped Phase')
        axs[0].plot(self.timeline, linear_fit, '-', color = 'orange', linewidth=3, label='Model velocity')
        axs[1].plot(self.timeline, kd_ts, 'g.', alpha=1, label='kd')
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        # axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
        # axs[1].grid(True)
        # axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
        # plt.savefig("f_013a.png")
        plt.show

    def unwrap(self, w: np.array, unwrap_param: dict):
        
        node_limit = 20000
        tyear = 365
        # max_kd = 32

        max_slope = unwrap_param['max_slope']
        # self.timeline = self.timeline[0:10]
        # self.ts_length = 10
        timeline = self.timeline[unwrap_param['min_t_index']]
        timeline_index = range(len(self.timeline)) # index of timeline vector

        # local_timeline = [x / tyear for x in range(0, self.ts_length)]
        # cos_term = np.cos(local_timeline)
        # sin_term = np.sin(local_timeline)
        
        # local_timeline = np.array([x / tyear for x in range(self.ts_length)], dtype=float)


        
        cos_term = np.cos(self.timeline / tyear)
        sin_term = np.sin(self.timeline / tyear)
        # cos_term = np.cos(local_timeline2)

        # sin_term = np.sin(local_timeline1)
        # print(local_timeline)
        
        # cos_term = local_timeline1
        # sin_term = local_timeline2
        
        # cos_term = np.cos(np.random.rand(self.ts_length))
        # sin_term = np.sin(np.random.rand(self.ts_length))
        
        unw = Model(name='CPLEXbase')
        
        # **************** Define DISCRETE VARIABLES ******************************
        kd = unw.integer_var_list(name = "kd", keys = self.ts_length, lb = -max_slope, ub = max_slope)
        unw.add_constraint(kd[0] == 0) # # Set constraints for kd(0)     
       
        # **************** Define CONTINUOUS VARIABLES ******************************
        # Define continuous variables:
        a = unw.continuous_var(name="a_cost", lb=-100, ub=100)
        # Reparameterized seasonal variables:
        X = unw.continuous_var(name="X", lb=-0.2, ub=0.2)  # corresponds to A*cos(phi)
        Y = unw.continuous_var(name="Y", lb=-0.2, ub=0.2)  # corresponds to -A*sin(phi)
        b_slope = unw.continuous_var(name="b_slope", lb=-100, ub=100)
        
        # epsilon = unw.continuous_var(name="epsilon", lb=-100, ub=100)
        
        reg = [b_slope**2 + X**2 + Y**2 + kd[t_k]**2 for t_k in timeline_index]
        epsilon = 10e-8
        # Assume timeline_index is defined (e.g., range(len(self.timeline)))
        # g = [b_slope * self.timeline[t_k] + X * cos_term[t_k] + Y * sin_term[t_k] - kd[t_k] for t_k in timeline_index]
        
        g = [w[t_k] + b_slope * self.timeline[t_k] + 
             X * cos_term[t_k] + Y * sin_term[t_k] - kd[t_k] 
             for t_k in timeline_index]
        
        # g = [w[t_k] + b_slope * self.timeline[t_k] + 
        #      X + Y - kd[t_k] 
        #      for t_k in timeline_index]
        
        # **************** Define OBJECTIVE FUNCTION ******************************
        objective = unw.sum((g[k] - a)**2 + epsilon * reg[k] for k in timeline_index)
        unw.minimize(objective)

        # # Set limit on explored nodes
        unw.parameters.mip.limits.nodes = node_limit  # Limita il numero massimo di nodi esplorati a 10.000
        
        # # Set time limits
        unw.parameters.timelimit = 60  # Limita il tempo di esecuzione a 60 secondi
        # unw.print_information()
        
        # # Solve the model
        solution = unw.solve()
        
        kd_ts = [ -kd[k].solution_value for k in timeline_index]
        u_result = [w[k] + kd_ts[k] for k in timeline_index]
        print("Value of a: ", a.solution_value)
        linear_fit = [-b_slope.solution_value * t_k   for t_k in self.timeline]
        
        # linear_fit = w
        # self.unwrap_plot(w, u_result, linear_fit, kd_ts)
        
        return {'m': np.array(linear_fit), 'u': np.array(u_result)}


if __name__ == "__main__":
    from datetime import datetime
    import numpy as np
    import seaborn as sns
    from src.ts_collection import TSCollection, TSSubset
    from src.ts_packets import Unwrapping, Seasonality, SNR
    
    starttime = datetime.now()
    generaldata_folder = "/mnt/DATI_PC/AA1_PROGETTI/PS_DATA/Real/"
    collection_folder = "Toscana_2/"
    collection_file = "TOSCANA_ps.tsc"

    tscollection = TSCollection()
    tscollection.load(generaldata_folder + collection_folder + collection_file)

    ts_number_list = [22303]
    # ts_number_list = [77, 11453, 26071, 17560, 34348, 35205, 35623, 38047, 56539, 33646, 16476, 16729]

    collection_subset = TSSubset(tscollection.get_collection_dict(), ts_number_list)
    starttime1 = datetime.now()


    # # ############ GENERATE NEW CPLEXmean_UNWRAPPING ++++++++++++++++++++++++++++
    # # # # 1. Define the unwrapping object 2. Create new Unwrapping data
    unwrap_param = {'min_t_index': 0, 'max_t_index': 200, 'max_slope': 16, "n_cpu": 4}
    cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
    cplex_unwrapping.new(unwrapping_name = 'cplexmean_season_test_subset',
                             unwrapping_algo = 'CPLEXmean_season_unwrap', unwrap_param = unwrap_param,
                             unwrapping_note = "used an integer slope")
    # # cplex_unwrapping.save()


    print("Time CPLEXmean_poly : ", datetime.now()- starttime)

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
        

        plt.plot(collection_subset.absolute_timeline, wt, '.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, ut, 'g.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, -kd_ref, 'r.', label="Original Series")

        plt.title(f"Reference: {ts}")
        # plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()
        
        plt.plot(collection_subset.absolute_timeline, wc, '.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, uc, '.', label="Original Series")
        # plt.plot(collection_subset.absolute_timeline, cplex_unwrapping.get_data('m').loc[ts], '.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, -kd_calc, '.', label="Original Series")

        plt.title(f"Unwrapping: {ts}")
        # plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()