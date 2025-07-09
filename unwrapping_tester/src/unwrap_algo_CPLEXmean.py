#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:31:01 2024

@author: xap
"""

import numpy as np
import matplotlib.pyplot as plt

import cplex
import docplex.mp
from docplex.mp.model import Model

from src.unwrapping_algo import UnwrappingAlgo


class CPLEXmeanAlgo(UnwrappingAlgo):
    """

    In this CPLEX-based unwrapping model, the optimization introduces two continuous variables:
        - a: a common central value (acts as a reference or 'center star' for all points)
        - b_slope: the slope of the linear trend over time
    
    Instead of minimizing pairwise differences between all data points (which would require a quadratic number of terms),
    the model simplifies the cost function by minimizing the squared difference between each corrected value and
    a single shared value 'a'. This significantly reduces computational complexity.
    
    Intuitively, 'a' serves as a centroid or anchor value that all corrected data points aim to be close to,
    which is particularly effective when the true signal has a relatively constant or slowly varying baseline.

    Optimization model:
        - Variables:
            - Continuous: a (common value), b_slope (slope of linear trend)
            - Integer: kd[t_k] (discrete unwrapping corrections per time step)
        - Objective:
            - Minimize the sum of squared differences between (g[t] - a)
              where g[t] = w[t] + b_slope * timeline[t] - kd[t]
        - Constraints:
            - kd[0] = 0
            - Bounds on variables defined by unwrap_param

    Attributes inherited from UnwrappingAlgo:
        timeline (np.array): Relative timeline for the signal.
        ts_length (int): Number of time steps.
    """

    def unwrap_plot(self, w, u_result, linear_fit, kd_ts):
        """
        Plots wrapped, unwrapped, and model-fitted time series.

        Args:
            w (np.array): Wrapped signal.
            u_result (list): Unwrapped signal.
            linear_fit (list): Linear model result from optimization.
            kd_ts (list): Estimated integer corrections applied to unwrap.
        """
        fig, axs = plt.subplots(2, 1, figsize=(5, 5))
        axs[0].plot(self.timeline, w , 'r.', alpha=0.7, label='Wrapped phase')
        axs[0].plot(self.timeline, u_result, 'b.', alpha=1, label='Unwrapped Phase')
        axs[0].plot(self.timeline, linear_fit, '-', color='orange', linewidth=3, label='Model velocity')
        axs[1].plot(self.timeline, kd_ts, 'g.', alpha=1, label='kd')
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        plt.show()

    def unwrap(self, w: np.array, unwrap_param: dict) -> dict:
        """
        Performs phase unwrapping using a linear CPLEX optimization model.

        Args:
            w (np.array): Wrapped input signal to be unwrapped.
            unwrap_param (dict): Dictionary containing algorithm parameters, must include:
                - 'max_slope' (int): Maximum allowed absolute slope for kd and model variables.

        Returns:
            dict: A dictionary containing:
                - 'm': np.array of the linear model fit
                - 'u': np.array of the unwrapped signal
        """
        node_limit = 20000
        max_slope = unwrap_param['max_slope']
        timeline_index = range(len(self.timeline))

        unw = Model(name='CPLEXmean')

        # --- Define discrete integer variables (kd) ---
        kd = unw.integer_var_list(name="kd", keys=self.ts_length, lb=-max_slope, ub=max_slope)
        unw.add_constraint(kd[0] == 0)

        # --- Define continuous variables ---
        a = unw.continuous_var(name="a_cost", lb=-100, ub=100)
        b_slope = unw.continuous_var(name="b_slope", lb=-100, ub=100)

        # --- Model equation ---
        g = [w[t_k] + b_slope * self.timeline[t_k] - kd[t_k] for t_k in timeline_index]

        # --- Objective function ---
        objective = unw.sum((g[k] - a) ** 2 for k in timeline_index)
        unw.minimize(objective)

        # --- CPLEX parameters ---
        unw.parameters.mip.limits.nodes = node_limit
        unw.parameters.timelimit = 60  # seconds

        # --- Solve ---
        solution = unw.solve()

        # --- Extract results ---
        kd_ts = [-kd[k].solution_value for k in timeline_index]
        u_result = [w[k] + kd_ts[k] for k in timeline_index]
        print("Value of a: ", a.solution_value)
        linear_fit = [-b_slope.solution_value * t_k for t_k in self.timeline]

        # Optional plot
        # self.unwrap_plot(w, u_result, linear_fit, kd_ts)

        return {'m': np.array(linear_fit), 'u': np.array(u_result)}

        
    
if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path
    from src.ts_collection import TSCollection, TSSubset
    from src.ts_packets import Unwrapping
    
    
    starttime = datetime.now()
    # generaldata_folder = "/mnt/DATI_PC/AA1_PROGETTI/PS_DATA/Real/"
    # Go three level up respect to current level
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    generaldata_folder = str(base_dir / "PS_DATA" / "Real") + '/'    
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
    cplex_unwrapping.new(unwrapping_name = 'cplexmean_test_subset',
                             unwrapping_algo = 'CPLEXmean_unwrap', unwrap_param = unwrap_param,
                             unwrapping_note = "used an integer slope")

    print("Time CPLEXmean_poly : ", datetime.now()- starttime)
    
    
    # # cplex_unwrapping.save()

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
        
