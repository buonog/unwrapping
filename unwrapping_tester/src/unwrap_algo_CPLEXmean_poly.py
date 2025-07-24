#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:42:43 2024

@author: xap

    Optimization algorithm based on CPLEX
   
        Polynomial Model of order p
        Continuous variable: 1 + p (a and b1, b2, ...bp)
        Discrete variables: #ts_length  (kd(t_k))
        Total: ts_length + 1 + p variable
        
        Cost function: sum((g[k] - a) ** 2)
        kd[0] = 0
    

"""

import numpy as np
import matplotlib.pyplot as plt

import cplex
import docplex.mp
from docplex.mp.model import Model

from src.unwrapping_algo import UnwrappingAlgo

class CPLEXmeanpolyAlgo(UnwrappingAlgo):
    """Optimization algorithm based on CPLEX
        MULTI-CPU
        Polynomial Model
        Continuous variable: n (a and b1, b2, ...)
        Discrete variables: #ts_length  (kd(t_k))
        Total: ts_length + n+1 variable
        
        Cost function: sum((g[k] - a) ** 2)
        Constraint: kd[0] = 0
    """
    
    def unwrap_plot(self, w, u_result, linear_fit, kd_ts):
        # Plot wrapped and unwrapped function
        fig, axs = plt.subplots(2, 1, figsize=(5, 5))
        axs[0].plot(self.timeline, w, 'r.', alpha=0.7, label='Wrapped phase')
        axs[0].plot(self.timeline, u_result, 'b.', alpha=1, label='Unwrapped Phase')
        axs[0].plot(self.timeline, linear_fit, '-', color='orange', linewidth=3, label='Model velocity')
        axs[1].plot(self.timeline, kd_ts, 'g.', alpha=1, label='kd')
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        # plt.savefig("f_013a.png")
        plt.show()
        
    def build_polynomial_model(self, unw, p, timeline_norm, w, a, kd, timeline_index):
        """
        Creates a model with a polynomial of order p:
        - Defines continuous variables b1...bp
        - Normalizes the timeline
        - Builds the regularization term and the constraint function g
        """
        
        # Crea le variabili b1, ..., bp
        b_vars = [
            unw.continuous_var(name=f"b{i}", lb=-10, ub=10) for i in range(1, p + 1)
        ]
    
        # Termine di regolarizzazione: a^2 + somma dei bi^2 + somma kd^2
        reg = a**2 + sum(b**2 for b in b_vars) + sum(kd[t_k]**2 for t_k in timeline_index)
    
        # Funzione g[t_k] = w[t_k] + b1*t + b2*t^2 + ... + bp*t^p - kd[t_k]
        g = []
        for t_k in timeline_index:
            poly_sum = sum(b_vars[j] * timeline_norm[t_k]**(j+1) for j in range(p))
            g.append(w[t_k] + poly_sum - kd[t_k])
    
        return b_vars, reg, g

    def reduced_chi_squared(self, u_result, poly_fit, pol_degree):
        residuals = np.array(u_result) - np.array(poly_fit)
        N = self.ts_length
        p = pol_degree + 1  # numero parametri stimati (a + b1...bp)
        dof = N - p
        chi2 = np.sum(residuals**2)
        # print("χ²: ", chi2)
        chi2_ridotto = chi2 / dof
        # print(f"pol_degree={pol_degree}, χ²={chi2:.4f}; χ² ridotto = {chi2_ridotto:.4f}")
        print(f"pol_degree={pol_degree}; {chi2:.4f}; {chi2_ridotto:.4f}")

        return chi2 / dof

    def unwrap(self, w: np.array, unwrap_param: dict):
        node_limit = 20000
        max_slope = unwrap_param['max_slope']
        # max_slope = 100
        pol_degree = unwrap_param['polynomial_degree']
        
        # Creazione dell'indice per la timeline
        timeline_index = range(len(self.timeline))
        
        # Normalizzazione della timeline to have a better conditioned matrix
        t_min = self.timeline.min()
        t_max = self.timeline.max()
        timeline_norm = (self.timeline - t_min) / (t_max - t_min)
       
        # Creazione del modello CPLEX
        unw = Model(name='CPLEXbase')
        
        # **************** Define DISCRETE VARIABLES ******************************
        kd = unw.integer_var_list(name="kd", keys=self.ts_length, lb=-max_slope, ub=max_slope)
        unw.add_constraint(kd[0] == 0)  # vincolo: kd[0] = 0     
        # Define central node parameter
        a = unw.continuous_var(name="a_cost", lb=-100, ub=100)
        # Define polynomial coefficients
        b_vars, reg, g = self.build_polynomial_model(unw, p=pol_degree, timeline_norm=timeline_norm , w=w, a=a, kd=kd, timeline_index=timeline_index)

        epsilon = 10e-10


        # **************** Define OBJECTIVE FUNCTION ******************************
        objective = unw.sum((g[k] - a) ** 2  for k in timeline_index)
        # objective = unw.sum((g[k] - a) ** 2  for k in timeline_index) + epsilon * reg
        # objective = unw.sum((g[k] - a) ** 2  for k in timeline_index[0:]) + epsilon * reg

        unw.minimize(objective)
        
        # Impostazione dei parametri di CPLEX: limite sul numero di nodi e tempo massimo di esecuzione
        unw.parameters.mip.limits.nodes = node_limit
        unw.parameters.timelimit = 60
        
        # Introduzione del supporto multi-CPU: se specificato in unwrap_param, settiamo il numero di thread
        n_cpu = unwrap_param.get("n_cpu", None)
        if n_cpu is not None:
            unw.parameters.threads = n_cpu
        
        # unw.print_information()
        
        # Risoluzione del modello
        solution = unw.solve()
        
        # Estrazione dei risultati dalla soluzione
        kd_ts = [-kd[k].solution_value for k in timeline_index]
        u_result = [w[k] + kd_ts[k] for k in timeline_index]
        # print("Value of a: ", a.solution_value)
        
        # Generate the fit curve
        poly_fit = [-sum(b_vars[i].solution_value * t_k**(i+1) 
                    for i in range(len(b_vars)))    
                    for t_k in timeline_norm]
        

        
        # chi2_ridotto = self.reduced_chi_squared(u_result, poly_fit, pol_degree)
        # print(f"pol_degree={pol_degree}, χ² ridotto = {chi2_ridotto:.4f}")
        
        #5
        # poly_fit = [-b1.solution_value * t_k - b2.solution_value * t_k**2 - b3.solution_value * t_k**3 - b4.solution_value * t_k**4 - b5.solution_value * t_k**5 for t_k in timeline_norm]

        # self.unwrap_plot(w, u_result, poly_fit, kd_ts)
        
        return {'m': np.array(poly_fit), 'u': np.array(u_result)}


if __name__ == "__main__":
    from datetime import datetime
    import numpy as np
    import seaborn as sns
    from src.ts_collection import TSCollection, TSSubset
    from src.ts_packets import Unwrapping
    
    starttime = datetime.now()
    generaldata_folder = "/mnt/DATI_PC/AA1_PROGETTI/PS_DATA/Real/"
    collection_folder = "Toscana_2/"
    collection_file = "TOSCANA_ps.tsc"

    tscollection = TSCollection()
    tscollection.load(generaldata_folder + collection_folder + collection_file)

    ts_number_list = [16476]
    # ts_number_list = [77, 11453, 26071, 17560, 34348, 35205, 35623, 38047, 56539, 33646, 16476, 16729]

    collection_subset = TSSubset(tscollection.get_collection_dict(), ts_number_list)
    starttime1 = datetime.now()


    # # ############ GENERATE NEW CPLEXmeanpoly_UNWRAPPING ++++++++++++++++++++++++++++
    # # # # 1. Define the unwrapping object 2. Create new Unwrapping data
    unwrap_param = {'min_t_index': 0, 'max_t_index': 200, 'max_slope': 16, 'polynomial_degree': 2, "n_cpu": 4}
    cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
    cplex_unwrapping.new(unwrapping_name = 'cplexmeanpoly_test_subset',
                             unwrapping_algo = 'CPLEXmeanpoly_unwrap', unwrap_param = unwrap_param,
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
        plt.plot(collection_subset.absolute_timeline, cplex_unwrapping.get_data('m').loc[ts], '.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, -kd_calc, '.', label="Original Series")

        plt.title(f"Unwrapping: {ts}")
        # plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()
        

    print(datetime.now()- starttime)
