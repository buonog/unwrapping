#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:31:01 2024

@author: xap
"""

import numpy as np
# import pandas as pd
import random

import matplotlib.pyplot as plt

import cplex
import docplex.mp
from docplex.mp.model import Model

from src.unwrapping_algo import UnwrappingAlgo

  

class CPLEXvarianceAlgo(UnwrappingAlgo):
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
        # max_kd = 32

        max_slope = unwrap_param['max_slope']
        
        # timeline = self.timeline[unwrap_param['min_t_index']]
        timeline_index = range(len(self.timeline)) # index of timeline vector

        unw = Model(name='CPLEXbase')
        
        # **************** Define DISCRETE VARIABLES ******************************
        kd = unw.integer_var_list(name = "kd", keys = self.ts_length, lb = -max_slope, ub = max_slope)
        unw.add_constraint(kd[0] == 0) # # Set constraints for kd(0)     
       
        # **************** Define CONTINUOUS VARIABLES ******************************
        # a = unw.continuous_var(name = "a_cost", lb = -100, ub = 100)
        b_slope = unw.continuous_var(name = "b_slope", lb = -100, ub = 100)
        # b0 = unw.continuous_var(name = "b0", lb = -10, ub = 10)

        g = [w[t_k] + b_slope * self.timeline[t_k] - kd[t_k] for t_k in timeline_index]

        # **************** Define OBJECTIVE FUNCTION ******************************
        # objective = unw.sum((g[k] - a) ** 2 for k in timeline_index)
        # objective = unw.sum((g[k] - g[h]) ** 2 for k in timeline_index for h in timeline_index)
        d = unwrap_param['d']  # Definisci l'ampiezza dell'intorno
        
        g_expr = {t: w[t] + b_slope * self.timeline[t] - kd[t] for t in timeline_index}
        
        # Calcolo delle somme necessarie per la varianza
        sum_g  = unw.sum(g_expr[t] for t in timeline_index)
        sum_g2 = unw.sum(g_expr[t] * g_expr[t] for t in timeline_index)
        
        # Funzione obiettivo: minimizzare la varianza (a meno di costanti positive)
        objective = sum_g2 - (1.0/self.ts_length) * (sum_g * sum_g)

        # # Set minimization of the objective
        unw.minimize(objective)
        
        # # Set limit on explored nodes
        unw.parameters.mip.limits.nodes = node_limit  # Limita il numero massimo di nodi esplorati a 10.000
        
        # # Set time limits
        unw.parameters.timelimit = 60  # Limita il tempo di esecuzione a 60 secondi
        
        unw.print_information()
        
        # # Solve the model
        solution = unw.solve()
        
        kd_ts = [ -kd[k].solution_value for k in timeline_index]
        u_result = [w[k] + kd_ts[k] for k in timeline_index]
        # print("Value of a: ", a.solution_value)
        linear_fit = [-b_slope.solution_value * t_k   for t_k in self.timeline]
        
        # linear_fit = w
        self.unwrap_plot(w, u_result, linear_fit, kd_ts)
        
        return {'m': np.array(linear_fit), 'u': np.array(u_result)}
        