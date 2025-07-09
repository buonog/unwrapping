#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unwrapper class
choose and manage the algorithm to unwrap

Created on Tue Nov 19 11:48:47 2024

@author: xap
"""

# Version
# ---------------------------- history ----------------------------
# A01 Context class and Interfaces definition;


import numpy as np   

from src.unwrapping_algo import UnwrappingAlgo
from src.unwrap_partial import PartialUnwrapAlgo
from src.unwrap_algo_PLLminlength import PLLAlgoMinLength
from src.unwrap_algo_PLLmaxcoherence import PLLAlgoMaxCoherence
from src.unwrap_algo_PLL_1 import PLLAlgo1
from src.unwrap_algo_PLL_2 import PLLAlgo2
from src.unwrap_algo_CPLEXbase2 import CPLEXbaseAlgo
from src.unwrap_algo_CPLEXstep import CPLEXstepAlgo
from src.unwrap_algo_CPLEXnoslope import CPLEXnoslopeAlgo
from src.unwrap_algo_CPLEXmean2 import CPLEXmeanAlgo
from src.unwrap_algo_CPLEXmean_poly import CPLEXmeanpolyAlgo
from src.unwrap_algo_CPLEXmean_season import CPLEXmeanSeasonAlgo
from src.unwrap_algo_PYOMOmean_season import PYOMOSeasonAlgo

from src.unwrap_algo_CPLEXloess import CPLEXloessAlgo

from src.unwrap_algo_CPLEXlocal import CPLEXlocalAlgo
from src.unwrap_algo_CPLEXrandom import CPLEXrandomAlgo
from src.unwrap_algo_CPLEXvariance import CPLEXvarianceAlgo
from src.unwrap_algo_CPLEXhybridrandom import CPLEXhybridrandomAlgo
from src.unwrap_algo_CPLEXhybridrandomglobal import CPLEXhybridrandomglobalAlgo





class UnwrappingContext():
    unwrapAlgoType: UnwrappingAlgo
    
    def __init__(self) -> None:
        self.unwrapAlgoType = ""
        
    def setStrategy(self, unwrapAlgoType: str) -> None:
        if unwrapAlgoType == "partial_unwrap":
            self.unwrapAlgoType = PartialUnwrapAlgo()
            
        elif unwrapAlgoType == "CPLEXbase_unwrap":
            self.unwrapAlgoType = CPLEXbaseAlgo()
            
        elif unwrapAlgoType == "CPLEXstep_unwrap":
            self.unwrapAlgoType = CPLEXstepAlgo()   
            
        elif unwrapAlgoType == "CPLEXnoslope_unwrap":
            self.unwrapAlgoType = CPLEXnoslopeAlgo()
            
        elif unwrapAlgoType == "CPLEXmean_unwrap":
            self.unwrapAlgoType = CPLEXmeanAlgo() 
            
        elif unwrapAlgoType == "CPLEXmeanpoly_unwrap":
            self.unwrapAlgoType = CPLEXmeanpolyAlgo() 
            
        elif unwrapAlgoType == "CPLEXmean_season_unwrap":
            self.unwrapAlgoType = CPLEXmeanSeasonAlgo() 
        
        elif unwrapAlgoType == "PYOMOmean_season_unwrap":
            self.unwrapAlgoType = PYOMOSeasonAlgo() 
            
        elif unwrapAlgoType == "CPLEXloess_unwrap":
            self.unwrapAlgoType = CPLEXloessAlgo() 
            
        elif unwrapAlgoType == "CPLEXlocal_unwrap":
            self.unwrapAlgoType = CPLEXlocalAlgo()
            
        elif unwrapAlgoType == "CPLEXrandom_unwrap":
            self.unwrapAlgoType = CPLEXrandomAlgo()   
            
        elif unwrapAlgoType == "CPLEXvariance_unwrap":
            self.unwrapAlgoType = CPLEXvarianceAlgo()  
            
        elif unwrapAlgoType == "CPLEXhybridrandom_unwrap":
            self.unwrapAlgoType = CPLEXhybridrandomAlgo() 
            
        elif unwrapAlgoType == "CPLEXhybridrandomglobal_unwrap":
            self.unwrapAlgoType = CPLEXhybridrandomglobalAlgo() 
            
        elif unwrapAlgoType == "PLLminlength_unwrap":
            self.unwrapAlgoType = PLLAlgoMinLength()
            
        elif unwrapAlgoType == "PLLmaxcoherence_unwrap":
            self.unwrapAlgoType = PLLAlgoMaxCoherence()
        else:
            print("***********Choose a correct unwrapping algorithm!!!")
        
    def setRelativeTimeline(self, relative_timeline)-> None:
        self.unwrapAlgoType.setRelativeTimeline(relative_timeline)
        
    # def setCursorIndex(self, relative_timeline)-> None:
    #     self.unwrapAlgoType.setCursorIndex(relative_timeline)

    def unwrap(self, wrapped_timeseries: np.array, parameters: dict) -> np.array:
        return self.unwrapAlgoType.unwrap(wrapped_timeseries, parameters)    
