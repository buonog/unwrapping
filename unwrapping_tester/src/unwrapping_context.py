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


# Factory dictionary mapping algorithm names to their class constructors
ALGO_FACTORY = {
    "partial_unwrap": PartialUnwrapAlgo,
    "CPLEXbase_unwrap": CPLEXbaseAlgo,
    "CPLEXstep_unwrap": CPLEXstepAlgo,
    "CPLEXnoslope_unwrap": CPLEXnoslopeAlgo,
    "CPLEXmean_unwrap": CPLEXmeanAlgo,
    "CPLEXmeanpoly_unwrap": CPLEXmeanpolyAlgo,
    "CPLEXmean_season_unwrap": CPLEXmeanSeasonAlgo,
    "PYOMOmean_season_unwrap": PYOMOSeasonAlgo,
    "CPLEXloess_unwrap": CPLEXloessAlgo,
    "CPLEXlocal_unwrap": CPLEXlocalAlgo,
    "CPLEXrandom_unwrap": CPLEXrandomAlgo,
    "CPLEXvariance_unwrap": CPLEXvarianceAlgo,
    "CPLEXhybridrandom_unwrap": CPLEXhybridrandomAlgo,
    "CPLEXhybridrandomglobal_unwrap": CPLEXhybridrandomglobalAlgo,
    "PLLminlength_unwrap": PLLAlgoMinLength,
    "PLLmaxcoherence_unwrap": PLLAlgoMaxCoherence,
    # "PLL1_unwrap": PLLAlgo1,
    # "PLL2_unwrap": PLLAlgo2,
}


class UnwrappingContext:
    """
    Context class for managing and applying unwrapping strategies.

    Attributes:
        unwrapAlgoType (UnwrappingAlgo): The current unwrapping algorithm instance.
    """

    unwrapAlgoType: UnwrappingAlgo

    def __init__(self) -> None:
        """Initializes the context with no selected algorithm."""
        self.unwrapAlgoType = None

    def setStrategy(self, unwrapAlgoType: str) -> None:
        """
        Selects the unwrapping algorithm strategy using a factory dictionary.

        Args:
            unwrapAlgoType (str): The identifier of the desired algorithm.

        Raises:
            ValueError: If the algorithm name is not recognized.
        """
        try:
            self.unwrapAlgoType = ALGO_FACTORY[unwrapAlgoType]()
        except KeyError:
            raise ValueError(f"Unwrapping algorithm '{unwrapAlgoType}' not supported. "
                             f"Available options: {list(ALGO_FACTORY.keys())}")

    def setRelativeTimeline(self, relative_timeline) -> None:
        """
        Sets the relative timeline used by the unwrapping algorithm.

        Args:
            relative_timeline (array-like): Timeline values relative to the first acquisition.
        """
        if not self.unwrapAlgoType:
            raise RuntimeError("Unwrapping algorithm not set. Call setStrategy() first.")
        self.unwrapAlgoType.setRelativeTimeline(relative_timeline)

    def unwrap(self, wrapped_timeseries: np.ndarray, parameters: dict) -> dict:
        """
        Applies the current unwrapping algorithm to a wrapped time series.

        Args:
            wrapped_timeseries (np.ndarray): The wrapped input time series.
            parameters (dict): Parameters specific to the algorithm.

        Returns:
            dict: A dictionary with keys 'u' (unwrapped) and 'm' (model).
        """
        if not self.unwrapAlgoType:
            raise RuntimeError("Unwrapping algorithm not set. Call setStrategy() first.")
        return self.unwrapAlgoType.unwrap(wrapped_timeseries, parameters)

    @staticmethod
    def list_algorithms():
        """Prints and returns the list of available algorithms."""
        print("Available unwrapping algorithms:")
        for name in ALGO_FACTORY.keys():
            print(f"  - {name}")
        return list(ALGO_FACTORY.keys())
    
