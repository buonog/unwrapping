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
# B01 Context class and Interfaces definition; correction of strategies

from abc import ABC, abstractmethod
import numpy as np


class UnwrappingAlgo(ABC):
    """
    Abstract base class for unwrapping algorithms.

    Provides shared utilities and ensures input integrity for unwrapping implementations.
    """

    relative_timeline: np.ndarray

    def setRelativeTimeline(self, relative_timeline: np.array) -> None:
        """
        Sets and validates the relative timeline for unwrapping.

        Args:
            relative_timeline (np.array or list): Timeline values relative to the first acquisition.

        Raises:
            TypeError: If input is not a list or numpy array.
            ValueError: If input is empty or not strictly increasing.
        """
        if not isinstance(relative_timeline, (np.ndarray, list)):
            raise TypeError("relative_timeline must be a list or numpy array.")
        if len(relative_timeline) == 0:
            raise ValueError("relative_timeline cannot be empty.")
        
        timeline = np.array(relative_timeline)
        if not np.all(np.diff(timeline) > 0):
            raise ValueError("relative_timeline must be strictly increasing.")

        self.timeline = timeline
        self.ts_length = len(self.timeline)

    def generate_linear_function(self, x_start: int, x_end: int, y_start: float, y_end: float) -> np.array:
        """
        Generates a linear function between two timeline indices.

        Args:
            x_start (int): Start index.
            x_end (int): End index.
            y_start (float): Function value at start.
            y_end (float): Function value at end.

        Returns:
            np.array: Linear function values across the segment.
        """
        x = self.timeline[x_start:x_end + 1]
        m = (y_end - y_start) / (self.timeline[x_end] - self.timeline[x_start])
        q = y_start - m * self.timeline[x_start]
        return m * x + q

    def estimate_coherence(self, w: np.array, linear_function: np.array) -> float:
        """
        Computes phase coherence between wrapped data and model.

        Args:
            w (np.array): Wrapped phase signal.
            linear_function (np.array): Linear trend/model signal.

        Returns:
            float: Coherence score in [0, 1].
        """
        exp_phase_diff = np.exp(1j * (w - linear_function) * 2 * np.pi)
        return np.abs(np.sum(exp_phase_diff)) / len(w)

    def line_unwrap(self, w: np.array, line_variable: np.array) -> np.array:
        """
        Unwraps a signal along a model line.

        Args:
            w (np.array): Wrapped signal.
            line_variable (np.array): Model trend.

        Returns:
            np.array: Unwrapped signal.
        """
        return (line_variable - w + np.sign(line_variable[-1]) * 0.5).astype(int) + w

    @abstractmethod
    def unwrap(self, wrapped_timeseries: np.array, unwrap_param: dict) -> dict:
        """
        Abstract method. Validates input and defines unwrapping interface.

        Args:
            wrapped_timeseries (np.array): Wrapped signal to unwrap.
            unwrap_param (dict): Parameters for the algorithm.

        Returns:
            dict: {'m': model_fit, 'u': unwrapped_signal}

        Raises:
            TypeError: If input is not a NumPy array.
            ValueError: If its length doesn't match timeline.
        """
        if not isinstance(wrapped_timeseries, np.ndarray):
            raise TypeError("wrapped_timeseries must be a NumPy array.")
        if wrapped_timeseries.shape[0] != self.ts_length:
            raise ValueError(f"wrapped_timeseries length must be {self.ts_length} to match timeline.")

        model = []
        unwrapped = []
        return {'m': np.array(model), 'u': np.array(unwrapped)}
