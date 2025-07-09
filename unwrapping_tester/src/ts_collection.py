#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:33:35 2024

@author: xap
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# from statsmodels.tsa.seasonal import seasonal_decompose
import src.ts_input_templates as tsInput


class TSCollection():
    """
    Class for managing a collection of time series packets (PS time series).

    A TSCollection stores a dictionary containing numpy arrays representing
    wrapped phase (`w`), unwrapped phase (`u`), unwrapping offsets (`kd`), 
    and timeline information (`absolute_timeline`, etc.).

    Attributes:
        name (str): Name of the collection.
        folder (str): Path to the folder containing the data.
        timeseriesLength (int): Length of each time series (not initialized here).
        collection_size (int): Number of time series in the collection.
        collection_dict (dict): Dictionary storing time series data.
        collection_extension (str): File extension used for saving collections.
    """

    name: str 
    folder: str
    timeseriesLength: int 
    collection_size: int
    collection_dict: dict
    collection_extension: str

    def __init__(self) -> None:
        """Initializes an empty TSCollection."""
        self.collection_size = 0
        collection_extension = '.tsc'

    def load(self, collection_file: str):
        """
        Loads a collection of time series from a file.

        Args:
            collection_file (str): Path to the collection file to load.

        Raises:
            ValueError: If the file cannot be read properly.
        """
        try:
            reader = tsInput.get_data_reader(collection_file)
            self.collection_dict = reader.read_data(collection_file)
            print("\n\nFile read successfully!")
            self.info

            self.name = self.collection_dict['name']
            self.folder = self.collection_dict['folder']
            self.collection_size = len(self.collection_dict['w'])
        except ValueError as InputError:
            print(InputError) 
        return None

    def save(self, collection_file: str) -> None:
        """
        Saves the current collection to a file in pickle format.

        Args:
            collection_file (str): Path (without extension) where to save the collection.
        """
        try:
            pd.to_pickle(self.collection_dict, collection_file + self.collection_extension)
        except ValueError as OutputError:
            print(OutputError) 

    def get_collection_dict(self) -> dict:
        """
        Returns:
            dict: The entire dictionary of the collection.
        """
        return self.collection_dict

    def get_data(self, attribute: str) -> np.array:
        """
        Returns the specified data array from the collection.

        Args:
            attribute (str): Key of the attribute to retrieve (e.g., 'w', 'u').

        Returns:
            np.array: The corresponding array from the collection.
        """
        return self.collection_dict[attribute]

    @property
    def absolute_timeline(self) -> np.array:
        """
        Returns:
            np.array: The absolute timeline associated with the collection.
        """
        return self.collection_dict['absolute_timeline']

    @property
    def info(self) -> list:
        """
        Prints an overview of the keys in the collection dictionary, including types and shapes.

        Returns:
            list: [Printed output only; no return value used]
        """
        print("Keys in the data dictionary:")
        for key, value in self.collection_dict.items():
            if isinstance(value, (np.ndarray, pd.Series)):
                print(f"  {key}: {type(value).__name__} with shape {value.shape}")
            else:
                print(f"  {key}: {value}")


class TSSubset(TSCollection):
    """
    Subclass of TSCollection representing a subset of time series.

    Creates a new collection object that includes only selected time series
    from the original collection.

    Attributes inherited from TSCollection:
        collection_dict, folder, name, etc.

    Args:
        collection (dict): The full collection dictionary.
        ts_list (list): List of indices of time series to include in the subset.
    """

    def __init__(self, collection: dict, ts_list: list):
        """
        Initializes a subset of a collection using specified time series.

        Args:
            collection (dict): Dictionary containing the full collection data.
            ts_list (list): List of time series indices to extract.

        Raises:
            KeyError: If required keys are missing in the original collection.
        """
        self.collection_dict = {}
        self.folder = collection['folder']
        self.name = collection['name']

        self.collection_dict['absolute_timeline'] = collection['absolute_timeline']
        self.collection_dict['relative_timeline'] = collection['relative_timeline']

        self.collection_dict['w'] = pd.DataFrame(collection['w'][ts_list], index=ts_list)
        self.collection_dict['u'] = pd.DataFrame(collection['u'][ts_list], index=ts_list)
        self.collection_dict['kd'] = pd.DataFrame(collection['kd'][ts_list], index=ts_list)
        print("Subcollection created successfully!")



# class PSSelection():
#     ps: list 
    
#     def __init__(self, collection: PSCollection, selection: list)-> None:
#         self.ps = []
#         if selection == None:
#             selection = range(len(collection['w']))
        
#         for ps_number in selection:
#             new_ps = PS(collection, ps_number)
#             self.ps.append(new_ps)
            
#     def get_selection(self)-> list:
#         return self.ps
        
# class PS():
#     ps_number: int
#     w: np.array 
#     u: np.array
#     kd: np.array
#     unwrap: dict
    
#     def __init__(self, collection: dict, ps_number: int):
#         self.ps_number = ps_number
#         self.w = collection['w'][ps_number]
#         self.u = collection['u'][ps_number]
#         self.kd = collection['kd'][ps_number]
#         self.unwrap = {}
        
#     def unwrap_add(self, unwrap_key: str, unwrapped_timeseries: np.array)->None:
#         self.unwrap[unwrap_key: unwrapped_timeseries]
#         return 0



