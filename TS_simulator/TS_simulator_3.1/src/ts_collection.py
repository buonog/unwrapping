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
    """class that includes more Packets 
        (the collections of PS timeseries):
        a dictionary includes the following numpy arrays:
            w, u, kd, timeline,..."""
    name: str 
    folder: str
    timeseriesLength: int 
    collection_size: int
    collection_dict: dict
    collection_extension: str
    
    def __init__(self) -> None:
        self.collection_size = 0
        collection_extension = '.tsc'

    def load(self, collection_file: str):
        """Read data from file and calculates kd and relative_timeline """
        try:
            reader = tsInput.get_data_reader(collection_file)
            self.collection_dict = reader.read_data(collection_file)
            
            # Mostra i risultati in console
            print("\n\nFile read successfully!")
            self.info

            self.name = self.collection_dict['name']
            self.folder = self.collection_dict['folder']
            self.collection_size = len(self.collection_dict['w'])
        except ValueError as InputError:
            print(InputError) 
            
        return None

    def save(self, collection_file: str)-> None:
        # print("eccoci qui")
        try:
            # Save to a pickle file
            pd.to_pickle(self.collection_dict, collection_file + self.collection_extension)
        except ValueError as OutputError:
            print(OutputError) 
    
    def get_collection_dict(self)-> dict:
        return self.collection_dict
    
    def get_data(self, attribute: str)-> np.array:
        return self.collection_dict[attribute]
    
    @property
    def absolute_timeline(self) -> np.array:
        return self.collection_dict['absolute_timeline']
    
    @property
    def info(self) -> list:
        print("Keys in the data dictionary:")
        for key, value in self.collection_dict.items():
            if isinstance(value, (np.ndarray, pd.Series)):
                print(f"  {key}: {type(value).__name__} with shape {value.shape}")
            else:
                # pass
                print(f"  {key}: {value}")

 
    


class TSSubset(TSCollection):
    # absolute_timeline: np.array
    # relative_timeline: np.array
    # w: np.array
    # u: np.array
    # kd: np.array 
    
    def __init__(self, collection: dict, ts_list: list):
        self.collection_dict = {}
        self.folder = collection['folder']
        self.name = collection['name']
        self.collection_dict['absolute_timeline'] = collection['absolute_timeline']
        self.collection_dict['relative_timeline'] = collection['relative_timeline']

        self.collection_dict['w'] = pd.DataFrame(collection['w'][ts_list], index=ts_list)
        self.collection_dict['u'] = pd.DataFrame(collection['u'][ts_list], index=ts_list)
        self.collection_dict['kd'] = pd.DataFrame(collection['kd'][ts_list], index=ts_list)




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



