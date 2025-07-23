#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:33:35 2024

@author: xap
"""
import numpy as np
import pandas as pd
# from src.ts_collection import TSCollection
import src.sts_input_strategies as stsInput
import src.sts_packet as stsPacket


class InvalidDuration(Exception):
    pass


class synthTSCollection():
    """Subclass of TSCollection
    this class is strictrly connected to a file with all the parameters
    used to generate the collection. 
    Attributes: 
        name, timeseriesLength
        np.array m, np.array n, np.array season,
        np.array u, np.array w
        packets_param_list (list of dictionary; each dictionary includes
                            parameters for a single packet"""
    # From superclass
    name: str 
    folder: str
    timeseriesLength: int 
    collection_size: int
    collection_dict : dict
    collection_extension: str
    # packets_param_list: list

    
    def __init__(self) -> None:
        self.collection_size = 0
        self.collection_dict = {}
        self.collection_extension = '.stsc'
        self.collection_dict['generation_params_list'] = []
        # self.packets_param_list = []
        
    def open(self, collection_file: str)-> None:

        try:
            collection_dict = pd.read_pickle(collection_file)
        except ValueError as InputError:
            print(InputError) 
        
        try:
            # self.packets_param_list = collection_dict["parameters"]
            self.collection_dict = collection_dict
            self.collection_size = len(self.timeline)
        except ValueError as InputError:
            print(InputError)
            
    def save(self, collection_file: str)-> None:
        # print("eccoci qui")
        try:
            # Save to a pickle file
            self.collection_dict['collection_extension'] = self.collection_extension
            pd.to_pickle(self.collection_dict, collection_file + self.collection_extension)
        except ValueError as OutputError:
            print(OutputError) 

    def generate(self, collection_param_file: str)-> None:
        file_name = collection_param_file
        # packet_list = []
        
        # read packets parameters
        try:
            reader = stsInput.get_data_reader(file_name)
            # param_data = reader.read_data(file_name)
            self.collection_dict = reader.read_data(file_name)
            
            # extract collection parameters
            # self.name = param_data['collection']['collection_name']
            # self.timeseriesLength = param_data['collection']['timeseriesLength']
            self.name = self.collection_dict['name']
            self.timeseriesLength = self.collection_dict['timeseriesLength']
            
            self.collection_dict['timeline'] = np.empty((0, self.timeseriesLength))
            self.collection_dict['m'] = np.empty((0, self.timeseriesLength))
            self.collection_dict['n'] = np.empty((0, self.timeseriesLength))
            self.collection_dict['season'] = np.empty((0, self.timeseriesLength))
            self.collection_dict['u'] = np.empty((0, self.timeseriesLength))
            self.collection_dict['w'] = np.empty((0, self.timeseriesLength))
            self.collection_dict['kd'] = np.empty((0, self.timeseriesLength))
            
            # extract packets parameters list
            # self.collection_dict['generation_params_list'] = param_data['packets_list']
        except ValueError as InputError:
            print(InputError) 
        
        try:
            # for packet_index in range(len(self.packets_param_list)):
            for packet_index in range(len(self.collection_dict['generation_params_list'])):
                packet_param = self.collection_dict['generation_params_list'][packet_index]
                
                # instantiate packet and packetbuilder
                new_packet = stsPacket.TSPacket(packet_param["packet"]["size"], self.timeseriesLength, packet_param)
                new_packet_builder = stsPacket.PacketBuilder(new_packet)
                new_packet_builder.build()
  
                self.collection_dict['timeline'] = np.vstack((self.collection_dict['timeline'], new_packet.packet_dict['timeline']))
                self.collection_dict['m'] = np.vstack((self.collection_dict['m'], new_packet.packet_dict['m']))
                self.collection_dict['n'] = np.vstack((self.collection_dict['n'], new_packet.packet_dict['n']))
                self.collection_dict['season'] = np.vstack((self.collection_dict['season'], new_packet.packet_dict['season']))
                self.collection_dict['u'] = np.vstack((self.collection_dict['u'], new_packet.packet_dict['u']))
                self.collection_dict['w'] = np.vstack((self.collection_dict['w'], new_packet.packet_dict['w']))
                self.collection_dict['kd'] = np.vstack((self.collection_dict['kd'], new_packet.packet_dict['kd']))
                
                packet_param['kd'] = new_packet.calculate_kd_maxmin()

                # print('collection dim= ', len(self.collection_dict['m']))

            # collection_size is the number of timeseries in the collection 
            self.collection_size = len(self.collection_dict['timeline'])
        except ValueError as InputError:
            print(InputError) 

            
    def get_collection_dict(self)-> dict:
        return self.collection_dict
    
    def get_data(self, attribute: str)-> np.array:
        return self.collection_dict[attribute]
    
    def add_timeseries_value(self, attribute: str, value)-> np.array:
        self.collection_dict[attribute] = np.vstack((self.collection_dict[attribute], value))
    
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

    def set_collection_param(self, packets_param_list: list)-> None:
        self.collection_dict['generation_params_list'] = packets_param_list
        
    def get_collection_param(self, packets_param_list: list)-> None:
        self.collection_dict['generation_params_list'] = packets_param_list
    
    # def set_timeline(self, timeline: np.array)-> None:
    #     self.timeline = np.vstack((self.timeline, timeline))
    
    # def set_m(self, m: np.array)-> None:
    #     self.m = np.vstack((self.m, m))
        
    # def set_n(self, n: np.array)-> None:
    #     self.n = np.vstack((self.n, n))
    
    # def set_season(self, season: np.array)-> None:
    #     self.season = np.vstack((self.season, season))
    
    # def set_u(self, u: np.array)-> None:
    #     self.u = np.vstack((self.u, u))
        
    # def set_w(self, w: np.array)-> None:
    #     self.w = np.vstack((self.w, w))            

    # def set_kd(self, kd: np.array)-> None:
    #     self.kd = np.vstack((self.kd, kd)) 
    

    
    # def get_timeline(self)-> np.array:
    #     return self.timeline
    
    # def get_m(self)-> np.array:
    #     return self.m
        
    # def get_n(self)-> np.array:
    #     return self.n
    
    # def get_season(self)-> None:
    #     return self.season
    
    # def get_u(self)-> np.array:
    #     return self.u 
        
    # def get_w(self)-> np.array:
    #     return self.w          

    # def get_kd(self)-> np.array:
    #     return self.kd



