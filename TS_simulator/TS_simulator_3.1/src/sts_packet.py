#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:33:35 2024

@author: xap
"""
import numpy as np
import src.sts_models as stsModels

class InvalidDuration(Exception):
    pass

class Timeline():
     
    """ Time measurement sequence; measured in days from the first meaasurement
    attributes: length of the vector, total duration in days, list of 
    measurements time
    """
    length: int
    timeline_param: dict
    measurements: np.array
    
    def __init__(self, length, timeline_param) -> list:
        self.length = length
        self.timeline_param = timeline_param
        self.measurements = [0]
        self.duration = 0
        
    def get_measurements(self) -> tuple:
        ts_index = 0
        toggle = np.random.choice([0, 1])
        toggle_values = [1,0]
        while(ts_index < self.length - 1):
            
            new_date = min(np.random.geometric(p=self.timeline_param["timeline_change_prob"])
                                + self.timeline_param["timeline_change_min"],
                                self.length - ts_index - 1)
            
            # toggle between 1 and 2 : to simulate measurement step of 1 or 2 day periods)
            toggle = toggle_values[toggle]
            incremental_value = 1 + toggle
            
            #create the vector of incremental measurement events 
            # (every value is the time interval value in days)
            self.measurements += ([incremental_value * 
                                   self.timeline_param["measurements_interval"]] * new_date)
            
            ts_index += new_date
            
        self.measurements = np.cumsum(self.measurements) # determines the absolute dates (in days)
        self.duration = self.measurements[-1] + 1
        # print(len(self.measurements), self.duration)
        return self.measurements
    
    def get_duration(self) -> int:
        if self.duration == 0:
            raise InvalidDuration("Sorry! You have to call Timeline.get_measurements method before getting duration")
        return self.duration

        
class TSPacket():
    packet_size: int 
    timeseriesLength: int 
    packet_dict = dict
    
    # packet_dict['timeline']: np.array
    # m: np.array # trend model
    # n: np.array # noise time series
    # season: np.array
    # w:  np.array
    # u_true:  np.array # u true
    # kd: np.array
    
    def __init__(self, packet_size: int, timeseriesLength: int, param: dict) -> None:
        self.packet_dict = {} 
        self.packet_size = packet_size
        self.timeseriesLength = timeseriesLength
        self.param = param
        self.packet_dict['timeline'] = np.empty((packet_size, timeseriesLength))
        self.packet_dict['m'] = np.empty((packet_size, timeseriesLength))
        self.packet_dict['n'] = np.empty((packet_size, timeseriesLength))
        self.packet_dict['season'] = np.empty((packet_size, timeseriesLength))
        self.packet_dict['u'] = np.empty((packet_size, timeseriesLength))
        self.packet_dict['w'] = np.empty((packet_size, timeseriesLength))
        self.packet_dict['kd'] = np.empty((packet_size, timeseriesLength))
    
    def get_packet_size(self) -> int:
        return self.packet_size
    
    def get_timeseriesLength(self) -> int:
        return self.timeseriesLength

    def get_param(self) -> dict:
        return self.param
    
    def set_data(self, data_type: str, data, packet_index: int) ->  None:
        self.packet_dict[data_type][packet_index] = data
    
    # def set_timeline(self, timeline: np.array, packet_index: int)-> None:
    #     self.timeline[packet_index] = timeline
    
    # def set_m(self, m: np.array, packet_index: int)-> None:
    #     self.m[packet_index] = m
        
    # def set_n(self, n: np.array, packet_index: int)-> None:
    #     self.n[packet_index] = n
    
    # def set_season(self, season: np.array, packet_index: int)-> None:
    #     self.season[packet_index] = season
    
    # def set_u(self, u: np.array, packet_index: int)-> None:
    #     self.u[packet_index] = u
        
    # def set_w(self, w: np.array, packet_index: int)-> None:
    #     self.w[packet_index] = w
        
    # def set_kd(self, kd: np.array, packet_index: int)-> None:
    #     self.kd[packet_index] = kd
        
    def calculate_kd_maxmin(self):
        
        return {'kd_min': np.abs(self.packet_dict['kd']).min(), 'kd_max': np.abs(self.packet_dict['kd']).max()}
        


class PacketBuilder():
    packet: TSPacket
    timeline_param: dict 
    model_param: dict 
    noise_param: dict 
    season_param: dict 

    def __init__(self, packet: TSPacket)-> None:
        self.packet = packet
        self.packet_size = packet.get_packet_size()
        self.timeseriesLength = packet.get_timeseriesLength()
        param = packet.get_param()
        
        self.timeline_param = param["timeline"]
        self.model_param = param["model"]
        self.noise_param = param["noise"]
        self.season_param = param["seasonality"]
        
        # ----- Configure the packet builder  -----
        # Instantiate three different context for strategies
        self.model = stsModels.mContext()
        self.noise = stsModels.mContext()
        self.season = stsModels.mContext()

        # Set model strategy
        chosen_model = self.model_param["type"]        
        if chosen_model == "sig":
            self.model.setStrategy(stsModels.SigmoidModel)
        elif chosen_model == "exp":
            self.model.setStrategy(stsModels.ExponentialModel)
        elif chosen_model == "poly":
            self.model.setStrategy(stsModels.PolynomialModel)
        # elif chosen_model == "spline":
        #     self.model = spsModels.SplineModel()
        
        # Set noise strategy
        chosen_noise = self.noise_param["type"]        
        if chosen_noise == "gaussian":
            self.noise.setStrategy(stsModels.GaussianNoise)

        # Set seasonality strategy
        chosen_season = self.season_param["type"]        
        if chosen_season == "mixed":
            self.season.setStrategy(stsModels.MixedSeasonality)       
        
    
    def build(self) -> None:
        """generate packet from time series"""
        for k in range(self.packet_size):
            #""" generate time series (m, n, season, u, w, kd)"""
            timelineObj = Timeline(self.timeseriesLength, self.timeline_param)
            timeline = timelineObj.get_measurements()
            duration = timelineObj.get_duration()
            
            m_data = self.model.get_timeseries_data(duration, self.model_param, 1)
            m_ts = m_data["timeseries"][timeline] # subsampling according to measurement dates
            
            n_data = self.noise.get_timeseries_data(duration, self.noise_param)
            n_ts = n_data["timeseries"][timeline] # subsampling according to measurement dates
            
            season_data = self.season.get_timeseries_data(duration, self.season_param)
            season_ts = season_data["timeseries"][timeline] # subsampling according to measurement dates
    
            # TIME SERIES COMPOSITION
            u_ts = m_ts + n_ts + season_ts # already subsampled
            u_ts -= u_ts[0]
            m_ts -= n_ts[0]
            
            # TIME SERIES WRAPPING
            w_ts = np.angle(np.exp(1j * u_ts * 2 * np.pi))/(2 * np.pi)
            kd_ts = (w_ts - u_ts).astype(int)
            
            
            if k % 100 == 0: print(k, "/", self.packet_size) # trace time series generation
            
            # STORE TIME SERIES in PACKET
            self.packet.set_data('timeline', timeline, k)
            self.packet.set_data('m', m_ts, k)
            self.packet.set_data('n', n_ts, k)
            self.packet.set_data('season', season_ts, k)
            self.packet.set_data('u', u_ts, k)
            self.packet.set_data('w', w_ts, k)
            self.packet.set_data('kd', kd_ts, k)



            # self.packet.set_timeline(timeline, k)
            # self.packet.set_m(m_ts, k)
            # self.packet.set_n(n_ts, k)
            # self.packet.set_season(season_ts, k)
            # self.packet.set_u(u_ts, k)
            # self.packet.set_w(w_ts, k)
            # self.packet.set_kd(kd_ts, k)



# class PSPacket():
#     packet_size: int 
#     timeseriesLength: int 
     
#     timeline: np.array
#     m: np.array # trend model
#     n: np.array # noise time series
#     season: np.array
#     w:  np.array
#     u_true:  np.array # u true
#     kd: np.array
    
#     def __init__(self, packet_size: int, timeseriesLength: int, param: dict) -> None:
#         self.packet_size = packet_size
#         self.timeseriesLength = timeseriesLength
#         self.param = param
#         self.timeline = np.empty((packet_size, timeseriesLength))
#         self.m = np.empty((packet_size, timeseriesLength))
#         self.n = np.empty((packet_size, timeseriesLength))
#         self.season = np.empty((packet_size, timeseriesLength))
#         self.u = np.empty((packet_size, timeseriesLength))
#         self.w = np.empty((packet_size, timeseriesLength))
#         self.kd = np.empty((packet_size, timeseriesLength))
    
#     def get_packet_size(self) -> int:
#         return self.packet_size
    
#     def get_timeseriesLength(self) -> int:
#         return self.timeseriesLength

#     def get_param(self) -> dict:
#         return self.param
    
#     def set_timeline(self, timeline: np.array, packet_index: int)-> None:
#         self.timeline[packet_index] = timeline
    
#     def set_m(self, m: np.array, packet_index: int)-> None:
#         self.m[packet_index] = m
        
#     def set_n(self, n: np.array, packet_index: int)-> None:
#         self.n[packet_index] = n
    
#     def set_season(self, season: np.array, packet_index: int)-> None:
#         self.season[packet_index] = season
    
#     def set_u(self, u: np.array, packet_index: int)-> None:
#         self.u[packet_index] = u
        
#     def set_w(self, w: np.array, packet_index: int)-> None:
#         self.w[packet_index] = w
        
#     def set_kd(self, kd: np.array, packet_index: int)-> None:
#         self.kd[packet_index] = kd
        
#     def calculate_kd_maxmin(self):
        
#         return {'kd_min': np.abs(self.kd).min(), 'kd_max': np.abs(self.kd).max()}
        


# class PacketBuilder():
#     packet: PSPacket
#     timeline_param: dict 
#     model_param: dict 
#     noise_param: dict 
#     season_param: dict 

#     def __init__(self, packet: PSPacket)-> None:
#         self.packet = packet
#         self.packet_size = packet.get_packet_size()
#         self.timeseriesLength = packet.get_timeseriesLength()
#         param = packet.get_param()
        
#         self.timeline_param = param["timeline"]
#         self.model_param = param["model"]
#         self.noise_param = param["noise"]
#         self.season_param = param["seasonality"]
        
#         # ----- Configure the packet builder  -----
#         # Instantiate three different context for strategies
#         self.model = spsModels.mContext()
#         self.noise = spsModels.mContext()
#         self.season = spsModels.mContext()

#         # Set model strategy
#         chosen_model = self.model_param["type"]        
#         if chosen_model == "sig":
#             self.model.setStrategy(spsModels.SigmoidModel)
#         elif chosen_model == "exp":
#             self.model.setStrategy(spsModels.ExponentialModel)
#         elif chosen_model == "poly":
#             self.model.setStrategy(spsModels.PolynomialModel)
#         # elif chosen_model == "spline":
#         #     self.model = spsModels.SplineModel()
        
#         # Set noise strategy
#         chosen_noise = self.noise_param["type"]        
#         if chosen_noise == "gaussian":
#             self.noise.setStrategy(spsModels.GaussianNoise)

#         # Set seasonality strategy
#         chosen_season = self.season_param["type"]        
#         if chosen_season == "mixed":
#             self.season.setStrategy(spsModels.MixedSeasonality)       
        
    
#     def build(self) -> None:
#         """generate packet from time series"""
#         for k in range(self.packet_size):
#             #""" generate time series (m, n, season, u, w, kd)"""
#             timelineObj = Timeline(self.timeseriesLength, self.timeline_param)
#             timeline = timelineObj.get_measurements()
#             duration = timelineObj.get_duration()
            
#             m_data = self.model.get_timeseries_data(duration, self.model_param, 1)
#             m_ts = m_data["timeseries"][timeline] # subsampling according to measurement dates
            
#             n_data = self.noise.get_timeseries_data(duration, self.noise_param)
#             n_ts = n_data["timeseries"][timeline] # subsampling according to measurement dates
#             n_mean_square = np.mean(np.array(n_ts)**2)
            
#             season_data = self.season.get_timeseries_data(duration, self.season_param)
#             season_ts = season_data["timeseries"][timeline] # subsampling according to measurement dates
    
#             # TIME SERIES COMPOSITION
#             u_ts = m_ts + n_ts + season_ts # already subsampled
#             u_ts -= u_ts[0]
#             m_ts -= n_ts[0]
            
#             # TIME SERIES WRAPPING
#             w_ts = np.angle(np.exp(1j * u_ts * 2 * np.pi))/(2 * np.pi)
#             kd_ts = (w_ts - u_ts).astype(int)
            
            
#             if k % 100 == 0: print(k, "/", self.packet_size) # trace time series generation
            
#             # STORE TIME SERIES in PACKET
#             self.packet.set_timeline(timeline, k)
#             self.packet.set_m(m_ts, k)
#             self.packet.set_n(n_ts, k)
#             self.packet.set_season(season_ts, k)
#             self.packet.set_u(u_ts, k)
#             self.packet.set_w(w_ts, k)
#             self.packet.set_kd(kd_ts, k)
