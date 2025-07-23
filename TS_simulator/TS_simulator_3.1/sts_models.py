#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ps_collection class
generate, opern save and export ps collections time series
Created on Tue Feb 14 11:48:47 2023

@author: xap
"""

# Version
# ---------------------------- history ----------------------------
# B01 Context class and Interfaces definition; correction of strategies

import numpy as np


class PSModels():
    """ strategy interface"""
        
    def get_timeseries_data(self, ps_length, param1: dict, param2) -> {np.array, list}:
        timeseries = np.array 
        metadata = [+1]
        return (timeseries, metadata)

# *************************************************************************************        
class SigmoidModel(PSModels):
    """ implementation of strategy interface PSModels"""
        
    # to create sigmoid functions
    def sigmoid(self, x, x0, k, h, L, asymmetry):
        x1 = x0 - L/2
        x2 = x0 + L/2
        x3 = x0
        return h / (1 + np.exp(-k*(x - x0))) * np.where((x - x0) >= -L*asymmetry, 1, np.exp((x - x0) * (k - 1) / (L * (1 - asymmetry)))) + \
               h / (1 + np.exp(-k*(x - x1))) * np.where((x - x1) >= -L/2, 1, np.exp((x - x1) * (k - 1) / (L / 2))) + \
               h / (1 + np.exp(-k*(x - x2))) * np.where((x - x2) >= -L/2, 1, np.exp((x - x2) * (k - 1) / (L / 2))) + \
               h / (1 + np.exp(-k*(x - x3))) * np.where((x - x3) >= -L*asymmetry, 1, np.exp((x - x3) * (k - 1) / (L * (1 - asymmetry))))
    
    # Smooth steps joined through monotone curves 
    def joined_sigmoids(self, x, length, asymmetry):
        sigmoid_number = np.random.randint(self.m_param["ts_sigmoid_number_min"], self.m_param["ts_sigmoid_number_max"])  # steps random number(between 3 and 6)
        posizioni_simmetria = np.sort(np.random.uniform(0, length, sigmoid_number))  # Steps random position
        altezze = np.random.exponential(scale=self.m_param["ts_sigmoid_heights"], size=sigmoid_number)  # Steps random heights (distribuzione esponenziale inversa)
        gradini = []
        for i in range(sigmoid_number):
            gradino = self.sigmoid(x, posizioni_simmetria[i], k=15, h=altezze[i], L=np.random.uniform(self.m_param["ts_sigmoid_length_min"], self.m_param["ts_sigmoid_length_max"]), asymmetry=asymmetry)
            gradini.append(gradino)
        return np.sum(gradini, axis=0), posizioni_simmetria, altezze

    # Definizione della funzione per la parte finale con pendenza variabile verso l'alto
    def final_curve(self, x, pendenza_iniziale, pendenza_finale):
        return np.linspace(pendenza_iniziale, pendenza_finale, len(x))
    
    def limita_massimo(self, curva, massimo):
        massimo_attuale = np.max(curva)
        if massimo_attuale > massimo:
            curva = curva * (massimo / massimo_attuale)
        return curva
    
    def get_timeseries_data(self, ps_length: int, m_param: dict, param2 = None)  -> {np.array, list}:
        self.m_param = m_param
        # ------------------ MODEL Time series Sections Generation ---------------
        # Generazione della curva
        model_length = ps_length/self.m_param["ts_length_ratio"]  # Lunghezza casuale della curva (tra 15 e 20)
        # model_length = np.random.uniform(15, 20)  # Lunghezza casuale della curva (tra 15 e 20)
        asymmetry = np.random.uniform(0.1, 0.5)  # Random fraction of the first part of sigmoid diagram (tra 0 e 0.5)
        x = np.linspace(0, model_length, ps_length)
        ts_sigmoid, posizioni_simmetria, altezze_gradini = self.joined_sigmoids(x, model_length, asymmetry)
        
        # Definizione della pendenza iniziale e finale della parte finale della curva
        # pendenza_iniziale = np.random.uniform(0, 15)  # Pendenza casuale iniziale
        pendenza_finale = np.random.uniform(0, 15)  # Pendenza casuale finale
        ts_final = self.final_curve(x, 0, pendenza_finale)
    
        # Aggiunta della parte finale alla curva
        model_temp = ts_sigmoid + ts_final
        
        # Definizione del massimo desiderato per la curva
        ps_slope = np.random.uniform(self.m_param["time_series_slope_min"], self.m_param["time_series_slope_max_sig"])
        ts_global_max = ps_length*ps_slope  # Modifica questo valore come desideri
    
        # Applica il limite di massimo a tutte le curve
        model_temp = self.limita_massimo(model_temp, ts_global_max)
        model_temp -= model_temp[0]
        ts_sign = np.random.choice([-1, 1])
        model_temp = model_temp * ts_sign
        
        # return value: dict(timeseries, list(metadata))        
        return {"timeseries": np.array(model_temp), "ts_metadata": [ts_sign]}


# ***************************************************************************************        
class PolynomialModel(PSModels):
        
    def time_conv(self, time_int):
        """ Convert number of measurements in days; 
        input: number of measurements """
        time_conv_factor = float(1/365)
        return (time_conv_factor * time_int)
        
    def get_timeseries_data(self, ps_length: int, m_param: dict, param2 = None) -> {np.array, list}:  
        self.m_param = m_param
        
        # Initialize PS generation parameters       
        ts_index = 0    # global time series index 
        ts_level = 0    # initial level index    
                               
        model_temp = []              
 
        # # random time series length
        # if self.m_param["time_series_points_min"] != self.m_param["time_series_points_max"]:
        #     ps_length = np.random.randint(self.m_param["time_series_points_min"], self.m_param["time_series_length_max"])
        # else:
        #     ps_length = self.m_param["time_series_points_min"]
        
        
        # MODEL Time series Sections Generation
        # Generate the trend value for the first section
        s_trend = np.exp(np.random.uniform(-np.log(self.m_param["ts_sections_trend_max"]),
                                            np.log(self.m_param["ts_sections_trend_max"])))             
        while(ts_index < ps_length):
            # SECTIONS LENGTH Random Generation
            ####### Choose probabilistic distribution for section length
            # UNIFORM DISTRIBUTION
            #s_length = np.random.randint(param.ts_sections_length_min, ps_length)
            
            # GEOMETRIC DISTRIBUTION
            s_length = min(np.random.geometric(p=self.m_param["ts_geometric_series_param"])
                                + self.m_param["ts_sections_length_min"], ps_length - ts_index)
            # s_length_list.append(s_length)
            
            # SECTIONS SLOPE Random Generation
            slope_correction = max(1, np.random.normal(loc=s_length*0.1,
                                                            scale=self.m_param["ts_sections_slope_var"], size=None))
            s_slope = np.random.uniform(self.m_param["ts_sections_slope_min"],
                                             self.m_param["ts_sections_slope_max"])/slope_correction
            
            # SECTIONS POINT Generation
            s_points = [(((s_slope*self.time_conv(s_length)) 
                               *pow((x/s_length),s_trend)) + ts_level)
                             for x in range(1, s_length+1)] 
            ts_level = s_points[-1]
    
            # Complete Time series created assembling adjacent sections points
            model_temp = model_temp + s_points
            
            # NEXT SECTIONS TREND Random Generation
            s_trend = np.exp(np.random.normal(loc=-np.log(s_trend), 
                                                   scale=self.m_param["ts_sections_trend_var"], size=None))
    
            ts_index += s_length

        model_temp[0] = 0
        ts_sign = np.random.choice([-1, 1])
        # print(ts_sign)
        model_temp = (np.array(model_temp) * ts_sign).tolist()
        
        # return value: dict(timeseries, list(metadata))        
        return {"timeseries": np.array(model_temp), "ts_metadata": [ts_sign]}


# ***************************************************************************************        
class ExponentialModel(PSModels):

    def generate_single_exponential_curve(self, x_offset, growth_rate, x):
        """Genera una singola curva esponenziale con un dato tasso di crescita
        e un offset lungo x.        
        Args: growth_rate (float): Tasso di crescita dell'esponenziale.
        """
        # x_start, x_end = x_range
        # x = np.linspace(x_start, x_end, 1000)  # Genera un array di punti x
        y = np.exp(growth_rate * (x - x_offset))  # Calcola i valori y della curva esponenziale
        return y
    
    def limita_massimo(self, curva, massimo):
        massimo_attuale = np.max(curva)
        if massimo_attuale > massimo:
            curva = curva * (massimo / massimo_attuale)
        return curva 

    
    def get_timeseries_data(self, ps_length, m_param, param2 = None) -> {np.array, list}:
        # ------------------ MODEL Time series Sections Generation ---------------
        # Generazione della curva
        model_length = ps_length/m_param["ts_length_ratio"]  # Lunghezza casuale della curva (tra 15 e 20)
        # model_length = np.random.uniform(15, 20)  # Lunghezza casuale della curva (tra 15 e 20)
        
        x = np.linspace(0, model_length, ps_length)

        # Parametri configurabili
        max_heigth = np.random.uniform(1, m_param["max_heigth"])
        max_slope = np.random.uniform(0.2, m_param["max_slope"])
        growth_rate = max_slope/(1+np.log(max_heigth))
        x_offset = model_length - np.log(max_heigth)/growth_rate

        # Generazione e visualizzazione della curva
        model_temp = self.generate_single_exponential_curve(x_offset, growth_rate, x)
        
        # Definizione del massimo desiderato per la curva
        ps_slope = np.random.uniform(m_param["time_series_slope_min"], m_param["time_series_slope_max_exp"])
        ts_global_max = ps_length*ps_slope  # Modifica questo valore come desideri

        # Applica il limite di massimo a tutte le curve
        model_temp = self.limita_massimo(model_temp, ts_global_max)
        model_temp -= model_temp[0]
        ts_sign = np.random.choice([-1, 1])
        #print(ts_sign)
        model_temp = model_temp * ts_sign
        
        # return value: dict(timeseries, list(metadata))        
        return {"timeseries": np.array(model_temp), "ts_metadata": [ts_sign]}

# ***************************************************************************************        
class SplineModel(PSModels):


    # to create sigmoid functions
    def sigmoid(self, x, x0, k, h, L, asymmetry):
        x1 = x0 - L/2
        x2 = x0 + L/2
        x3 = x0
        return h / (1 + np.exp(-k*(x - x0))) * np.where((x - x0) >= -L*asymmetry, 1, np.exp((x - x0) * (k - 1) / (L * (1 - asymmetry)))) + \
               h / (1 + np.exp(-k*(x - x1))) * np.where((x - x1) >= -L/2, 1, np.exp((x - x1) * (k - 1) / (L / 2))) + \
               h / (1 + np.exp(-k*(x - x2))) * np.where((x - x2) >= -L/2, 1, np.exp((x - x2) * (k - 1) / (L / 2))) + \
               h / (1 + np.exp(-k*(x - x3))) * np.where((x - x3) >= -L*asymmetry, 1, np.exp((x - x3) * (k - 1) / (L * (1 - asymmetry))))
    
    # Smooth steps joined through monotone curves 
    def joined_sigmoids(self, x, length, asymmetry):
        sigmoid_number = np.random.randint(self.m_param["ts_sigmoid_number_min"], self.m_param["ts_sigmoid_number_max"])  # steps random number(between 3 and 6)
        posizioni_simmetria = np.sort(np.random.uniform(0, length, sigmoid_number))  # Steps random position
        altezze = np.random.exponential(scale=self.m_param["ts_sigmoid_heights"], size=sigmoid_number)  # Steps random heights (distribuzione esponenziale inversa)
        gradini = []
        for i in range(sigmoid_number):
            gradino = self.sigmoid(x, posizioni_simmetria[i], k=15, h=altezze[i], L=np.random.uniform(self.m_param["ts_sigmoid_length_min"], self.m_param["ts_sigmoid_length_max"]), asymmetry=asymmetry)
            gradini.append(gradino)
        return np.sum(gradini, axis=0), posizioni_simmetria, altezze

    # Definizione della funzione per la parte finale con pendenza variabile verso l'alto
    def final_curve(self, x, pendenza_iniziale, pendenza_finale):
        return np.linspace(pendenza_iniziale, pendenza_finale, len(x))
    
    def limita_massimo(self, curva, massimo):
        massimo_attuale = np.max(curva)
        if massimo_attuale > massimo:
            curva = curva * (massimo / massimo_attuale)
        return curva


# ******************************* NOISE ***************************************
    
class GaussianNoise(PSModels):
    
    def phase2dis(self, phi):
        return phi
        
    def get_timeseries_data(self, ps_length:int, n_param: dict, param2 = None) -> {np.array, list}:
        noise_temp = [] 
        self.n_param = n_param
        # global time series index
        ts_index = 0
        section_progressive = 0            
        while(ts_index < ps_length):
            # NOISE SECTIONS LENGTH Random Generation                
            
            # GEOMETRIC DISTRIBUTION
            n_length = min( np.random.geometric(p=self.n_param["n_geometric_series_param"])
                                + self.n_param["n_sections_length_min"], ps_length - ts_index)
 
            # NOISE SECTIONS Density Random Generation
            # to calculate n_level: extract a number that is the percentage of pi
            # multiplicate the number times 100*pi
            # convert in distance and divide by 3 (to obtain standard deviation)
            
            n_sections_level_max = self.n_param["n_sections_level_max"]

            n_level = self.phase2dis(np.pi * 0.01 * 
                                     np.random.uniform(self.n_param["n_sections_level_min"],
                                                            n_sections_level_max)) / 10                
            # NOISE POINT Generation
            n_points = list(np.random.normal(0, n_level, n_length))

            # Complete Noise Time series created assembling adjacent sections
            noise_temp = noise_temp + n_points                
            ts_index += n_length
            
        # return value: dict(timeseries, list(metadata))        
        return {"timeseries": np.array(noise_temp), "ts_metadata": []}


class AdaptiveGaussianNoise(PSModels):
    def get_timeseries_data(self, ps_length:int, n_param: dict, model_temp) -> {np.array, list}:
        noise_temp = np.array 

        # global time series index
        ts_index = 0
        section_progressive = 0            
        while(ts_index < ps_length):
            
            # GEOMETRIC DISTRIBUTION
            n_length = min( np.random.geometric(p=self.n_param["n_geometric_series_param"])
                                + self.n_param["n_sections_length_min"], ps_length - ts_index)
 
            # NOISE SECTIONS Density Random Generation
            # to calculate n_level: extract a number that is the percentage of pi
            # multiplicate the number times 100*pi
            # convert in distance and divide by 3 (to obtain standard deviation)
            
            if self.n_param["adaptive_noise"] == True:
                m_loc = model_temp[ts_index:ts_index+n_length]
                n_sections_level_max = self.n_max(m_loc)
                # print(n_sections_level_max)
            else:
                n_sections_level_max = self.n_param["n_sections_level_max"]

            n_level = self.phase2dis(np.pi * 0.01 * 
                                          np.random.uniform(self.n_param["n_sections_level_min"],
                                                            n_sections_level_max)) / 10                
            # NOISE POINT Generation
            n_points = list(np.random.normal(0, n_level, n_length))

            # Complete Noise Time series created assembling adjacent sections
            noise_temp = noise_temp + n_points                
            ts_index += n_length
            
        # return value: dict(timeseries, list(metadata))        
        return {"timeseries": np.array(noise_temp), "ts_metadata": []}      



# ******************************* SEASONALITY *********************************

class MixedSeasonality(PSModels):
   
    def get_timeseries_data(self, ps_length:int, season_param: dict, model_temp) -> {np.array, list}:
        days_in_year = 365
        measuretophase = 2*np.pi / days_in_year
        season_temp = np.array 
        
                
        # ------------------------- SEASONALITY time series Generation --------------------
        # Generation of seasonality amplitude
        s_amp = np.random.uniform(0.05, season_param["ts_seasonality_amplitude_max"])#*self.gap2pi_ # sum of 5 harmonics
        # Generation of seasonality phase shift
        season_shift = np.random.randint(0, days_in_year)
        # Generation of time series harmonics coefficients:    
        hsin = np.random.uniform(0, 1, 6)   # 5 random values
        hcos = np.random.uniform(0, 1, 6)   # 5 random values
            
        # Random choice of seasonality model
        if np.random.choice([True, False], p = [season_param["season_mod_prob"], 1 - season_param["season_mod_prob"]]):                                
            # Generation of periodic function with 5 harmonics
            season_temp = [s_amp * (hsin[1] * np.sin(measuretophase * x) + hcos[1] * np.cos(measuretophase * x)
                          # + hsin[2] * np.sin(2 * self.measuretophase * x) + hcos[2] * np.cos(2 * self.measuretophase * x)
                          # + hsin[3] * np.sin(3 * self.measuretophase * x) + hcos[3] * np.cos(3 * self.measuretophase * x)
                          # + hsin[4] * np.sin(4 * self.measuretophase * x) + hcos[4] * np.cos(4 * self.measuretophase * x)
                          # + hsin[5] * np.sin(5 * self.measuretophase * x) + hcos[5] * np.cos(5 * self.measuretophase * x)
                          )
                          for x in range(0 + season_shift, ps_length + season_shift)]
        else:
            duty_cycle = 0.1
            x = np.linspace(season_shift, ps_length + season_shift, ps_length)  # Intervallo di tempo
            # season_temp = list( s_amp *((x % (1/self.time_conv_factor)) < (duty_cycle / self.time_conv_factor)).astype(float))
            # Creazione dell'onda di impulsi triangolare asimmetrica
            season_temp = np.zeros_like(x)
            period = 1 / self.time_conv_factor
            high_time = duty_cycle * period
            is_inverted = np.random.choice([1, -1])
                
            for i in range(len(x)):
                t = x[i] % period
                if t < high_time:
                    season_temp[i] = (is_inverted * 2 * (t / high_time) - is_inverted * 1) * s_amp  # Impulso triangolare
                else:
                    season_temp[i] = 0  # Spazio tra gli impulsi           
        
        # return value: dict(timeseries, list(metadata))        
        return {"timeseries": np.array(season_temp), "ts_metadata": []}  

            



class mContext():
    psModelType: PSModels
    
    def __init__(self) -> None:
        self.psModelType = ExponentialModel()
        
    def setStrategy(self, psModelType):
        self.psModelType = psModelType()

    def get_timeseries_data(self, ps_length: int, param1: dict, param2 = None) -> (np.array, int):
        return self.psModelType.get_timeseries_data(ps_length, param1, param2)    
