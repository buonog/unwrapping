#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:57:59 2024

@author: xap
"""
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def seasonality_algo(unw, absolute_timeline, season_param):
    # Transpose and set index
    df = unw.T.set_index(absolute_timeline)
    
    # Step 1: Define the regular interval
    # regular_interval = 12  # in days (1 measurement every 12 days)
    regular_interval = season_param['max_interval']
    
    season_period = 365 // regular_interval
    
    # Step 1: Define the date range
    start_date = df.index.min()
    end_date = df.index.max()
    regular_dates = pd.date_range(start=start_date, end=end_date, freq=f"{regular_interval}D")
    
    # Step 2: Create a regular DataFrame and update it with existing data
    regular_df = pd.DataFrame(index=regular_dates, columns=df.columns)
    regular_df.update(df)
    
    # Step 3: Remove NaN rows efficiently
    regular_df = regular_df.dropna(how="any")  # Ensures no missing values remain
    
    # Step 4: Apply seasonal decomposition for each column
    seasonality_results = {}
    for column in regular_df.columns:
        decomposition = seasonal_decompose(regular_df[column], model="additive", period=season_period)
        seasonality_results[column] = decomposition.seasonal
    
    # Combine the seasonal components into a new DataFrame
    seasonality_df = pd.DataFrame(seasonality_results)
    
    # Function for moving average
    def moving_average(series, window):
        return series.rolling(window=window, center=True).mean()
    
    # Step 5: Smooth the seasonal time series
    smoothed_seasonality_dict = {}
    av_window = season_param['averaging_window']
    
    for column in seasonality_df.columns:
        original = seasonality_df[column]
        smoothed = moving_average(original, window=av_window)
        smoothed_seasonality_dict[column] = smoothed.fillna(original)  # Fill NaN with original values
    
    smoothed_seasonality = pd.concat(smoothed_seasonality_dict, axis=1)
    smoothed_seasonality.columns = seasonality_df.columns
    
    # Step 6: Identify missing dates (dates in `df` but not in `regular_df`)
    missing_dates = df.index.difference(regular_df.index)    
    # Step 7: Create a DataFrame for the missing dates
    missing_df = pd.DataFrame(index=missing_dates, columns=seasonality_df.columns)
    
    # Step 8: Combine `seasonality_df` (seasonal components for regular dates) with missing dates
    seasonality_with_missing = pd.concat([smoothed_seasonality, missing_df]).sort_index()
    
    # Step 9: Interpolate the seasonal component for the missing dates
    seasonality_with_missing = seasonality_with_missing.interpolate(method="time")
    
    # Transpose the dataframe
   
    return seasonality_with_missing.T