#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:41:47 2024

@author: xap
"""

from abc import ABC, abstractmethod

import numpy as np
import streamlit as st
import plotly.graph_objects as go   

  
    

def create_fig_unwrapped_ts(ts_id: int):
    """ create figure for wrapped, unwrapped, model time series"""

    timeline = st.session_state['ts'].get_data('timeline')[ts_id]
    ts_model = st.session_state['ts'].get_data('m')[ts_id]
    ts_season = st.session_state['ts'].get_data('season')[ts_id]    
    ts_unwrapped = st.session_state['ts'].get_data('u')[ts_id]
    ts_wrapped = st.session_state['ts'].get_data('w')[ts_id]
    ts_unwrappedseason = ts_unwrapped - ts_season
    
    fig1 = go.FigureWidget()
    fig1.update_layout(autosize=True, height=400,margin=dict(l=10,r=10,b=30,t=30,pad=4))
    fig1.update_layout(yaxis_title = " ", xaxis_showgrid = True,
                        xaxis_tickfont_size = 18, 
                        yaxis_tickfont_size = 18,
                        legend_font_size = 18, 
                        legend_font_color = 'black', plot_bgcolor='#ffffff')
    fig1.update_xaxes(gridcolor='#cccccc')
    fig1.update_yaxes(gridcolor='#cccccc')    
    
    fig1.add_scatter(y = ts_model, x = timeline, mode="markers", legendrank=4, visible='legendonly')
    diagramma11 = fig1.data[0]
    diagramma11.name = 'model'           #legenda
    diagramma11.marker.color = 'blue'
    diagramma11.marker.opacity = 0.7  
    # diagramma11.visible = st.session_state["checkboxes"]["chshow_model"]
    
    
    # diagramma12
    # unwrapped model + noise without seasonality
    fig1.add_scatter(y = ts_unwrappedseason, x = timeline, mode="markers", legendrank=2, visible='legendonly')
    diagramma12 = fig1.data[1] 
    diagramma12.name = "without season" #legenda
    diagramma12.marker.color = 'violet'
    diagramma12.marker.opacity = 0.7
    # diagramma12.visible = st.session_state["checkboxes"]["chshow_unwr_noise"]
    
    # diagramma13
    # wrapped time series
    fig1.add_scatter(y = ts_wrapped, x = timeline, mode="markers", legendrank=3, visible='legendonly')
    diagramma13 = fig1.data[2] 
    diagramma13.name = "wrapped" #legenda
    diagramma13.marker.color = 'red'
    diagramma13.marker.opacity = 0.7
    # diagramma13.visible = st.session_state["checkboxes"]["chshow_wrapped"]
    
    # diagramma14
    # unwrapped model + noise + seasonality
    fig1.add_scatter(y = ts_unwrapped, x = timeline, mode="markers", legendrank=1)
    diagramma14 = fig1.data[3] 
    diagramma14.name = "unwrapped" #legenda
    diagramma14.marker.color = 'green'
    diagramma14.marker.opacity = 0.8
    # diagramma14.visible = st.session_state["checkboxes"]["chshow_unwrseason"] 
    
   
    # diagramma12 - 2pi
    fig1.add_scatter(y = ts_wrapped - 1, x = timeline, mode="markers", visible='legendonly')       
    diagramma12_pi = fig1.data[4]
    diagramma12_pi.name = "-pi"
    diagramma12_pi.marker.color = 'gray'
    diagramma12_pi.marker.opacity = 0.2
    # diagramma12_pi.visible = st.session_state["checkboxes"]["chshow_2pireplicas"]
    # diagramma12 + 2pi
    fig1.add_scatter(y = ts_wrapped + 1, x = timeline, mode="markers", visible='legendonly')         
    diagramma12pi = fig1.data[5] 
    diagramma12pi.name = "+pi"
    diagramma12pi.marker.color = 'gray'
    diagramma12pi.marker.opacity = 0.2
    # diagramma12pi.visible = st.session_state["checkboxes"]["chshow_2pireplicas"]
        
    return fig1

def create_fig_ts(timeseries: np.array, ts_id: int, name : str, markercolor = 'black'):
    """ create figure for kd, noise and seasonality  time series"""
    fig1 = go.FigureWidget()
    fig1.update_layout(autosize=True, height=200,margin=dict(l=10,r=10,b=30,t=30,pad=4))
    fig1.update_layout(yaxis_title = " ", xaxis_showgrid = True,
                        xaxis_tickfont_size = 18, 
                        yaxis_tickfont_size = 18,
                        legend_font_size = 18, 
                        legend_font_color = 'black', plot_bgcolor='#ffffff')
    fig1.update_xaxes(gridcolor='#cccccc')
    fig1.update_yaxes(gridcolor='#cccccc')
    
    fig1.add_scatter(y = timeseries[ts_id], x = st.session_state['ts'].get_data('timeline')[ts_id], mode="markers")
    diagramma21 = fig1.data[0]
    diagramma21.name = 'name'  #legenda
    diagramma21.marker.color = markercolor
    diagramma21.marker.opacity = 0.7  
    return fig1

