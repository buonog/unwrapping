#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:43:17 2024

@author: xap
"""
import streamlit as st

import os
import src.streamlit.components.functions  as functions
import src.streamlit.components.figures as figures



class Page1():
    def __init__(self)-> None:
        
        # LAYING OUT THE TOP SECTION OF THE APP
        row1_1, row1_2, row1_3, row1_4, row1_4b, row1_5, row1_6 = st.columns((3, 2, 1.5, 1, 3, 2, 1), gap = "large")

        with row1_1:
            # TITLE    
            st.write("## TS Simulator")
            st.write("#")

        with row1_2:
            # st.write("#")
            ts_id = st.selectbox("##### TS Number:", list(range(0, st.session_state['ts'].collection_size)))
            
        with row1_3:
            # st.write("#")
            st.button("##### New", key="new_collection", help=None, on_click=functions.generate_ts, args=(), kwargs=None)

        with row1_4:
            # st.write("#")
            st.button("##### Open", key="load_collection",
                help=None, on_click=functions.open_ts, args=None, kwargs=None)


        with row1_5:
            save_folder = st.text_input("New Collection name", "sTS_collection_")
            st.button("##### Save", key="save_collection", help=None, on_click=functions.save_ts, args=(save_folder,),
                          kwargs=None, disabled = st.session_state.data_off)
    
        with row1_6:
            # st.write("#")
            st.button("##### Export", key="export_collection", help=None, 
                          on_click=functions.export_ts, args=None, kwargs=None, disabled = st.session_state.data_off)

  
        row2_1, row2_2= st.columns([1, 5], gap = "large")
        with row2_1:
            st.write("### Unwrapped Time Series")        
            if st.session_state.data_off == False:      
                with row2_2:
                    fig_unwrapped_ts = figures.create_fig_unwrapped_ts(ts_id)
                    st.plotly_chart(fig_unwrapped_ts, use_container_width=True)
    
        row3_1, row3_2, row3_3= st.columns([1, 5, 0.6], gap = "large")
        with row3_1:
            st.write("### Noise") 
        with row3_2:
            if st.session_state.data_off == False: 
                # fig_noise_ts = figures.create_fig_ts(st.session_state['ts'].get_n(), ts_id, "Noise", "black")
                fig_noise_ts = figures.create_fig_ts(st.session_state['ts'].get_data('n'), ts_id, "Noise", "black")

                
                st.plotly_chart(fig_noise_ts, use_container_width=True)
 
        row4_1, row4_2, row4_3= st.columns([1, 5, 0.6], gap = "large")
        with row4_1:
            st.write("### Phase Jumps") 
            
        with row4_2:
            if st.session_state.data_off == False: 
                # fig_ph_ts = figures.create_fig_ts(st.session_state['ts'].get_kd(), ts_id, "kd", 'red')
                fig_ph_ts = figures.create_fig_ts(st.session_state['ts'].get_data('kd'), ts_id, "kd", 'red')

                st.plotly_chart(fig_ph_ts, use_container_width=True) 

        row5_1, row5_2, row5_3= st.columns([1, 5, 0.6], gap = "large")
        with row5_1:
            st.write("### Seasonality") 
        if st.session_state.data_off == False:       
            with row5_2:
                # fig_season_ts = figures.create_fig_ts(st.session_state['ts'].get_season(), ts_id, "seasonality", 'green')
                fig_season_ts = figures.create_fig_ts(st.session_state['ts'].get_data('season'), ts_id, "seasonality", 'green')

                st.plotly_chart(fig_season_ts, use_container_width=True)

