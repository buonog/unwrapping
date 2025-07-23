#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:43:17 2024

@author: xap
"""
import streamlit as st
import pandas as pd
import src.streamlit.components.functions  as functions


class Page2():
    def __init__(self)-> None:
        df = functions.create_packet_dataframe(st.session_state['ts'].collection_dict['generation_params_list'])
        # LAYING OUT THE TOP SECTION OF THE APP
        row1_1, row1_2 = st.columns((1, 5), gap = "large")

        with row1_1:
            # TITLE    
            st.write("### Packets synthesis")
            st.write("#")

        with row1_2:
            # st.write("#")
            st.dataframe(df)    
            # st.dataframe(df, width=800)    


  
        row2_1, row2_2= st.columns([1, 3], gap = "large")
        with row2_1:
            st.write("### Packets parameters")
        

        with row2_2:
            # packet_id = st.selectbox("##### Packets:", range(len(st.session_state['ps'].packets_param_list)))
            # st.dataframe(pd.DataFrame(st.session_state['ps'].packets_param_list[packet_id])) 
            packet_id = st.selectbox("##### Packets:", range(len(st.session_state['ts'].collection_dict['generation_params_list'])))

