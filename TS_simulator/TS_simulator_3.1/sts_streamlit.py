#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:44:13 2024

@author: xap
"""

import streamlit as st
from pathlib import Path

from src.sts_collection import synthTSCollection

import src.streamlit.pages.page1 as pg1
import src.streamlit.pages.page2 as pg2


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title= "TS Simulator", page_icon=":satellite:")
# Setting padding at the top
st.markdown("""<style> div.block-container{padding-top:2rem;}
            </style>""", unsafe_allow_html=True)

# Initialization
# Control flag to state the presence of data to be shown
# ts_number = 0
if "initial_data_dir" not in st.session_state:
    base_dir = Path(__file__).resolve().parent.parent.parent
    st.session_state['initial_data_dir']  = str(base_dir / "PS_DATA" / "Simulated") + '/sTS_Collections/'    
    # st.session_state['initial_data_dir'] = '/mnt/DATI_PC/AA1_PROGETTI/PS_DATA/Simulated/sTS_Collections/' 

if "data_off" not in st.session_state:
    st.session_state['data_off'] = True
if "ts" not in st.session_state:
    st.session_state['ts'] = synthTSCollection() 

# Creation of synthetic packet dataframe

            
# LAYOUT SETTINGS

tab1, tab2 = st.tabs(["Timeseries", "Parameters"])

with tab1:
    page1 = pg1.Page1()

with tab2:
    page2 = pg2.Page2()

