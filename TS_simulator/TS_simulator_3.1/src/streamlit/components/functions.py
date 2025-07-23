#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:43:17 2024

@author: xap
"""
import streamlit as st
from datetime import datetime
import pandas as pd

import os
import shutil

import tkinter as tk
from tkinter import filedialog as fd

# def select_folder() -> str:
#    root = tk.Tk()
#    root.withdraw()
#    folder_path = filedialog.askdirectory(master=root ) 
#    root.destroy()
#    return folder_path



def generate_ts() -> None:
    starttime = datetime.now()
    try:       
        filetypes = (
            ('Excel file', '*.xlsx'),
            ('Pandas dataframe file', '*.pkt'),
            ('All files', '*.*')
        )
    
        st.session_state['parameters_filename'] = fd.askopenfilename(
            title='Open a parameter file',
            initialdir=st.session_state['initial_data_dir'],
            filetypes=filetypes)
    except:
        st.write("####")
        st.warning("Select a parameter's file (.xlsx)!", icon="⚠️")    
        return 0
        
    
    st.session_state['ts'].generate(st.session_state['parameters_filename'])
    st.session_state['data_off'] = False
    endtime = datetime.now()
    print("total_time = ", endtime - starttime)
        
    return 0

def open_ts() -> None: 
    try:       
        filetypes = (
            ('synthetic time series', '*.stsc'),
            ('All files', '*.*')
        )
    
        st.session_state.file_name = fd.askopenfilename(
            title='Open a file',
            initialdir=st.session_state['initial_data_dir'],
            filetypes=filetypes)
         
        st.session_state['ts'].open(st.session_state.file_name)
        st.session_state['data_off'] = False
    except:
        st.write("####")
        st.warning("Select a stsc file to open!", icon="⚠️")
 
    return 0

def save_ts(save_folder: str) -> None:
    save_file_name = save_folder
    save_folder = st.session_state['initial_data_dir'] +  '/' + save_folder + '/'
    source_param_file = st.session_state['parameters_filename']
    dest_param_file = save_folder + save_file_name + '.xlsx'
    try:
        os.mkdir(save_folder)
        shutil.copy(source_param_file, dest_param_file)
        st.session_state['ts'].save(save_folder + save_file_name)
    except OSError as DirectoryError:
        st.write("####")
        st.warning(DirectoryError, icon="⚠️")
        print(DirectoryError) 
    return 0


def export_ts() -> None:
    print("export TS collection")
    st.session_state['ts'].export_ts()
    return 0


def create_packet_dataframe(data):
    rows = []
    
    for packet in data:
        # Create a row dictionary to hold flat data for this packet
        row = {}
        
        row['size'] = packet['packet'].get('size', None)  
        row['model type'] = packet['model'].get('type', None)
        row['kd_min'] = packet['kd'].get('kd_min', None)
        row['kd_max'] = packet['kd'].get('kd_max', None)
        row['noise_min'] = packet['noise'].get('n_sections_level_min', None)
        row['noise_max'] = packet['noise'].get('n_sections_level_max', None)
        row['seasonality_max'] = packet['seasonality'].get('ts_seasonality_amplitude_max', None)
        

        # row['model_type'] = packet['model'].get('type', None)  # example item from 'model'
        # row['noise_level'] = packet['noise'].get('level', None)  # example item from 'noise'
        # row['season_period'] = packet['season'].get('period', None)  # example item from 'season'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
        
    return df