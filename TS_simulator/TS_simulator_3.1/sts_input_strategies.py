#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:07:43 2024

@author: xap
"""
from abc import ABC, abstractmethod

import pandas as pd
   

class DataReader(ABC):
    @abstractmethod
    def read_data(self):
        pass    
    
class XlsxReader(DataReader):
    
    def read_data_from_multiple_sheet(self, file_name: str):
        # Read all sheets into a dictionary of DataFrames
        all_sheets = pd.read_excel(file_name, sheet_name=None, header=None, usecols=[0, 1, 2])

        # Initialize the list to store dictionaries
        packet_list = []
        
        collection_dict = {}
        if 'summary' in all_sheets:
            collection_df = all_sheets['summary']
            if collection_df.shape[1] >= 2:  # Ensure there are at least two columns to read
                # Assuming the collection name and timeserieslength are in the first and second rows, for example
                collection_dict['name'] = collection_df.iloc[0, 1]
                collection_dict['timeseriesLength'] = collection_df.iloc[1, 1]
        
        # Iterate over each sheet, skipping the "summary" sheet
        for sheet_name, df in all_sheets.items():
            # Skip the first sheet named "summary"
            if sheet_name.lower() == 'summary':
                continue
            
            # Ensure the DataFrame has exactly three columns
            if df.shape[1] != 3:
                print(f"Sheet '{sheet_name}' does not have exactly three columns in the specified range. Skipping.")
                continue
            
            # Rename columns for clarity
            df.columns = ['Outer Key', 'Inner Key', 'Value']
            
            # Drop rows where 'Outer Key' is NaN
            df = df.dropna(subset=['Outer Key'])
            
            # Initialize the dictionary for this sheet
            sheet_dict = {}
            
            # Iterate over each row to populate the dictionary
            for _, row in df.iterrows():
                outer_key = row['Outer Key']
                inner_key = row['Inner Key']
                value = row['Value']
                
                # Create the outer key if it doesn't exist
                if outer_key not in sheet_dict:
                    sheet_dict[outer_key] = {}
                
                # Assign the inner key and value
                sheet_dict[outer_key][inner_key] = value
            
            # Append the dictionary for this sheet to the list
            packet_list.append(sheet_dict)
        collection_dict['generation_params_list'] = packet_list
        # return {'collection': collection_dict, 'packets_list': packet_list}
        return collection_dict


    def read_data_from_single_sheet(self, file_name: str):
        # Read the single sheet in the Excel file
        df = pd.read_excel(file_name, header=None)

        # Filter rows to include only those where the first column is filled
        df = df[df[0].notna()]

        # Initialize output structures
        collection_dict = {}
        packet_list = []
        current_packet_number = None
        current_packet_dict = {}

        # Iterate over each row
        for _, row in df.iterrows():
            # Check if this row is related to 'collection'
            if row[0] == 'collection':
                # Save collection data in collection_dict
                collection_dict[row[3]] = row[4]  # Assumes the key is in the fourth column and value in the fifth

            # Check if this row is related to 'packet'
            elif row[0] == 'packet':
                packet_number = row[1]  # The packet number is in the second column
                inner_key = row[2]      # The key for the main packet dictionary is in the third column
                sub_key = row[3]        # The key in the sub-dictionary is in the fourth column
                value = row[4]          # The value in the sub-dictionary is in the fifth column

                # If the packet number changes, append the current packet dictionary to packet_list and reset
                if current_packet_number != packet_number:
                    if current_packet_dict:  # If there's an existing packet dictionary, save it
                        packet_list.append(current_packet_dict)
                    # Start a new packet dictionary for the new packet number
                    current_packet_dict = {}
                    current_packet_number = packet_number

                # If the inner key doesn't exist in the current packet dictionary, initialize it
                if inner_key not in current_packet_dict:
                    current_packet_dict[inner_key] = {}

                # Add the sub_key and value to the dictionary for the current inner key
                current_packet_dict[inner_key][sub_key] = value

        # Append the last packet dictionary if it exists
        if current_packet_dict:
            packet_list.append(current_packet_dict)

        collection_dict['generation_params_list'] = packet_list

        # Return the structured data
        # return {'collection': collection_dict, 'packets_list': packet_list}
        return collection_dict


    def read_data(self, file_name: str):
        try:
            excel_file = pd.ExcelFile(file_name)
        
            # Count the sheets
            num_sheets = len(excel_file.sheet_names)
            if num_sheets > 1:
                return self.read_data_from_multiple_sheet(file_name)
            else:
                return self.read_data_from_single_sheet(file_name)
           
        except FileNotFoundError:
            print(f"File '{file_name}' not found.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []    



class DictionaryReader(DataReader):
    def read_data(self, file_name):
        
        try:
            # Read all sheets into a dictionary of DataFrames
            all_sheets = pd.read_excel(file_name, sheet_name=None, header=None, usecols=[0, 1, 2])
            
            # Initialize the list to store dictionaries
            dict_list = []
            
            # Iterate over each sheet, skipping the "summary" sheet
            for sheet_name, df in all_sheets.items():
                # Skip the first sheet named "summary"
                if sheet_name.lower() == 'summary':
                    continue
                
                # Ensure the DataFrame has exactly three columns
                if df.shape[1] != 3:
                    print(f"Sheet '{sheet_name}' does not have exactly three columns in the specified range. Skipping.")
                    continue
                
                # Rename columns for clarity
                df.columns = ['Outer Key', 'Inner Key', 'Value']
                
                # Drop rows where 'Outer Key' is NaN
                df = df.dropna(subset=['Outer Key'])
                
                # Initialize the dictionary for this sheet
                sheet_dict = {}
                
                # Iterate over each row to populate the dictionary
                for _, row in df.iterrows():
                    outer_key = row['Outer Key']
                    inner_key = row['Inner Key']
                    value = row['Value']
                    
                    # Create the outer key if it doesn't exist
                    if outer_key not in sheet_dict:
                        sheet_dict[outer_key] = {}
                    
                    # Assign the inner key and value
                    sheet_dict[outer_key][inner_key] = value
                
                # Append the dictionary for this sheet to the list
                dict_list.append(sheet_dict)
            
            return dict_list
        
        except FileNotFoundError:
            print(f"File '{file_name}' not found.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

def get_data_reader(file_name):
    if file_name.endswith('.xlsx'):
        return XlsxReader()
    if file_name.endswith('.py'):
        return DictionaryReader()    
    else:
        raise ValueError(f"Unsupported file format: {file_name}")