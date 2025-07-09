#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:07:43 2024

@author: xap
"""
from abc import ABC, abstractmethod
import pandas as pd
import os
import numpy as np
import h5py
import datetime

# =========================
# Abstract Reader
# =========================

class DataReader(ABC):
    """
    Abstract base class for time series file readers.

    This class defines a standard interface for reading different file formats
    containing wrapped and unwrapped time series displacements.

    The output dictionary from any subclass must include:
        - 'w': wrapped displacements (numpy array)
        - 'u': unwrapped displacements (numpy array)
        - 'kd': integer difference between wrapped and unwrapped
        - 'absolute_timeline': list of acquisition dates (pandas.Series)
        - 'relative_timeline': timeline relative to first date (numpy array of int)
        - 'name': file name without extension
        - 'folder': path to the file's directory
    """

    def read_data(self, file_name: str) -> dict:
        """
        Wrapper to safely read data using the appropriate format handler.

        Args:
            file_name (str): Full path to the data file.

        Returns:
            dict: A dictionary containing timeseries and metadata.
        """
        try:
            collection_dictionary = self.read_file(file_name)
            return collection_dictionary   
        except FileNotFoundError:
            print(f"File '{file_name}' not found.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    @abstractmethod
    def read_file(self, file_name: str) -> dict:
        """
        Abstract method to be implemented by subclasses for reading a file.

        Args:
            file_name (str): Full path to the file.

        Returns:
            dict: The structured data extracted from the file.
        """
        collection_dictionary = {}
        return collection_dictionary


# =========================
# Specific Readers
# =========================

class OldPklReader(DataReader):
    """
    Reader for old-style pickled DataFrames with a single row.

    The DataFrame is expected to have:
        - Column 0: wrapped displacements (array)
        - Column 1: unwrapped displacements (array)

    NOTE: Method currently not implemented.
    """

    def read_file(self, file_name: str):
        pass
        # Example (disabled):
        # collection_file = pd.read_pickle(file_name)
        # w = collection_file.iloc[0]["wrapped"]
        # u = collection_file.iloc[0]["unwrapped"]
        # return w, u


class sTscReader(DataReader):
    """
    Reader for `.stsc` pickle files containing structured dictionary data.
    """

    def read_file(self, file_name: str) -> dict:
        collection_dictionary = pd.read_pickle(file_name)
        collection_dictionary['kd'] = (np.round(collection_dictionary['w'] - collection_dictionary['u'])).astype(int)

        temp_timedelta = (collection_dictionary['absolute_timeline'] - collection_dictionary['absolute_timeline'].iloc[0]).dt.days
        collection_dictionary['relative_timeline'] = temp_timedelta.to_numpy()

        collection_name = os.path.splitext(os.path.basename(file_name))[0]
        collection_folder = os.path.dirname(file_name) + os.sep
        collection_dictionary['name'] = collection_name
        collection_dictionary['folder'] = collection_folder
        return collection_dictionary


class TscReader(DataReader):
    """
    Reader for `.tsc` pickle files structured as dictionaries.
    Similar to `sTscReader`, but for standard TSC format.
    """

    def read_file(self, file_name: str) -> dict:
        collection_dictionary = pd.read_pickle(file_name)
        collection_dictionary['kd'] = (np.round(collection_dictionary['w'] - collection_dictionary['u'])).astype(int)

        temp_timedelta = (collection_dictionary['absolute_timeline'] - collection_dictionary['absolute_timeline'].iloc[0]).dt.days
        collection_dictionary['relative_timeline'] = temp_timedelta.to_numpy()

        collection_name = os.path.splitext(os.path.basename(file_name))[0]
        collection_folder = os.path.dirname(file_name) + os.sep
        collection_dictionary['name'] = collection_name
        collection_dictionary['folder'] = collection_folder
        return collection_dictionary


class MatlabReader(DataReader):
    """
    Reader for MATLAB `.mat` files (v7.3+, HDF5 format) with keys:
        - 'unWrappedDisplacements_rad'
        - 'wrappedDisplacements_rad'
        - 'AcqDates_datenum'
    """

    def read_file(self, file_name: str) -> dict:
        collection_dictionary = {}

        with h5py.File(file_name, 'r') as f:
            u_data = (np.array(f['unWrappedDisplacements_rad'][()]).squeeze().astype(float) / (2 * np.pi)).T
            w_data = (np.array(f['wrappedDisplacements_rad'][()]).squeeze().astype(float) / (2 * np.pi)).T
            datenum_data = np.array(f['AcqDates_datenum'][()]).squeeze()

        absolute_timeline = pd.Series([self.matlab_datenum_to_datetime(d) for d in datenum_data])

        collection_dictionary['u'] = u_data
        collection_dictionary['w'] = w_data
        collection_dictionary['absolute_timeline'] = absolute_timeline
        collection_dictionary['kd'] = (np.round(w_data - u_data)).astype(int)
        collection_dictionary['relative_timeline'] = (absolute_timeline - absolute_timeline.iloc[0]).dt.days.to_numpy()

        collection_name = os.path.splitext(os.path.basename(file_name))[0]
        collection_folder = os.path.dirname(file_name) + os.sep
        collection_dictionary['name'] = collection_name
        collection_dictionary['folder'] = collection_folder

        return collection_dictionary

    def matlab_datenum_to_datetime(self, datenum):
        """
        Converts MATLAB datenum to Python datetime.

        Args:
            datenum (float): MATLAB serial date number.

        Returns:
            datetime.datetime: Corresponding Python datetime object.
        """
        dt = datetime.datetime.fromordinal(int(datenum)) \
             + datetime.timedelta(days=datenum % 1) \
             - datetime.timedelta(days=366)
        return dt


class UnwrappingReader(DataReader):
    """
    Reader for `.unw` pickle files used in unwrapping results.

    Expected to be a dictionary similar to TSC format, but may vary.
    """

    def read_file(self, file_name: str) -> dict:
        return pd.read_pickle(file_name)


# =========================
# Reader Factory
# =========================

def get_data_reader(file_name) -> DataReader:
    """
    Factory function to select the appropriate reader based on file extension.

    Args:
        file_name (str): Path to the file to read.

    Returns:
        DataReader: Instance of the appropriate subclass.

    Raises:
        ValueError: If file extension is not supported.
    """
    if file_name.endswith('.stsc'):
        return sTscReader()
    elif file_name.endswith('.tsc'):
        return TscReader()
    elif file_name.endswith('.mat'):
        return MatlabReader()
    elif file_name.endswith('.unw'):
        return UnwrappingReader()
    else:
        raise ValueError(f"\nUnsupported file format ERROR!!!!!!: \n{file_name}")


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    generaldata_folder = str(base_dir / "PS_DATA" / "Real") + '/'    

    collection_folder = "Brennero/"
    collection_file = "BrenneroNoReUnwr4DIF.mat"
    
    file_name = generaldata_folder + collection_folder + collection_file

    try:
        reader = get_data_reader(file_name)
    except ValueError as ve:
        print(ve)
        exit(1)

    data = reader.read_data(file_name)

    print("File read successfully!")
    print("Keys in the data dictionary:")
    for key, value in data.items():
        if isinstance(value, (np.ndarray, pd.Series)):
            print(f"  {key}: {type(value).__name__} with shape {value.shape}")
        else:
            print(f"  {key}: {value}")

    # Optional: saving example
    # output_folder = data.get('folder', '.')
    # name = data.get('name', 'output')
    # if 'w' in data:
    #     pd.DataFrame(data['w']).to_csv(os.path.join(output_folder, f"{name}_wrapped.csv"), index=False)
    # if 'u' in data:
    #     pd.DataFrame(data['u']).to_csv(os.path.join(output_folder, f"{name}_unwrapped.csv"), index=False)
