from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

import src.ts_input_templates as tsInput
from src.unwrapping_context import UnwrappingContext
from src.seasonality_algo import seasonality_algo
from src.snr_algo import snr_calc
from src.fractal_algo import FractalDimensionEstimator

class InvalidDuration(Exception):
    """Custom exception raised when duration checks fail."""
    pass


class Packet(ABC):
    """
    Abstract base class for time series packets.

    A Packet is a container that stores data (wrapped, unwrapped, metadata, etc.)
    derived from a TSCollection. It acts as a general interface for specialized
    processing such as unwrapping, SNR computation, seasonality analysis, etc.

    Attributes:
        collection (TSCollection): Reference to the main collection.
        collection_name (str): Name of the time series collection.
        collection_folder (str): Folder path where the collection is located.
        packet_dict (dict): Dictionary containing packet data and metadata.
    """

    def __init__(self, collection) -> None:
        self.collection = collection
        self.collection_name = collection.name
        self.collection_folder = collection.folder
        self.packet_dict = {}
        self.packet_dict['absolute_timeline'] = collection.collection_dict['absolute_timeline']

    def load(self, packet_file: str) -> dict:
        """
        Loads a packet from file into the packet dictionary.

        Args:
            packet_file (str): Name of the file (without prefix) to load.

        Returns:
            dict: The loaded packet dictionary.
        """
        try:
            file_name = self.collection_folder + self.collection_name + '_' + packet_file
            reader = tsInput.get_data_reader(file_name)
            self.packet_dict = reader.read_data(file_name)
            print(f"Packet {file_name} opened successfully!")
        except ValueError as InputError:
            print(InputError)
        return self.packet_dict

    def save(self):
        """
        Saves the current packet dictionary to a `.unw` file in pickle format.
        """
        try:
            file_name = self.collection_folder + self.collection_name + '_' + self.packet_dict['name'] + '.unw'
            pd.to_pickle(self.packet_dict, file_name)
            print(f"File {file_name} saved successfully!")
        except ValueError as OutputError:
            print(OutputError)

    def compare(self, reference_collection):
        """Stub for comparing packet data to a reference collection."""
        pass

    def get_data(self, data_type: str):
        """Returns data from the packet dictionary."""
        return self.packet_dict[data_type]

    def set_data(self, data_type: str, data) -> None:
        """Sets data in the packet dictionary."""
        self.packet_dict[data_type] = data

    @property
    def info(self) -> list:
        """Prints summary information about the packet contents."""
        print("\n -------------- PACKET INFO ----------------")
        print("Keys in the packet dictionary:")
        for key, value in self.packet_dict.items():
            if value is None:
                continue
            elif isinstance(value, (np.ndarray, pd.Series)):
                print(f"  {key}: {type(value).__name__} with shape {value.shape}")
            elif isinstance(value, pd.DataFrame):
                print(f"  {key}: DataFrame with shape {value.shape}")
            elif isinstance(value, str):
                print(f"  {key}: {value}")
            elif isinstance(value, (list, dict)) and len(value) == 0:
                continue
            else:
                print(f"  {key}: {value}")


class Unwrapping(Packet):
    """
    Packet class for handling phase unwrapping and comparison.
    """

    def new(self, unwrapping_name: str, unwrapping_algo: str, unwrap_param: dict, unwrapping_note=None) -> None:
        """
        Initializes a new unwrapping packet and performs unwrapping.

        Args:
            unwrapping_name (str): Name of the unwrapping run.
            unwrapping_algo (str): Name of the unwrapping algorithm.
            unwrap_param (dict): Parameters for the algorithm.
            unwrapping_note (str, optional): Additional notes or metadata.
        """
        self.packet_dict.update({
            'name': unwrapping_name,
            'note': unwrapping_note,
            'algo': unwrapping_algo,
            'w': pd.DataFrame(),
            'm': pd.DataFrame(),
            'u': pd.DataFrame(),
            'comparison': pd.DataFrame()
        })
        self.unwrap(unwrap_param)

    def unwrap(self, unwrap_param={}) -> None:
        """
        Executes the unwrapping algorithm using relative timeline from collection.
        """
        relative_timeline = self.collection.get_data('relative_timeline')

        unwrapping_context = UnwrappingContext()
        unwrapping_context.setStrategy(self.packet_dict['algo'])
        unwrapping_context.setRelativeTimeline(relative_timeline)

        self.packet_dict['w'] = self.collection.get_data('w')
        unwrapped_results = self.packet_dict['w'].apply(lambda ts: unwrapping_context.unwrap(ts.to_numpy(), unwrap_param), axis=1)

        self.set_data('absolute_timeline', self.collection.get_data('absolute_timeline'))
        self.set_data('relative_timeline', relative_timeline)
        self.set_data('m', pd.DataFrame(unwrapped_results.apply(lambda x: x["m"]).tolist(), index=self.packet_dict['w'].index))
        self.set_data('u', pd.DataFrame(unwrapped_results.apply(lambda x: x["u"]).tolist(), index=self.packet_dict['w'].index))

    def unwrap_relative(self, unwrap_param={}) -> None:
        """
        Executes unwrapping using a simple range-based timeline.
        """
        relative_timeline = range(len(self.collection.get_data('relative_timeline')))
        self.unwrap_param = unwrap_param
        self.unwrap(relative_timeline)

    def compare(self, kd_other, other_name) -> pd.Series:
        """
        Compares this unwrapping result with another kd series.

        Args:
            kd_other (DataFrame): The comparison kd series.
            other_name (str): Name of the other result (used for labeling).

        Returns:
            pd.Series: Accuracy percentage per time series.
        """
        kd_self = (self.packet_dict['w'] - self.packet_dict['u']).astype(int)

        if kd_self.shape != kd_other.shape:
            raise ValueError("Shapes of kd arrays do not match.")

        accuracy = ((kd_self == kd_other).sum(axis=1) / kd_self.shape[1]) * 100
        comparison_df = pd.DataFrame({other_name: accuracy}, index=kd_self.index)

        if self.packet_dict['comparison'].empty:
            self.packet_dict['comparison'] = comparison_df
        else:
            self.packet_dict['comparison'] = pd.concat([self.packet_dict['comparison'], comparison_df], axis=1)

        return accuracy

    def compare_by_offset(self, kd_other, tol=1e-6):
        """
        Compare kd arrays by offset difference with tolerance.

        Args:
            kd_other (ndarray): kd values to compare against.
            tol (float): Tolerance level for error counting.

        Returns:
            tuple: (diff array, accuracy by offset per series)
        """
        kd_self = (np.round(self.packet_dict['w'] - self.packet_dict['u'])).astype(int)

        if kd_self.shape != kd_other.shape:
            raise ValueError("The kd vectors of the two packets have different shapes.")

        diff = np.abs(kd_self - kd_other)
        candidate = np.asarray((np.round(np.mean(diff, axis=1))).astype(int))
        candidate_expanded = candidate[:, np.newaxis]
        diff_candidate = diff - candidate_expanded
        count_positive = np.sum(diff_candidate > tol, axis=1)

        accuracy_by_offset = (1 - count_positive / diff.shape[1]) * 100
        return diff, accuracy_by_offset


class Seasonality(Packet):
    """
    Packet for storing and computing seasonality metrics.
    """

    def new(self, name: str, note=None) -> None:
        self.packet_dict['name'] = name
        self.packet_dict['note'] = note
        self.packet_dict['season'] = pd.DataFrame()

    def seasonality_calc(self, unw, season_param):
        """
        Performs seasonality analysis on the unwrapped time series.

        Args:
            unw (DataFrame): Unwrapped data.
            season_param (dict): Parameters for the seasonality algorithm.
        """
        self.packet_dict['season'] = seasonality_algo(unw, self.packet_dict['absolute_timeline'], season_param)


class SNR(Packet):
    """
    Packet for storing and computing Signal-to-Noise Ratio (SNR).
    """

    def new(self, name: str, note=None) -> None:
        self.packet_dict['name'] = name
        self.packet_dict['note'] = note
        self.packet_dict['snr'] = pd.DataFrame()

    def snr_calc(self, unw, snr_param) -> None:
        """
        Computes noise variance and SNR from the unwrapped data.

        Args:
            unw (DataFrame): Unwrapped time series.
            snr_param (dict): Parameters for SNR computation.
        """
        noise_results = unw.apply(lambda ts: snr_calc(ts.to_numpy(), snr_param), axis=1)
        noise_df = noise_results.apply(pd.Series)
        noise_df.columns = ['noise_variance', 'snr']
        self.packet_dict['snr'] = noise_df


class FDE(Packet):
    """
    Packet for computing Fractal Dimension Estimation (FDE).
    """

    def new(self, data, name: str, note=None) -> None:
        """
        Initializes FDE estimator and prepares empty results.

        Args:
            data (DataFrame): Time series data for FDE.
            name (str): Packet name.
            note (str, optional): Additional info.
        """
        self.packet_dict['name'] = name
        self.packet_dict['note'] = note
        self.packet_dict['fde'] = pd.DataFrame()
        self.fde = FractalDimensionEstimator(data, use_gpu=False, n_jobs=1, verbose=True)

    def fde_calc(self, fde_param) -> None:
        """
        Computes fractal dimension using specified parameters.

        Args:
            fde_param (dict): Parameters including method, scales, plotting.
        """
        fde_results = self.fde.estimate(
            method=fde_param["method"],
            box_min_scale=fde_param["box_min_scale"],
            box_max_scale=fde_param["box_max_scale"],
            plot_loglog=fde_param["plot_loglog"]
        )
        self.packet_dict['fde'] = pd.DataFrame(fde_results)
