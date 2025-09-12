import warnings
import numpy as np
from numpy.typing import ArrayLike
from scipy import signal
from typing import List, Tuple, Union
from neurokin.utils.neural.importing import time_to_sample


def simply_mean_data_binarize(sync_ch: np.ndarray):
    """
    Return a simply binarized array by setting anything below the mean to 0 and everything above to 1.
    Use only in very clear-cut cases

    :param sync_ch: input sync channel
    :return: binarized array
    """
    if np.any(np.isnan(sync_ch)):
        raise ValueError("The input array contains nan, please cast them to a number using np.nan_to_num or equivalent")
    mean = np.mean(sync_ch)
    binarized = np.where(sync_ch < mean, 0, 1)
    return binarized


def get_stim_timestamps(sync_ch: np.ndarray, expected_pulses: int = None) -> np.ndarray:
    """
    Get indexes of only threshold crossing up from 0, i.e. edge detection.
    Sometimes there are spurious signals on the stimulation channel, when the stimulator turns off;
    if expected_pulses is given this function trims to the expected number of pulses.

    :param sync_ch: the stimulation sync channel data
    :param expected_pulses: number of expected pulses
    :return: trimmed indexes
    """

    def process_channel(sync_ch):
        if np.any(np.isnan(sync_ch)):
            raise ValueError(
                "The input array contains nan, please cast them to a number using np.nan_to_num or equivalent")
        threshold_crossing = np.diff(sync_ch > 0, prepend=False)
        idxs_edges = np.where(threshold_crossing)[0]
        stim_starts = idxs_edges[::2]
        if expected_pulses is not None:
            if not isinstance(expected_pulses, int) or expected_pulses <= 0:
                raise ValueError("The expected pulses must be a positive integer")
            if expected_pulses < len(stim_starts):
                warnings.warn(
                    "Warning: number of pulses is greater than the expected pulses, trimming to the expected pulses. "
                    "(expected behaviour by function design)")
            if expected_pulses > len(stim_starts):
                warnings.warn(
                    "Warning: number of pulses is less than the expected pulses, returning all the pulses found. "
                    "(expected behaviour by function design)")
            return stim_starts[:expected_pulses]
        else:
            return stim_starts

    if sync_ch.ndim == 1:
        return process_channel(sync_ch)
    elif sync_ch.ndim == 2:
        return np.array([process_channel(row) for row in sync_ch], dtype=object)
    else:
        raise ValueError("sync_ch must be a 1D or 2D array")


def get_timestamps_stim_blocks(neudata, n_amp_tested, pulses, time_stim):
    """
    Given a DBS recording with multiple stimulation amplitudes tested it gives the time stamps of the onset and end
    of each block of stimulation.

    :param neudata: NeuralData object, containing sync_data
    :param n_amp_tested:
    :param pulses:
    :param time_stim:
    :return:
    """
    if len(neudata.sync_data.shape) > 1:
        raise ValueError("Warning: depending on the experiment setting the sync data might be a multichannel array \n"
                         "please use the .pick_sync_data(idx) method to pick the correct sync channel.")
    total_pulses = n_amp_tested * pulses
    sync = get_stim_timestamps(neudata.sync_data, total_pulses)
    onset = sync[0:len(sync):pulses]
    stim_len = time_to_sample(time_stim, neudata.fs, is_t2=True)
    # FIXME check that neudata.fs is correct if stim and raw have different fs
    timestamps_blocks = [(onset[i], onset[i] + stim_len) for i in range(len(onset))]
    return timestamps_blocks


def get_median_distance(a: ArrayLike) -> float:
    """
    Gets median distance between points

    :param a: array-like data
    :return: median
    """
    if not isinstance(a, np.ndarray):
        raise TypeError("Input must be an array like type")
    distances = abs(np.diff(a))
    return np.median(distances)


def running_mean(x: ArrayLike, n: int) -> ArrayLike:
    """
    Returns the running mean with window n

    :param x: data
    :param n: window size
    :return: smoothed data
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array")
    if n <= 0:
        raise ValueError("Window size must be a positive integer")
    if n > x.shape[0]:
        raise ValueError("Window size must be smaller than the data length")
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def running_mean_2D(x: ArrayLike, n: int) -> ArrayLike:
    """
        Returns the running mean with window n, on 2D data

        :param x: data
        :param n: window size
        :return: smoothed data
        """
    smoothed_data = np.apply_along_axis(running_mean, 1, x, n)
    return smoothed_data


def trim_equal_len(raw: List[ArrayLike]) -> List[float]:
    """
    Trims a list of arrays to all have the same length.

    :param raw: raw data
    :return: list with trimmed data
    """
    if not all(np.asarray(arr).ndim == 1 for arr in raw):
        raise ValueError("Not all arrays are 1D")
    lens = [len(r) for r in raw]
    equaled = [r[:min(lens)] for r in raw]
    return equaled


def parse_raw(raw: np.ndarray, stimulation_idxs: np.ndarray, samples_before_stim: int,
              skip_one: bool = False, min_len_chunk: int = 1) -> np.ndarray:
    """
    Parses the signal given the timestamps of the stimulation.

    :param raw: raw data of one channel
    :param stimulation_idxs: indexes of the stimulation onset
    :param samples_before_stim: how much before the stimulation onset to parse
    :param skip_one: if True parses every second stimulation
    :param min_len_chunk: filters the chunks to have a minimal len between pulses
    :return: parsed raw signal into an array of equally sized chunks
    """
    stimulation_idxs = np.asarray(stimulation_idxs)
    if len(stimulation_idxs) == 0:
        raise ValueError("Stimulation indexes is empty, please check the input data")
    stimulation_idxs = stimulation_idxs - samples_before_stim
    if stimulation_idxs[0] < 0:
        stimulation_idxs[0] = 0

    if skip_one:
        stimulation_idxs = stimulation_idxs[::2]
    split_raw = np.split(raw, stimulation_idxs)[
                1:]  # skip chunk that precedes the first stimulation to avoid cropping errors
    split_raw = [chunk for chunk in split_raw if len(chunk) >= min_len_chunk]
    leveled_list = trim_equal_len(split_raw)
    trimmed_array = np.vstack(leveled_list)
    return trimmed_array


# TESTME
def get_average_amplitudes(parsed_raw, tested_amplitudes, pulses_number=None):
    """
    Assumes an experiment where a sequence of stimulation amplitudes are applied,
    every stimulation with a fixed number of pulses

    :param parsed_raw:
    :param tested_amplitudes:
    :param pulses_number:
    :return:
    """
    number_tested_amplitudes = len(tested_amplitudes)
    averaged_amplitudes = []
    if number_tested_amplitudes == 1:
        averaged_amp = average_block(parsed_raw, 0, len(parsed_raw))
        return [averaged_amp]
    if not pulses_number:
        pulses_number = int(len(parsed_raw) / len(tested_amplitudes))
    for i in range(number_tested_amplitudes):
        start = i * pulses_number
        stop = (i + 1) * pulses_number
        averaged_amp = average_block(parsed_raw, start, stop)
        averaged_amplitudes.append(averaged_amp)
    return averaged_amplitudes


# TESTME
def average_block(array: ArrayLike, start: int, stop: int) -> np.ndarray:
    """
    Averages a subset of elements

    :param array: input array
    :param start: start idx for parsing
    :param stop: stop idx for parsing
    :return: average of the subset
    """
    parsed = array[start:stop]
    averaged = np.mean(parsed, axis=0)
    return averaged


def find_closest_index(data: ArrayLike, datapoint: float) -> int:
    """
    Given an array of data and a datapoint it returns the index of the element that has
    the minimum difference to the datapoint

    :param data: data array-like
    :param datapoint: datapoint to find a close value to
    :return: index of the closest element
    """
    data = np.asarray(data)
    if np.any(np.isnan(data)):
        raise ValueError('The input array contains nan which will always return as the nearest to datapoint'
                         'please cast it to a number using np.nan_to_num or equivalent')
    diff = np.abs(data - datapoint)

    if data.ndim == 1:
        return diff.argmin()
    elif data.ndim == 2:
        flat_idx = diff.argmin()
        return np.unravel_index(flat_idx, data.shape)
    else:
        raise ValueError("Input must be a 1D or 2D array")


# TESTME

def find_closest_smaller_index(data: ArrayLike, datapoint: float) -> Union[int, Tuple[int, int]]:
    """
    Given an array of data and a datapoint, returns the index of the element that is
    the closest but lower than the datapoint.

    For a 1D array, returns an integer index.
    For a 2D array, returns a tuple (row, column) indicating the location of the closest element.

    :param data: data array-like (1D or 2D)
    :param datapoint: datapoint to find a close value to
    :return: index of the closest element below datapoint (int for 1D or tuple for 2D)
    """
    data = np.asarray(data)
    if np.any(np.isnan(data)):
        raise ValueError("The input array contains nan; please cast them to a number using np.nan_to_num or equivalent")

    mask = data < datapoint
    differences = np.where(mask, datapoint - data, np.inf)

    if np.all(np.isinf(differences)):
        raise ValueError("No element in the array is smaller than the datapoint")

    flat_idx = differences.argmin()

    if data.ndim == 1:
        return flat_idx
    elif data.ndim == 2:
        return np.unravel_index(flat_idx, data.shape)
    else:
        raise ValueError("Input must be a 1D or 2D array")


# TESTME with TDT data
def get_spectrogram_data(fs: float, raw: ArrayLike, nfft: int = None,
                         noverlap: int = None) -> Tuple[ArrayLike]:
    """
    Gets the data used to plot a spectrogram

    :param fs: sampling frequency
    :param raw: raw data
    :param nfft: nfft to compute the fft
    :param noverlap: number of overlap points
    :return:
    """
    _t, _f, sxx = signal.spectrogram(raw, fs=fs, nperseg=nfft, noverlap=noverlap)

    return sxx, _f, _t


# TESTME with TDT data
def calculate_power_spectral_density(data: ArrayLike, fs: float, **kwargs) -> tuple:
    """
    Calculate the frequencies and power spectral densities from the raw recording time series data.

    :param data: raw recording data
    :param fs: sampling frequency
    :return: frequencies, power spectral density
    """
    freq, pxx = signal.welch(data, fs, **kwargs)
    return freq, pxx


