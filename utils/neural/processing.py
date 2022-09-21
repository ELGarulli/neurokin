import numpy as np
from numpy.typing import ArrayLike
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from scipy import signal
from typing import List, Tuple
from matplotlib import pyplot as plt
from importing import time_to_sample

def get_stim_timestamps(sync_ch: np.ndarray, expected_pulses: int) -> np.ndarray:
    """
    Get indexes of only threshold crossing up from 0 to 1.
    Sometimes there are spurious signals on the stimulation channel, when the stimulator turns off.
    This function trims to the expected number of pulses.
    :param sync_ch: the stimulation sync channel data
    :param expected_pulses: number of expected pulses
    :return: trimmed indexes
    """
    threshold_crossing = np.diff(sync_ch > 0, prepend=False)
    idxs_edges = np.where(threshold_crossing)[0]
    stim_starts = idxs_edges[::2]
    stim_starts_trimmed = stim_starts[:expected_pulses]

    return stim_starts_trimmed


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
    if len(neudata.sync_data.shape)>1:
        raise ValueError("Warning: depending on the experiment setting the sync data might be a multichannel array \n"
                         "please use the .pick_sync_data(idx) method to pick the correct sync channel.")
    total_pulses = n_amp_tested * pulses
    sync = get_stim_timestamps(neudata.sync_data, total_pulses)
    onset = sync[0:len(sync):pulses]
    stim_len = time_to_sample(time_stim, neudata.fs, is_t2=True)
    timestamps_blocks = [(onset[i], onset[i] + stim_len) for i in range(len(onset))]
    return timestamps_blocks


def get_median_distance(a: ArrayLike) -> float:
    """
    Gets median distance between points
    :param a: array-like data
    :return: median
    """
    distances = []
    for i in range(len(a) - 1):
        distances.append(a[i] - a[i + 1])
    return abs(np.median(distances))


def running_mean(x: ArrayLike, n: int) -> ArrayLike:
    """
    Returns the running mean with window n
    :param x: data
    :param n: window size
    :return: smoothed data
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def trim_equal_len(raw: List[ArrayLike]) -> List[float]:
    """
    Trims a list of arrays to all have the same length.
    :param raw: raw data
    :return: list with trimmed data
    """
    lens = [len(r) for r in raw]
    equaled = [r[:min(lens)] for r in raw]
    return equaled


def parse_raw(raw: np.ndarray, stimulation_idxs: np.ndarray, samples_before_stim: int,
              skip_one: bool = False) -> np.ndarray:
    """
    Parses the signal given the timestamp of the stimulation.
    :param raw: raw data of one channel
    :param stimulation_idxs: indexes of the stimulation onset
    :param samples_before_stim: how much before the stimulation onset to parse
    :param skip_one: if True parses every second stimulation
    :return: parsed raw signal into an array of equally sized chunks
    """
    stimulation_idxs = stimulation_idxs + samples_before_stim
    if skip_one:
        stimulation_idxs = stimulation_idxs[::2]
    # skip first chunk that precedes the first stimulation to avoid cropping errors
    split_raw = np.split(raw, stimulation_idxs)[1:]
    leveled_list = trim_equal_len(split_raw)
    trimmed_array = np.vstack(leveled_list)
    return trimmed_array


def get_average_amplitudes(parsed_raw, tested_amplitudes, pulses_number):
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
        averaged_amp = average_subset(parsed_raw, 0, len(parsed_raw))
        return [averaged_amp]
    for i in range(number_tested_amplitudes):
        start = i * pulses_number
        stop = (i + 1) * pulses_number
        averaged_amp = average_subset(parsed_raw, start, stop)
        averaged_amplitudes.append(averaged_amp)
    return averaged_amplitudes


def average_subset(array: ArrayLike, start: int, stop: int) -> np.ndarray:
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


def get_fooofed_psd(freqs: ArrayLike, psd: ArrayLike, frange: List[int] = None) -> (ArrayLike, ArrayLike):
    """
    Computes the difference from the power spectrum and the aperiodic ie the periodic component
    :param freqs: frequencies corresponding to the y axis
    :param psd: power points corresponding to the x axis
    :param frange: range of frequencies where to compute the PSD
    :return: new set of frequencies, periodic component
    """
    fm = FOOOF()
    fm.fit(freqs=freqs, power_spectrum=psd, freq_range=frange)
    aperiodic_component = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    periodic_component = fm.power_spectrum - aperiodic_component
    return fm.freqs, periodic_component


def get_aperiodic(freqs: ArrayLike, psd: ArrayLike, frange: List[int] = None) -> (ArrayLike, ArrayLike):
    """
    Computes aperiodic component of the psd
    :param freqs:
    :param psd:
    :param frange:
    :return:
    """
    fm = FOOOF()
    fm.fit(freqs=freqs, power_spectrum=psd, freq_range=frange)
    init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    return fm.freqs, init_ap_fit


def find_closest_index(data: ArrayLike, datapoint: float) -> int:
    """
    Given an array of data and a datapoint it returns the index of the element that has
    the minimum difference to the datapoint
    :param data: data array-like
    :param datapoint: datapoint to find a close value to
    :return: index of the closest element
    """
    min_difference = np.inf
    idx = 0
    for i in range(len(data)):
        if abs(data[i] - datapoint) < min_difference:
            min_difference = abs(data[i] - datapoint)
            idx = i
    return idx


def find_closest_smaller_index(data: ArrayLike, datapoint: float) -> int:
    """
    Given an array of data and a datapoint it returns the index of the element that is the closest but lower than
    the datapoint
    :param data: data array-like
    :param datapoint: datapoint to find a close value to
    :return: index of the closest element, lower than the datapoint
    """
    min_difference = np.inf
    idx = 0
    for i in range(len(data)):
        if abs(data[i] - datapoint) < min_difference:
            if data[i] < datapoint:
                min_difference = abs(data[i] - datapoint)
                idx = i
    return idx


def get_fast_foofed_specgram(raw: ArrayLike, fs: float, nperseg: int,
                             noverlap: int, frange: List[int] = None) -> (ArrayLike, ArrayLike, ArrayLike):
    """
    Returns a matrix corresponding to a periodgram where from each column the aperiodic component has been subtracted.
    Because of computational intensity only the overall aperiodic component is computed.
    :param raw: raw signal
    :param fs: sampling frequency
    :param nperseg: number of points per segment
    :param noverlap: number of points to overlap
    :param frange: frequency range for the in which to compute the aperiodic component
    :return: timepoints array, frequencies array, normalized matrix
    """
    _t, _f, sxx = signal.spectrogram(raw, fs=fs, nperseg=nperseg, noverlap=noverlap)
    freqs, psd = signal.welch(raw, fs=fs, nperseg=nperseg, noverlap=noverlap)

    f, aperiodic = get_aperiodic(freqs=freqs, psd=psd, frange=frange)
    aperiodic = np.flipud(aperiodic[:, None])

    lower = find_closest_smaller_index(freqs, frange[0])
    upper = find_closest_smaller_index(freqs, frange[1])
    psd_matrix = np.log10(sxx[lower:upper])
    foofed = psd_matrix - aperiodic

    nt = len(foofed[-1]) * (nperseg - noverlap)
    t = np.linspace(0, nt, num=len(foofed[-1]))
    return t, f, foofed


def get_spectrogram_data(fs: float, raw: ArrayLike, nfft: int = None,
                         noverlap: int = None, **kwargs) -> Tuple[ArrayLike]:
    """
    Gets the data used to plot a spectrogram
    :param fs: sampling frequency
    :param raw: raw data
    :param nfft: nfft to compute the fft
    :param noverlap: number of overlap points
    :return:
    """
    pxx, freq, t, _ = plt.specgram(raw, NFFT=nfft, Fs=fs, noverlap=noverlap, **kwargs)

    return pxx, freq, t


def calculate_power_spectral_density(data: ArrayLike, fs: int, **kwargs) -> tuple:
    """
    Calculate the frequencies and power spectral densities from the raw recording time series data.

    :param data: raw recording data
    :param fs: sampling frequency
    :return: frequencies, power spectral density
    """
    freq, pxx = signal.welch(data, fs, **kwargs)
    return freq, pxx
