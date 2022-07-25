import tdt
import json
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Tuple


def import_tdt_channel_data(folderpath, ch=0, t1=0, t2=-1) -> (float, np.ndarray):
    """
    Wrapper for the import function of tdt, to be more user friendly.
    Warning: tdt function allows to specify for channels, however it's 1-based and if ch==0
    it returns all channels. Use the indexing carefully.
    :param folderpath: folderpath of the subject experiment
    :param t1: initial time to index in seconds
    :param t2: last time to index in seconds
    :return: frequency sample and raw sample
    """
    data = tdt.read_block(folderpath, evtype=['streams'], channel=ch)
    try:
        raw = data.streams.LFP1.data
        fs = data.streams.LFP1.fs
    except AttributeError:
        try:
            raw = data.streams.EOG1.data
            fs = data.streams.EOG1.fs
        except AttributeError:
            try:
                raw = data.streams.NPr1.data
                fs = data.streams.NPr1.fs
            except AttributeError:
                raise
    s1 = 0
    s2 = -1
    if t1 != 0:
        s1 = time_to_sample(timestamp=t1, fs=fs, is_t1=True)
    if t2 != -1:
        s2 = time_to_sample(timestamp=t2, fs=fs, is_t2=True)
    raw = raw[:, s1:s2]
    return fs, raw


def import_tdt_stimulation_data(folderpath, t1=0, t2=-1) -> np.ndarray:
    """
    Returns the stimulation channel, assuming it's stored in Wav1
    :param folderpath: folderpath of the subject experiment
    :param t1: initial time to index in seconds
    :param t2: last time to index in seconds
    :return: frequency sample and raw sample
    """
    data = tdt.read_block(folderpath, evtype=['streams'])
    try:
        stim_data = data.streams.Wav1.data
        fs = data.streams.LFP1.fs
    except AttributeError:
        raise Exception("No stimulation data found")
    if t1 != 0:
        s1 = time_to_sample(timestamp=t1, fs=fs, is_t1=True)
    if t2 != -1:
        s2 = time_to_sample(timestamp=t2, fs=fs, is_t2=True)
    stim_data = stim_data[s1:s2]

    return stim_data


def time_to_sample(timestamp: float, fs: float, is_t1: bool = False, is_t2: bool = False) -> int:
    """
    Function adapted from time2sample in TDTbin2py.py
    Returns the sample index given a time in seconds and the sampling frequency.
    It has to be specified if the timestamp refers to t1 or t2.
    :param timestamp: time in seconds
    :param fs: sampling frequency
    :param is_t1: specify if the timestamp is t1
    :param is_t2: specify if the timestamp is t2
    :return:
    """
    sample = timestamp * fs
    if is_t2:
        exact = np.round(sample * 1e9) / 1e9
        sample = np.floor(sample)
        if exact == sample:
            sample -= 1
    else:
        if is_t1:
            sample = np.ceil(sample)
        else:
            sample = np.round(sample)
    sample = np.uint64(sample)
    return sample


def unroll_array(array: ArrayLike, channel_number: int) -> ArrayLike:
    # TODO look for smarter way to do this, most probably scipy or numpy have a better function
    """
    Unrolls the array from a s1ch1, s1ch2, s1ch3 format to a 2D array with samples as columns and channels as rows.
    :param array: 1D array
    :param channel_number: number of channels
    :return: 2D array
    """
    s_number = int(len(array) / channel_number)
    a = np.zeros(shape=(channel_number, s_number))
    for i in range(channel_number):
        for j, z in zip(range(0, len(array), channel_number), range(s_number)):
            a[i][z] = array[i + j]
    return a


def import_open_ephys_channel_data(folderpath: str, experiment: str, recording: str) -> (float, np.ndarray):
    """
    Imports open ephys data from binary files. Sampling frequency is returned as a float and raw data are returned
    in arbitrary units
    :param folderpath:
    :return:
    """
    structure_path = folderpath + "/" + experiment + "/" + recording + "/structure.oebin"
    with open(structure_path) as f:
        structure = json.load(f)
    # TODO refactor to have path in consts
    source_processor_id = str(structure["continuous"][0]["source_processor_name"].replace(" ", "_")) + "-"+\
                          str(structure["continuous"][0]["source_processor_id"]) + ".0"

    binary_data_path = folderpath + "/" + experiment + "/" + recording + "/continuous/" + source_processor_id + '/continuous.dat'
    fs = structure["continuous"][0]["sample_rate"]
    n_ch = structure["continuous"][0]["num_channels"]
    neural_data_flat = np.fromfile(binary_data_path, dtype='<i2')
    neural_data_au = unroll_array(array=neural_data_flat, channel_number=n_ch)
    return fs, neural_data_au
