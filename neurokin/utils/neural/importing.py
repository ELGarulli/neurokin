import tdt
import json
import numpy as np
from numpy.typing import ArrayLike
from neurokin.constants.open_ephys_structure import STRUCTURE, CONTINUOUS, SOURCE_PROCESSOR_NAME, SOURCE_PROCESSOR_ID, \
    TRAILING_NUMBER, SAMPLE_RATE, CHANNEL_NUMBER


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
    sample = int(sample)
    return sample


def import_tdt_channel_data(folderpath, ch=0, t1=0, t2=-1, stream_name="Wav1", stim_name="Wav1",
                            sync_present=False) -> (
        float, ArrayLike):
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
    stim_data = None
    try:
        streams = data.streams
        stored = getattr(streams, stream_name)
        raw = stored.data
        fs = stored.fs

    except AttributeError:
        print("No stream named " + stream_name + ", please specify the correct stream_name")
        print("Please chose from: ")
        print(data.streams.__dict__.keys())
        return

    s1 = 0
    s2 = -1
    if t1 != 0:
        s1 = time_to_sample(timestamp=t1, fs=fs, is_t1=True)
    if t2 != -1:
        s2 = time_to_sample(timestamp=t2, fs=fs, is_t2=True)
    raw = raw[..., s1:s2]

    if sync_present:
        try:
            streams = data.streams
            stored = getattr(streams, stim_name)
            stim_data = stored.data
            fs = stored.fs

        except AttributeError:
            print("No stimulation data named " + stim_name + ", please specify the correct stream_name")
            print("Please chose from: ")
            print(data.streams.__dict__.keys())
            return
        stim_data = stim_data[s1:s2]
    return fs, raw, stim_data


def import_open_ephys_channel_data(folderpath: str, experiment: str, recording: str, channels=None,
                                   sync_present: bool = False,
                                   sync_ch: int = None) -> (
        float, np.ndarray):
    """
    Imports open ephys data from binary files.

    :param folderpath: Folderpath where the experiment is, including the Node
    :param experiment: experiment folder
    :param recording: recording folder
    :param channels: indicate which channels to return
    :return: Sampling frequency is returned as a float and raw data are returned in arbitrary units
    """
    structure_path = folderpath + "/" + experiment + "/" + recording + "/" + STRUCTURE + ".oebin"
    sync_data = None
    with open(structure_path) as f:
        structure = json.load(f)

    source_processor = str(structure[CONTINUOUS][0][SOURCE_PROCESSOR_NAME].replace(" ", "_")) + "-" + \
                       str(structure[CONTINUOUS][0][SOURCE_PROCESSOR_ID]) + TRAILING_NUMBER

    binary_data_path = folderpath + "/" + experiment + "/" + recording + "/" + \
                       CONTINUOUS + "/" + source_processor + "/" + CONTINUOUS + ".dat"

    fs = structure[CONTINUOUS][0][SAMPLE_RATE]
    n_ch = structure[CONTINUOUS][0][CHANNEL_NUMBER]

    neural_data_flat = np.fromfile(binary_data_path, dtype='<i2')
    n_samples = int(len(neural_data_flat) / n_ch)

    neural_data_au = np.reshape(a=neural_data_flat, newshape=(n_ch, n_samples), order='F')

    if sync_present:
        sync_data = neural_data_au[sync_ch]

    if channels:
        mask = np.zeros(n_ch, dtype=bool)
        mask[channels] = True
        neural_data_au = neural_data_au[mask, ...]
        if neural_data_au.shape[0] == 1:
            neural_data_au = neural_data_au[0]

    return fs, neural_data_au, sync_data


def import_binary_to_float32(filename, channel_number, sample_number):
    """
    Imports binary data stored in C major to a float32 array
    :param filename: file to import from
    :param sample_number: number of sample in each channel
    :param channel_number: number of channels
    :return the array with shape channel_number*sample_number
    """
    dt = np.dtype('f4')
    test = np.fromfile(filename, dtype=dt)
    float_array = np.reshape(a=test, newshape=(channel_number, sample_number), order='C')
    return float_array
