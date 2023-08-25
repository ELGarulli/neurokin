import csv

import numpy as np
import pandas as pd
from scipy import stats

from neurokin.utils.neural import (processing, importing)
from neurokin.neural_data import NeuralData

from typing import List, Tuple


def get_csv_first_block_len(csv_path: str) -> int:
    """
    Reads in a csv and returns the len of the first block. This is until a row is empty.
    :param csv_path: path of the file
    :return: len of block
    """
    with open(csv_path, newline='') as f:
        count = 0
        rows = csv.reader(f)
        for line in rows:
            if not line:
                break
            count += 1
    return count


def get_first_block_df(csv_path: str, skiprows: int = 0) -> pd.DataFrame:
    """
    Gets the first block from a csv path. That is until the first empty line.
    :param csv_path: path of the file
    :param skiprows: how many rows to skip at the beginning
    :return: pandas dataframe of the first block
    """
    block_end = get_csv_first_block_len(csv_path) - skiprows
    df = pd.read_csv(csv_path, skiprows=skiprows, nrows=block_end)

    return df


def get_first_last_frame_from_csv(csv_path: str) -> Tuple[int]:
    """
    Given a csv from Vicon, it will return the first and last frame of the region of interest.
    This is usually encoded as the first and last row of the second block in the file
    :param csv_path: path of the file
    :return:
    """
    with open(csv_path, newline='') as f:
        count = 0
        rows = csv.reader(f)
        for line in rows:
            if not line:
                break
            count += 1
    first_block_end = count + 5

    with open(csv_path, newline='') as f:
        count = 0
        rows = csv.reader(f)
        rowcount = 0
        for line in rows:
            if rowcount <= first_block_end:
                rowcount += 1
                continue
            elif not line:
                break
            count += 1

    df = pd.read_csv(csv_path, skiprows=first_block_end, nrows=count)
    first_frame = int(df.iloc[0][0])
    last_frame = int(df.iloc[-1][0])
    return first_frame, last_frame


def get_freeze_ts_bound(df: pd.DataFrame) -> List[float]:
    """
    Returns onset and end of a freezing event in timestamps
    :param df: events dataframe
    :return: onset and end
    """
    event_onset_idxs = df.index[(df["Name"] == "Foot Off") & (df["Context"] == "General")].tolist()
    if not len(event_onset_idxs) > 0:
        return [], []

    event_onset = [df.iloc[i]["Time (s)"] for i in event_onset_idxs]
    event_end = [df.iloc[i + 1]["Time (s)"] for i in event_onset_idxs if not i + 1 == len(df)]

    return event_onset, event_end


def get_event_timestamps_freezing_active(df: pd.DataFrame,
                                         timestamp_gait: List[List[float]]) -> List[List[float]]:
    """
    Returns freezing events after gait happened
    :param df: events dataframe
    :param timestamp_gait: list of gait timestamps
    :return: list of timestamps of freezing after gait events
    """

    event_onset, event_end = get_freeze_ts_bound(df)
    if not len(event_onset) > 0:
        return []

    events_onset_selected = []
    events_end_selected = []

    for i in range(len(event_onset)):
        if event_onset[i] >= timestamp_gait[0][0]:
            events_onset_selected.append(event_onset[i])
            events_end_selected.append(event_end[i])

    events = list(map(list, zip(events_onset_selected, events_end_selected)))

    return events


def get_event_timestamps_freezing_rest(df: pd.DataFrame,
                                       timestamp_gait: List[List[float]]) -> List[List[float]]:
    """
    Returns freezing events before gait happened
    :param df: events dataframe
    :param timestamp_gait: list of gait timestamps
    :return: list of timestamps of freezing before gait events
    """
    event_onset, event_end = get_freeze_ts_bound(df)
    if not len(event_onset) > 0:
        return []

    events_onset_selected = []
    events_end_selected = []
    for i in range(len(event_onset)):
        if len(timestamp_gait) > 0:
            if event_onset[i] > timestamp_gait[0][0]:
                break
        events_onset_selected.append(event_onset[i])
        events_end_selected.append(event_end[i])

    events = list(map(list, zip(events_onset_selected, events_end_selected)))
    return events


def get_event_timestamps_gait(df: pd.DataFrame) -> List[List[float]]:
    """
    Returns gait events timestamps
    :param df:
    :return: list of timestamps of gait
    """
    event_onset_idxs = df.index[(df["Name"] == "Foot Off") & (df["Context"].isin(["Left", "Right"]))].tolist()
    event_end_idxs = df.index[(df["Name"] == "Foot Strike") & (df["Context"].isin(["Left", "Right"]))].tolist()

    if not len(event_onset_idxs) > 0:
        return []

    if not len(event_end_idxs) > 0:
        return []

    event_end_idxs = event_end_idxs if event_end_idxs[0] > event_onset_idxs[0] else event_end_idxs[1:]

    event_onset = [df.iloc[i]["Time (s)"] for i in event_onset_idxs]
    event_end = [df.iloc[i]["Time (s)"] for i in event_end_idxs]

    events = list(map(list, zip(event_onset, event_end)))

    return events


def get_idxs_events_to_exclude(framerate: int,
                               last_frame: int,
                               ts_to_exclude_fog: List[List[float]],
                               ts_to_exclude_gait: List[List[float]]) -> List[List[int]]:
    """
    Returns the indexes corresponding to the timestamps to exclude. It merges freezing events and gait events
    :param framerate: recording framerate
    :param last_frame: last frame of the region of interest
    :param ts_to_exclude_fog: timestamps of freezing
    :param ts_to_exclude_gait: timestamps of gait
    :return: indexes of frames to be excluded
    """
    idxs_to_exclude = ts_to_exclude_fog + ts_to_exclude_gait
    for idx_pair in idxs_to_exclude:
        if np.isnan(idx_pair[1]):
            idx_pair[1] = last_frame
    idxs_to_exclude_frames = [[int(i[0] * framerate), int(i[1] * framerate)] for i in idxs_to_exclude]
    idxs_to_exclude_frames = sorted(idxs_to_exclude_frames, key=lambda x: x[0])

    return idxs_to_exclude_frames


def get_start_of_gait(framerate: int, ts_gait: List[List[float]]) -> int:
    """
    Returns the first frame of gait
    :param framerate: recording framerate
    :param ts_gait: timestamps of gait
    :return: first frame index of gait
    """
    idxs_gait = [[int(i[0] * framerate), int(i[1] * framerate)] for i in ts_gait]
    start_gait = idxs_gait[0][0]
    return start_gait


def create_exclusion_mask(idxs_to_exclude: List[List[float]], first_frame: int, last_frame: int) -> np.array:
    """
    Returns an array representing the region of interest of a run with
    True values only where there are no events to exclude
    :param idxs_to_exclude: list of the pairs of indexes to exclude [start, end]
    :param first_frame: first frame of the region of interest
    :param last_frame: last frame of the region of interest
    :return: exclusion mask
    """
    idxs_to_exclude = [[i[0] - first_frame, i[1] - first_frame] for i in idxs_to_exclude]
    full_idx_array = np.arange(first_frame, last_frame)
    mask = np.ones(full_idx_array.size, dtype=int)

    for idxs_ex in idxs_to_exclude:
        mask[idxs_ex[0]:idxs_ex[1]] = False

    # ensuring that there is a bool change even if the run starts or ends with a nlm
    if mask[0] == True:  # dont change to pep8 compliant notation. it breaks this.
        mask[0] = False
    if mask[-1] == True:
        mask[-1] = False

    return mask


def get_ts_from_exclusion_mask(mask: np.array, first_frame: int, framerate: int) -> List[List[float]]:
    """
    Gets bounds of the exclusion mask and converts it in timestamps
    :param mask:
    :param first_frame: first frame of the region of interest
    :param framerate: recording framerate
    :return:
    """
    events_bounds = np.where(np.diff(mask))[0]

    event_onset_idxs = events_bounds[0::2] + first_frame
    event_end_idxs = events_bounds[1::2] + first_frame

    event_onset = [i / framerate for i in event_onset_idxs]
    event_end = [i / framerate for i in event_end_idxs]

    return event_onset, event_end


def get_event_timestamps_nlm_active(framerate: int,
                                    first_frame: int,
                                    last_frame: int,
                                    ts_to_exclude_fog: List[List[float]],
                                    ts_to_exclude_gait: List[List[float]]):
    """
    Creates a mask representing the region of interest excluding all the passed events,
    and any event before the first gait frame
    :param framerate: recording framerate
    :param first_frame: first frame of the region of interest
    :param last_frame: last frame of the region of interest
    :param ts_to_exclude_fog: timestamps of freezing
    :param ts_to_exclude_gait: timestamps of gait
    :return: non-locomotion movements event after gait has happened
    """
    idxs_to_exclude = get_idxs_events_to_exclude(framerate,
                                                 last_frame,
                                                 ts_to_exclude_fog,
                                                 ts_to_exclude_gait)

    if first_frame is None:
        first_frame = idxs_to_exclude[0][0]
        last_frame = idxs_to_exclude[-1][1]
        print("########################################", first_frame, last_frame)

    mask = create_exclusion_mask(idxs_to_exclude, first_frame, last_frame)

    if len(ts_to_exclude_gait) > 0:
        start_gait = get_start_of_gait(framerate, ts_to_exclude_gait)
        mask[:start_gait - first_frame] = False  # setting to False all frames before gait

    event_onset, event_end = get_ts_from_exclusion_mask(mask, first_frame, framerate)

    events = list(map(list, zip(event_onset, event_end)))
    return events


def get_event_timestamps_nlm_rest(framerate: int,
                                  first_frame: int,
                                  last_frame: int,
                                  ts_to_exclude_fog: List[List[float]],
                                  ts_to_exclude_gait: List[List[float]]):
    """
    Creates a mask representing the region of interest excluding all the passed events,
    and any event after the first gait frame
    :param framerate: recording framerate
    :param first_frame: first frame of the region of interest
    :param last_frame: last frame of the region of interest
    :param ts_to_exclude_fog: timestamps of freezing
    :param ts_to_exclude_gait: timestamps of gait
    :return: non-locomotion movements event after gait has happened
    """
    # TODO fix fucking bug with first frame aligment
    idxs_to_exclude = get_idxs_events_to_exclude(framerate,
                                                            last_frame,
                                                            ts_to_exclude_fog,
                                                            ts_to_exclude_gait)


    if first_frame is None:
        first_frame = idxs_to_exclude[0][0]
        last_frame = idxs_to_exclude[-1][1]

    mask = create_exclusion_mask(idxs_to_exclude, first_frame, last_frame)

    for idxs_ex in idxs_to_exclude:
        mask[idxs_ex[0]:idxs_ex[1]] = False

    if len(ts_to_exclude_gait) > 0:
        start_gait = get_start_of_gait(framerate, ts_to_exclude_gait)
        mask[start_gait - first_frame:] = False  # setting to False all frames after gait

    event_onset, event_end = get_ts_from_exclusion_mask(mask, first_frame, framerate)

    events = list(map(list, zip(event_onset, event_end)))
    return events


def get_neural_correlate_psd(raw: np.array,
                             fs: float,
                             t_on: float,
                             t_off: float,
                             nfft: int,
                             nov: int,
                             zscore: bool = True) -> Tuple[np.array]:
    """
    Takes the beginning and end of an event, retrieves the corresponding chunk of neural data and computes the psd
    :param raw: raw neural single channel
    :param fs: sampling frequency
    :param t_on: beginning of event in seconds
    :param t_off: end of event in seconds
    :param nfft: nfft for fourier transform
    :param nov: overlap for fourier transform
    :param zscore: how to normalize
    :return: psd and corresponding frequencies
    """
    s_on = importing.time_to_sample(t_on, fs=fs, is_t1=True)
    s_off = importing.time_to_sample(t_off, fs=fs, is_t2=True)

    neural_event = raw[s_on:s_off]

    freqs, pxx = processing.calculate_power_spectral_density(neural_event, fs, nperseg=nfft, noverlap=nov,
                                                             scaling="spectrum")

    if zscore:
        pxx = stats.zscore(pxx)

    return pxx, freqs


def get_neural_ch(neural_path: str, ch: int, stream_name: str) -> Tuple[any]:
    """
    Retireves the raw neural signal and the sampling frequency
    :param neural_path: folder path to TDT recording
    :param ch: channel number 0 based
    :param stream_name: name  of the Stored listing
    :return: raw neural single channel and sampling frequency
    """
    neural = NeuralData(path=neural_path)
    neural.load_tdt_data(stream_name=stream_name)

    return neural.raw[ch], neural.fs


def time_to_frame_in_roi(timestamp: float, fs: float, first_frame: int) -> int:
    """
    Converts the time to frames in the region of interest (i.e. with an offset of the first frame)
    :param timestamp: time in seconds
    :param fs: sampling frequency
    :param first_frame: first frame of the region of interest
    :return:
    """
    frame = int(timestamp * fs) - first_frame
    frame = 0 if frame == -1 else frame
    if frame < 0:
        raise ValueError(f"First frame value is bigger than frame of event. "
                         f"First frame is {first_frame} and event frame is {frame}")

    return frame
