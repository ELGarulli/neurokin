import csv
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from scipy import stats

from neurokin.neural_data import NeuralData
from neurokin.utils.neural import (processing, importing)


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
    df = pd.read_csv(csv_path, skiprows=skiprows, nrows=block_end, low_memory=False)

    return df


def get_first_last_frame_from_csv(csv_path: str) -> Tuple[int, int]:
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
    first_frame = int(df.iloc[0, 0])
    last_frame = int(df.iloc[-1, 0])
    return first_frame, last_frame


def get_freeze_ts_bound(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
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


def traspose_idxs(idxs_to_exclude: List[List[int]], first_frame: int) -> List[List[int]]:
    """
    Transposes the indexes to an array where first_frame is 0. Moreover it takes care of small idiosyncrasies when
    an event appears to start at frame -1

    :param idxs_to_exclude: list of indexes to be set to false
    :param first_frame: first frame of the region of interest
    :return: indexes transposed, same shape as inut
    """
    idxs_transposed = []

    for idx_pair in idxs_to_exclude:
        start = idx_pair[0] - first_frame if idx_pair[0] - first_frame >= 0 else 0
        end = idx_pair[1] - first_frame
        idxs_transposed.append([start, end])
    return idxs_transposed


def transpose_start_of_gait(start_of_gait: int, first_frame: int) -> int:
    """
    Transposes the index of the start of gait where first_frame is 0.
    Moreover it takes care of small idiosyncrasies when an event appears to start at frame -1

    :param start_of_gait: start of the first gait event
    :param first_frame: first frame of the region of interest
    :return:
    """
    return start_of_gait - first_frame if start_of_gait - first_frame > 0 else 0


def create_exclusion_mask(idxs_to_exclude: List[List[int]], first_frame: int, last_frame: int) -> np.array:
    """
    Returns an array representing the region of interest of a run with
    True values only where there are no events to exclude

    :param idxs_to_exclude: list of the pairs of indexes to exclude [start, end]
    :param first_frame: first frame of the region of interest
    :param last_frame: last frame of the region of interest
    :return: exclusion mask
    """
    idxs_to_exclude = traspose_idxs(idxs_to_exclude, first_frame)
    full_idx_array = np.arange(first_frame, last_frame)
    mask = np.ones(full_idx_array.size, dtype=bool)

    # ensuring that there is a bool change even if the run starts or ends with a nlm
    mask[0] = False
    mask[-1] = False

    for idxs_ex in idxs_to_exclude:
        mask[idxs_ex[0]:idxs_ex[1]] = False

    return mask


def get_ts_from_exclusion_mask(mask: np.array, first_frame: int, framerate: int) -> Tuple[List[float], List[float]]:
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

    mask = create_exclusion_mask(idxs_to_exclude, first_frame, last_frame)

    if len(ts_to_exclude_gait) > 0:
        start_gait = get_start_of_gait(framerate, ts_to_exclude_gait)
        start_gait = transpose_start_of_gait(start_gait, first_frame)
        mask[:start_gait] = False  # setting to False all frames before gait

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
    idxs_to_exclude = get_idxs_events_to_exclude(framerate,
                                                 last_frame,
                                                 ts_to_exclude_fog,
                                                 ts_to_exclude_gait)

    if first_frame is None:
        first_frame = idxs_to_exclude[0][0]
        last_frame = idxs_to_exclude[-1][1]

    mask = create_exclusion_mask(idxs_to_exclude, first_frame, last_frame)

    if len(ts_to_exclude_gait) > 0:
        start_gait = get_start_of_gait(framerate, ts_to_exclude_gait)
        start_gait = transpose_start_of_gait(start_gait, first_frame)
        mask[start_gait:] = False  # setting to False all frames after gait

    event_onset, event_end = get_ts_from_exclusion_mask(mask, first_frame, framerate)

    events = list(map(list, zip(event_onset, event_end)))
    return events


def get_neural_correlate_psd(raw: np.array,
                             fs: float,
                             t_on: float,
                             t_off: float,
                             nfft: int,
                             nov: int,
                             zscore: bool = True) -> Tuple[np.array, np.array]:
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


def get_neural_ch(neural_path: str, ch: int, stream_name: str) -> Tuple[np.array, float]:
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


def get_events_dict(event_path, skiprows, framerate):
    """
    Runs trough all the .csv files in the experiment_path and returns a dictionary with the timestamps of each
    event, sorted by the type.

    :param event_path: path to the run to be analyzed
    :param skiprows: how many rows to skip as a header of the .csv
    :param framerate: frame rate of the acquisition system
    :return: events dictionary
    """
    events_dict = {}
    events_dict.setdefault("gait", [])
    events_dict.setdefault("fog_active", [])
    events_dict.setdefault("fog_rest", [])
    events_dict.setdefault("nlm_active", [])
    events_dict.setdefault("nlm_rest", [])

    df = get_first_block_df(csv_path=event_path, skiprows=skiprows)

    try:
        first_frame, last_frame = get_first_last_frame_from_csv(event_path)
    except EmptyDataError:
        first_frame, last_frame = None, None
    except ParserError:
        first_frame, last_frame = None, None
    try:
        df.sort_values('Time (s)', inplace=True, ignore_index=True)
    except KeyError:
        print("No events were labelled, assuming non-locomotor movement on the whole run")
        event_onset, event_end = [first_frame / framerate], [last_frame / framerate]
        events_dict["nlm_rest"] = list(map(list, zip(event_onset, event_end)))
        return events_dict

    events_dict["gait"] = get_event_timestamps_gait(df)

    if len(events_dict["gait"]) > 0:
        events_dict["fog_active"] = get_event_timestamps_freezing_active(df, events_dict["gait"])

    events_dict["fog_rest"] = get_event_timestamps_freezing_rest(df, events_dict["gait"])

    events_detected_fog = events_dict["fog_active"] + events_dict["fog_rest"]

    if len(events_dict["gait"]) > 0:
        events_dict["nlm_active"] = get_event_timestamps_nlm_active(framerate=framerate,
                                                                    first_frame=first_frame,
                                                                    last_frame=last_frame,
                                                                    ts_to_exclude_fog=events_detected_fog,
                                                                    ts_to_exclude_gait=events_dict["gait"])

    events_dict["nlm_rest"] = get_event_timestamps_nlm_rest(framerate=framerate,
                                                            first_frame=first_frame,
                                                            last_frame=last_frame,
                                                            ts_to_exclude_fog=events_detected_fog,
                                                            ts_to_exclude_gait=events_dict["gait"])

    return events_dict


def get_neural_correlates_dict(neural_path,
                               channel_of_interest,
                               stream_names,
                               events_df,
                               time_cutoff):
    """
    Returns a dictionary with the raw neural data corresponding to each event in the input events dictionary, from the
    selected channel of interest. Events duration is capped at time_cutoff to ensure homogeneity in the duration.

    :param neural_path: path to the folder with neural data
    :param channel_of_interest: channel to extract the raw data from
    :param stream_names: list of possible stream_names where the neural data is stored
    :param events_df:
    :param time_cutoff:
    :return:
    """
    states = [key for key in events_df.columns if key.startswith("event")]
    neural_dict = {key: [] for key in states}
    fs = None
    print(neural_path)
    try:
        neural_path = neural_path + next(os.walk(neural_path))[1][0]
    except (StopIteration, IndexError) as e:
        print(f"{e} No neural data found for {neural_path}")
        return neural_dict, fs

    is_valid_name = False
    for name in stream_names:
        try:
            raw, fs = get_neural_ch(neural_path, channel_of_interest, name)
            is_valid_name = True
            break
        except (AttributeError, TypeError):
            pass
    if not is_valid_name:
        raise Exception("All stream names were invalid")

    for state in states:
        neural_dict[state] = get_single_neural_type(events_df, state, time_cutoff, fs, raw)

    return neural_dict, fs


def get_single_neural_type(events_df, event_type, time_cutoff, fs, raw):
    """
    Fetches the neural chunks corresponding to all the events of a certain type. All events
    are cut at the time_cutoff, to ensure consistency across different lengths (this implies
    events shorter than time_cutoff are dropped).

    :param events_df: dataframe containing all events timestamps
    :param event_type: type of event to fetch the neural correlates for
    :param time_cutoff: max length of an event to consider (shorter events are dropped)
    :param fs: sampling frequency of the neural data
    :param raw: raw neural data, single channel
    :return: returns a list of the neural chunks corresponding to the events, cropped to time_cutoff
    """
    correlates = []
    states_events_list = events_df[event_type].values
    for states_events in states_events_list:
        for t_onset, t_end in states_events:
            t_end = check_time_cutoff(t_onset, t_end, time_cutoff)
            if t_end is None:
                continue
            else:
                s_on = importing.time_to_sample(t_onset, fs=fs, is_t1=True)
                s_off = importing.time_to_sample(t_end, fs=fs, is_t2=True)
                correlates.append(raw[s_on:s_off])
                if len(raw[s_on:s_off]) == 0:
                    raise ValueError("Warning the neural chunk is of 0 length, "
                                     "check match between neural and states files")

    return correlates


def compute_psd_for_row(row, events_columns, nfft, noverlap, zscore):
    """
    Computes Power Spectra Density arrays for all the events columns.

    :param row: a pd.Dataframe row, containing a list of neural correlates for each event type and fs
    :param events_columns: which columns contain the lists of neural chunks
    :param nfft: what number of fft segments to use
    :param noverlap: the overlap of the fft segments
    :param zscore: whether to z-score the psd or not
    :return: Returns a pd.Series containing lists of Power Spectra Density arrays for all the events columns.
    """
    fs = row['fs']
    psd_results = {}
    for event in events_columns:
        psd_results[event] = get_psd_single_event_type(row[event], fs=fs, nfft=nfft, noverlap=noverlap, zscore=zscore)
    return pd.Series(psd_results)


def get_psd_single_event_type(raw_neural_list, fs, nfft, noverlap, zscore):
    """
    Computes the Power Spectra Density array for all element in the list (corresponding to a single type of event)

    :param raw_neural_list: list containing neural data arrays
    :param fs: sampling frequency of the neural data
    :param nfft: what number of fft segments to use
    :param noverlap: the overlap of the fft segments
    :param zscore: whether to z-score the psd or not
    :return: list of PSDs
    """
    psds = []
    freqs = None
    for raw_neural in raw_neural_list:
        freqs_psd, pxx = processing.calculate_power_spectral_density(raw_neural,
                                                                     fs,
                                                                     nperseg=nfft,
                                                                     noverlap=noverlap,
                                                                     scaling="spectrum")
        if freqs is None:
            freqs = freqs_psd

        sanity_check = np.array_equal(freqs, freqs_psd)
        if not sanity_check:
            raise ValueError("The frequencies in the PSD calculation are unequal for different events. "
                             "Check for consistency in the events length")
        if zscore:
            pxx = stats.zscore(pxx)
        psds.append(pxx)
    return psds


def check_time_cutoff(t_onset, t_end, time_cutoff):
    """
    Checks if the difference between t_onset and t_end is lower than the time_cutoff.
    If not then computes the end edge as t_onset + t_end and returns it.

    :param t_onset: start of the event
    :param t_end: end of the event
    :param time_cutoff: minimum length of the event
    :return: None if event is too short, t_onset + t_end otherwise
    """
    if t_end - t_onset < time_cutoff:
        return None
    elif t_end - t_onset > time_cutoff:
        t_end = t_onset + time_cutoff
        return t_end
