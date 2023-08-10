import os
import glob
import csv

import numpy as np
import pandas as pd
from scipy import stats

from neurokin.utils.neural import (processing, importing)
from neurokin.neural_data import NeuralData


def get_event_len(csv_path):
    with open(csv_path, newline='') as f:
        c = 0
        r = csv.reader(f)
        for l in r:
            if not l:
                break
            c += 1
    return c


def get_events_df(event_path, skiprows):
    block_end = get_event_len(event_path) - skiprows
    df = pd.read_csv(event_path, skiprows=skiprows, nrows=block_end)

    return df


def get_event_timestamps_fog(df):
    event_onset_idxs = df.index[(df["Name"] == "Foot Off") & (df["Context"] == "General")].tolist()
    if not len(event_onset_idxs) > 0:
        return None

    event_onset = [df.iloc[i]["Time (s)"] for i in event_onset_idxs]
    event_end = [df.iloc[i + 1]["Time (s)"] for i in event_onset_idxs if not i + 1 == len(df)]

    #events = tuple(zip(event_onset, event_end))
    events = list(map(list, zip(event_onset, event_end)))

    return events


def get_event_timestamps_gait(df):
    event_onset_idxs = df.index[(df["Name"] == "Foot Off") & (df["Context"].isin(["Left", "Right"]))].tolist()
    event_end_idxs = df.index[(df["Name"] == "Foot Strike") & (df["Context"].isin(["Left", "Right"]))].tolist()

    if not len(event_onset_idxs) > 0:
        return None

    if not len(event_end_idxs) > 0:
        return None

    event_end_idxs = event_end_idxs if event_end_idxs[0] > event_onset_idxs[0] else event_end_idxs[1:]

    event_onset = [df.iloc[i]["Time (s)"] for i in event_onset_idxs]
    event_end = [df.iloc[i]["Time (s)"] for i in event_end_idxs]

    #events = tuple(zip(event_onset, event_end))
    events = list(map(list, zip(event_onset, event_end)))

    return events


def get_event_timestamps_interruption(df):
    event_onset_idxs = df.index[(df["Name"] == "Foot Strike") & (df["Context"].isin(["Left", "Right"]))].tolist()
    event_end_idxs = df.index[(df["Name"] == "Foot Off") & (df["Context"].isin(["Left", "Right"]))].tolist()  # [1:]

    if not len(event_onset_idxs) > 0:
        # event_onset_idxs = df.index[(df["Name"] == "Event") & (df["Context"] == "General")]
        # event_end_idxs = df.index[(df["Name"] == "Foot Off") & (df["Context"].isin(["Left", "Right", "General"]))].tolist()
        # print(event_onset_idxs, event_end_idxs)
        # if not len(event_onset_idxs) > 0:
        return None

    if not len(event_end_idxs) > 0:
        return None

    event_end_idxs = event_end_idxs if event_end_idxs[0] > event_onset_idxs[0] else event_end_idxs[1:]

    event_onset = [df.iloc[i]["Time (s)"] for i in event_onset_idxs]
    event_end = [df.iloc[i]["Time (s)"] for i in event_end_idxs]

    #events = list(zip(event_onset, event_end))
    events = list(map(list, zip(event_onset, event_end)))

    return events


def get_event_timestamps_non_locomotion_movement(df, framerate, idxs_to_exclude):
    first_event_frame = int(df.iloc[0]["Time (s)"] * framerate)
    last_event_frame = int(df.iloc[-2]["Time (s)"] * framerate)  # -2 as the last row is empty in the Time (s) column

    for idx_pair in idxs_to_exclude:
        if np.isnan(idx_pair[1]):
            idx_pair[1] = last_event_frame
    idxs_to_exclude_frames = [[int(i[0] * framerate), int(i[1] * framerate)] for i in idxs_to_exclude]

    full_idx_array = np.arange(first_event_frame, last_event_frame)
    mask = np.ones(full_idx_array.size, dtype=int)
    for idxs_ex in idxs_to_exclude_frames:
        mask[idxs_ex[0]:idxs_ex[1]] = False

    if mask.size == 0:
        event_onset = []
        event_end = []
        events = list(map(list, zip(event_onset, event_end)))
        return events

    # ensuring that there is a bool change even if the run starts or ends with a nlm
    if mask[0] == True:   # dont change to pep8 compliant notation. it breaks this.
        mask[0] = False
    if mask[-1] == True:
        mask[-1] = False

    events_bounds = np.where(np.diff(mask))[0]

    event_onset_idxs = events_bounds[0::2]
    event_end_idxs = events_bounds[1::2]

    event_onset = [i / framerate for i in event_onset_idxs]
    event_end = [i / framerate for i in event_end_idxs]
    #events = list(zip(event_onset, event_end))
    events = list(map(list, zip(event_onset, event_end)))
    return events


def get_neural_correlate(raw, fs, t_on, t_off, nfft, nov, zscore=True):
    s_on = importing.time_to_sample(t_on, fs=fs, is_t1=True)
    s_off = importing.time_to_sample(t_off, fs=fs, is_t2=True)

    neural_event = raw[s_on:s_off]

    pxx, freqs, t = processing.get_spectrogram_data(fs=fs, raw=neural_event, nfft=nfft, noverlap=nov)

    if zscore:
        pxx = stats.zscore(pxx)

    return pxx, freqs, t


def get_neural_correlate_psd(raw, fs, t_on, t_off, nfft, nov, zscore=True):
    s_on = importing.time_to_sample(t_on, fs=fs, is_t1=True)
    s_off = importing.time_to_sample(t_off, fs=fs, is_t2=True)

    neural_event = raw[s_on:s_off]

    freqs, pxx = processing.calculate_power_spectral_density(neural_event, fs, nperseg=nfft, noverlap=nov,
                                                             scaling="spectrum")

    if zscore:
        pxx = stats.zscore(pxx)

    return pxx, freqs


def get_neural_ch(neural_path, ch, stream_name):
    neural = NeuralData(path=neural_path)
    neural.load_tdt_data(stream_name=stream_name)

    return neural.raw[ch], neural.fs


def get_all_neural_correlates(nfft, nov, skiprows, experiment_path, animals, neural_events_files, ch, time_cutoff):
    pxxs = {}
    psds = {}

    for animal in animals:
        for key, value in neural_events_files[animal].items():

            if not key in pxxs:
                pxxs[key] = []
            if not key in psds:
                psds[key] = []

            for run in value:

                run_path = experiment_path + animal + "/" + run + "/"

                event_path = glob.glob(run_path + "*.csv")[0]
                neural_path = run_path + next(os.walk(run_path))[1][0]

                df = get_events_df(event_path=event_path, skiprows=skiprows)
                df.sort_values('Time (s)', inplace=True, ignore_index=True)

                if key == "fog":
                    events = get_event_timestamps_fog(df)
                elif key == "gait":
                    events = get_event_timestamps_gait(df)
                elif key == "interruption":
                    events = get_event_timestamps_interruption(df)

                raw, fs = get_neural_ch(neural_path, ch)
                for t_onset, t_end in events:
                    if t_end - t_onset < time_cutoff:
                        continue
                    elif t_end - t_onset > time_cutoff:
                        t_end = t_onset + time_cutoff

                    pxx, freqs_spec, t = get_neural_correlate(raw,
                                                              fs,
                                                              t_on=t_onset,
                                                              t_off=t_end,
                                                              nfft=nfft,
                                                              nov=nov,
                                                              zscore=False)
                    pxxs[key].append(pxx)

                    psd, freqs_psd = get_neural_correlate_psd(raw,
                                                              fs,
                                                              t_on=t_onset,
                                                              t_off=t_end,
                                                              nfft=nfft * 2 ** 3,
                                                              nov=nov,
                                                              zscore=True)
                    psds[key].append(psd)

    return pxxs, psds, freqs_psd, freqs_spec, t


def time_to_frame_in_roi(timestamp, fs, first_frame):
    frame = int(timestamp * fs) - first_frame
    frame = 0 if frame == -1 else frame
    if frame < 0:
        raise ValueError(f"First frame value is bigger than frame of event. "
                         f"First frame is {first_frame} and event frame is {frame}")

    return frame
