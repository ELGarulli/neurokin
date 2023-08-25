import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import os
import sys
from neurokin.kinematic_data import KinematicDataRun
from neurokin.utils.neural import processing
from neurokin.experiments.neural_correlates import (get_first_block_df,
                                                    get_event_timestamps_gait,
                                                    get_event_timestamps_fog,
                                                    time_to_frame_in_roi)

folder = "C:/Users/Elisa/Documents/GitHub/temp_data/c3d/"
output_folder = "C:/Users/Elisa/Documents/GitHub/analysis/output_misc/"
CONFIGPATH = "C:/Users/Elisa/Documents/GitHub/neurokin/config.yaml"

experiment_structure = {"NWE00130": {"230329": [2, 3, 4, 5, 6, 7, 8, 9],
                                    "230330": [2, 3, 4, 5, 6, 8, 12, 13]}}

step_left_marker = "lmtp"
step_right_marker = "rmtp"

window = 200
overlap = 100

skiprows = 2

vicon_fs = 200

df_all_runs_binned = []

to_shift = ['lankle','lmtp','lcrest','lknee','rhip','lhip','rmtp','lshoulder','rcrest','rankle','rshoulder','rknee']


def shift_df(df, bodyparts, axis):
    df_shifted = df.copy()
    for bp in bodyparts:
        df_shifted["scorer", bp, axis] = df_shifted["scorer", bp, axis] + abs(min(df_shifted["scorer", bp, axis]))
    return df_shifted


for animal, days in experiment_structure.items():
    for day, runs in days.items():
        for r in runs:
            run_n = "0" + str(r) if r < 10 else str(r)
            path = folder + animal + "/" + day + "/"

            run = path + "runway" + run_n + ".c3d"
            print(animal, run)
            kin_data = KinematicDataRun(run, CONFIGPATH)
            kin_data.load_kinematics()
            kin_data.get_c3d_compliance()
            bodyparts_to_drop = [i[1] for i in kin_data.markers_df.columns.to_list()[::3] if i[1].startswith("*")]
            kin_data.markers_df = kin_data.markers_df.drop(bodyparts_to_drop, axis=1, level=1, inplace=False)
            kin_data.markers_df = shift_df(kin_data.markers_df, to_shift, "y")

            event_path = path + "runway" + run_n + ".csv"

            df = get_first_block_df(csv_path=event_path, skiprows=skiprows)
            df.sort_values('Time (s)', inplace=True, ignore_index=True)

            events = np.full((len(kin_data.markers_df), 3), (0, 255, 0))
            events_gait = get_event_timestamps_gait(df)
            # events_interr = get_event_timestamps_interruption(df)
            events_fog = get_event_timestamps_fog(df)
            if events_gait:
                for i, j in events_gait:
                    i = time_to_frame_in_roi(timestamp=i, fs=vicon_fs, first_frame=kin_data.trial_roi_start)
                    j = time_to_frame_in_roi(timestamp=j, fs=vicon_fs, first_frame=kin_data.trial_roi_start)
                    print(i, j)
                    events[i:j] = (0, 0, 255)

            if events_fog:
                for i, j in events_fog:
                    i = time_to_frame_in_roi(timestamp=i, fs=vicon_fs, first_frame=kin_data.trial_roi_start)
                    j = time_to_frame_in_roi(timestamp=j, fs=vicon_fs, first_frame=kin_data.trial_roi_start)
                    events[i:j] = (255, 0, 0)

            events_df = pd.DataFrame(events)

            binned = events_df.rolling(window=window, step=overlap).mean().add_suffix("_mean")