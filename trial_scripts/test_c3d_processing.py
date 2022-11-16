import numpy as np

from utils.kinematics import kinematics_processing, angles
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal

filename = '../../temp_data/c3d/NWE00052/220915/runway01.c3d'
filename_csv = '../../temp_data/c3d/NWE00052/220915/runway01.csv'

temp_df = pd.read_csv(filename_csv, sep="\t")

sides = ["r", "l"]

to_detrend = ["rshoulder_z", "rcrest_z", "rhip_z", "rknee_z", "rankle_z", "rmtp_z", "lshoulder_z", "lcrest_z", "lhip_z",
              "lknee_z", "lankle_z", "lmtp_z"]
to_shift = ["rshoulder_y", "rcrest_y", "rhip_y", "rknee_y", "rankle_y", "rmtp_y", "lshoulder_y", "lcrest_y", "lhip_y",
            "lknee_y", "lankle_y", "lmtp_y"]

df_detrend = kinematics_processing.tilt_correct(temp_df, "lmtp_z", to_detrend)
df_detrend = kinematics_processing.shift_correct(df_detrend, "lmtp_y", to_shift)
fig, axs = plt.subplots(1, 2)

xs = temp_df["lmtp_x"]
ys = temp_df["lmtp_y"]
zs = temp_df["lmtp_z"]
ys_ = temp_df["rhip_y"]
zs_ = temp_df["rhip_z"]
axs[0].scatter(ys, zs)

axs[1].scatter(ys_, zs_)

ys_detrend = df_detrend["lmtp_y"]
zs_detrend = df_detrend["lmtp_z"]
ys_detrend_ = df_detrend["rhip_y"]
zs_detrend_ = df_detrend["rhip_z"]

axs[0].scatter(ys_detrend, zs_detrend)
axs[1].scatter(ys_detrend_, zs_detrend_)
axs[0].set_title("Left mtp marker")
axs[1].set_title("Right hip marker")

plt.show()
