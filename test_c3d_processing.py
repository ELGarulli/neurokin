from utils.kinematics import kinematics_processing, angles
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal

filename = '../temp_data/c3d/NWE00052/220915/runway01.c3d'
filename_csv = '../temp_data/c3d/NWE00052/220915/runway01.csv'

temp_df = pd.read_csv(filename_csv, sep="\t")

sides = ["r", "l"]

# for angle, markers in angles.angle_sides.items():
#    fig, ax = plt.subplots()
#    for s in sides:
#        side_angle = []
#        for frame in range(len(temp_df)):
#            a, b, c = kinematics_processing.get_points(temp_df, markers, s, frame)
#            side_angle.append(kinematics_processing.get_angle(a, b, c))
#        ax.plot(side_angle, label=s)
#    fig.suptitle(angle)
#    plt.show()


to_detrend = ["rshoulder_z", "rcrest_z", "rhip_z", "rknee_z", "rankle_z", "rmtp_z", "lshoulder_z", "lcrest_z", "lhip_z",
              "lknee_z", "lankle_z", "lmtp_z"]

df_detrend = kinematics_processing.shift_correct(temp_df, "lmtp_z", to_detrend)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xs = temp_df["lmtp_x"]
ys = temp_df["lmtp_y"]
zs = temp_df["lmtp_z"]
ax.scatter(xs, ys, zs)

xs_detrend = df_detrend["lmtp_x"]
ys_detrend = df_detrend["lmtp_y"]
zs_detrend = df_detrend["lmtp_z"]
ax.scatter(xs_detrend, ys_detrend, zs_detrend)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# ax.plot(temp_df["lmtp_x"], label="x")
# ax.plot(temp_df["lmtp_y"], label="y")
# ax.plot(temp_df["lmtp_z"], label="z")
# plt.legend()
plt.show()
