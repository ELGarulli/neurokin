from utils.kinematics import kinematics_processing, angles
import pandas as pd
from matplotlib import pyplot as plt

filename = '../temp_data/c3d/NWE00052/220915/runway01.c3d'
filename_csv = '../temp_data/c3d/NWE00052/220915/runway01.csv'

temp_df = pd.read_csv(filename_csv, sep="\t")

sides = ["r", "l"]

for angle, markers in angles.angle_sides.items():
    fig, ax = plt.subplots()
    for s in sides:
        side_angle = []
        for frame in range(len(temp_df)):
            a, b, c = kinematics_processing.get_points(temp_df, markers, s, frame)
            side_angle.append(kinematics_processing.get_angle(a, b, c))
        ax.plot(side_angle, label=s)
    fig.suptitle(angle)
    plt.show()
