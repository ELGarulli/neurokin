from matplotlib import pyplot as plt
from neurokin.kinematic_data import KinematicDataRun
from neurokin.neural_data import NeuralData
from neurokin.utils.neural import processing, neural_plot
import sys
import os

GAIT_PATH = "./neurokin/test_data/"
#NEURAL_PATH = "./temp_data/neural/220915/ENWE_00052-220915-153059/"
CONFIGPATH = "./config.yaml"
GAIT_RECORDING_FS = 80
output_folder = "./"


shift_reference_marker = "lmtp_y"
tilt_reference_marker = "lmtp_z"
# to_tilt = ["rshoulder_z", "rcrest_z", "rhip_z",
#            "rknee_z", "rankle_z", "rmtp_z",
#            "lshoulder_z", "lcrest_z", "lhip_z",
#            "lknee_z", "lankle_z", "lmtp_z"]
# to_shift = ["rshoulder_y", "rcrest_y", "rhip_y",
#             "rknee_y", "rankle_y", "rmtp_y",
#             "lshoulder_y", "lcrest_y", "lhip_y",
#             "lknee_y", "lankle_y", "lmtp_y"]

step_left_marker = "ForePawLeft" ### use z here
step_right_marker = "ForePawRight"

c3d_files = []
csv_files = []
for file in os.listdir(GAIT_PATH):
    if file.endswith(".c3d"):
        c3d_files.append(GAIT_PATH + file)
    if file.endswith(".csv"):
        csv_files.append(GAIT_PATH + file)

file = csv_files[0]
testfile = GAIT_PATH + 'reduced_test_df.csv'
print('current file:', file)


kin_data = KinematicDataRun(file, CONFIGPATH)       # creating a single run obj
kin_data.load_kinematics()
kin_data.fs = GAIT_RECORDING_FS
kin_data.get_c3d_compliance()




kin_data.extract_features()