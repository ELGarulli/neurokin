from neurokin.kinematic_data import KinematicDataRun
import os

GAIT_PATH = "C:\\Users\\Elisa\\Documents\\GitHub\\temp_data\\c3d\\NWE00052\\220915\\"
NEURAL_PATH = "C:\\Users\\Elisa\\Documents\\GitHub\\temp_data\\neural\\220915\\ENWE_00052-220915-153059\\"
CONFIGPATH = "C:\\Users\\Elisa\\Documents\\GitHub\\config.yaml"
output_folder = "C:\\Users\\Elisa\\Documents\\GitHub\\demo_output/"

nperseg = 2**12
noverlap =0

shift_reference_marker = "lmtp_y"
tilt_reference_marker = "lmtp_z"
to_tilt = ["rshoulder_z", "rcrest_z", "rhip_z",
           "rknee_z", "rankle_z", "rmtp_z",
           "lshoulder_z", "lcrest_z", "lhip_z",
           "lknee_z", "lankle_z", "lmtp_z"]
to_shift = ["rshoulder_y", "rcrest_y", "rhip_y",
            "rknee_y", "rankle_y", "rmtp_y",
            "lshoulder_y", "lcrest_y", "lhip_y",
            "lknee_y", "lankle_y", "lmtp_y"]

step_left_marker = "lmtp_z"
step_right_marker = "rmtp_z"

c3d_files = []
for file in os.listdir(GAIT_PATH):
    if file.endswith(".c3d"):
        c3d_files.append(GAIT_PATH + file)
file = c3d_files[3]


kin_data = KinematicDataRun(file, CONFIGPATH)       # creating a single run obj
kin_data.load_kinematics(correct_tilt=False,         # loading data and tilt-shift correcting
                         correct_shift=False)