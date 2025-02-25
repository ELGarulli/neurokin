import dlc2kinematics
import os
from neurokin.kinematic_data import KinematicDataRun

GAIT_PATH = "C:\\Users\\Elisa\\Documents\\GitHub\\temp_data\\c3d\\NWE00052\\220915\\"
NEURAL_PATH = "C:\\Users\\Elisa\\Documents\\GitHub\\temp_data\\neural\\220915\\ENWE_00052-220915-153059\\"
CONFIGPATH = "..\\config.yaml"

c3d_files = []
for file in os.listdir(GAIT_PATH):
    if file.endswith(".c3d"):
        c3d_files.append(GAIT_PATH + file)
file = c3d_files[3]

step_left_marker = "lmtp"
step_right_marker = "rmtp"

kin_data = KinematicDataRun(file, CONFIGPATH)  # creating a single run obj
kin_data.load_kinematics(correct_tilt=False,  # loading data and tilt-shift correcting
                         correct_shift=False)

kin_data.convert_DLC_like_to_df(smooth=True)

kin_data.compute_gait_cycles_bounds(left_marker=step_left_marker,  # computing left right bounds of steps
                                    right_marker=step_right_marker)

df = kin_data.markers_df

df_vel = dlc2kinematics.compute_velocity(df, bodyparts=['all'])
df_acc = dlc2kinematics.compute_acceleration(df, bodyparts=['all'])
df_speed = dlc2kinematics.compute_speed(df, bodyparts=['all'])
joint_angles = dlc2kinematics.compute_joint_angles(df, kin_data.config["angles"]["joints"])
joint_vel = dlc2kinematics.compute_joint_velocity(joint_angles)
joint_acc = dlc2kinematics.compute_joint_acceleration(joint_angles)

empty = kin_data.create_empty_features_df(kin_data.bodyparts, ["max_angle", "min_angle", "phase_max_amplitude"])
test = kin_data.get_angles_features(empty, left_side="L", name_starts_with=True)
