from kinematic_data import KinematicDataRun
import os
from utils.kinematics import gait_params_basics
from matplotlib import pyplot as plt
import numpy as np
############## EXPERIMENT SETTING PANEL #################

PATH = "../temp_data/c3d/NWE00052/220915/"
CONFIGPATH = "./config.yaml"
RECORDING_FS = 200
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

################ GETTING ALL C3D FILES IN THE FOLDER #############

c3d_files = []
for file in os.listdir(PATH):
    if file.endswith(".c3d"):
        c3d_files.append(PATH + file)

success_gait_anal = []
failed_gait_anal = []

####################### RUNNING ANALYSIS ########################

for file in c3d_files:
    kin_data = KinematicDataRun(file, CONFIGPATH)  # creating a single run obj
    kin_data.load_kinematics(correct_tilt=True,  # loading data and tilt-shift correcting
                             correct_shift=True,
                             to_tilt=to_tilt,
                             to_shift=to_shift,
                             shift_reference_marker=shift_reference_marker,
                             tilt_reference_marker=tilt_reference_marker)

    kin_data.compute_gait_cycles_bounds(left_marker=step_left_marker,  # computing left right bounds of steps
                                        right_marker=step_right_marker,
                                        recording_fs=RECORDING_FS)
    kin_data.print_step_partition(step_left_marker, step_right_marker)  # print step partition for inspection only
    kin_data.compute_angles_joints()  # computing angle joints
    kin_data.split_in_unilateral_df(left_side="left", right_side="right", name_starts_with=True,
                                    expected_columns_number=3)
    kin_data.gait_param_to_csv() # saving data to csv

    phase_shift = []
    a_phase = []
    b_phase = []
    for i in range(len(kin_data.left_mtp_lift)-1):
        a = kin_data.markers_df["lhip_z"][kin_data.left_mtp_lift[i]:kin_data.left_mtp_lift[i+1]]
        b = kin_data.markers_df["lknee_z"][kin_data.left_mtp_lift[i]:kin_data.left_mtp_lift[i+1]]
        phase_shift.append(gait_params_basics.compare_phase(a, b))
        a_phase.append(gait_params_basics.get_phase_at_max_amplitude(a))
        b_phase.append(gait_params_basics.get_phase_at_max_amplitude(b))
    kin_data.stepwise_gait_features_to_csv()
    success_gait_anal.append(file.split("/")[-1])  # note success or fail of analysis
    # except:
    #    #TODO specify ErrorType
    #    failed_gait_anal.append(file.split("/")[-1])                        # note success or fail of analysis

################################## REPORT ####################################

print("*************************** REPORT ******************************* \n" +
      "Gait file successfully created for: " + ", ".join(success_gait_anal) + "\n" +
      "Gait file failed to be created for: " + ", ".join(failed_gait_anal) + "\n" +
      "******************************************************************")
