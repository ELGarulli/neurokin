from kinematic_data import KinematicDataRun
import os

PATH = "../temp_data/c3d/NWE00052/220915/"

c3d_files = []
for file in os.listdir(PATH):
    if file.endswith(".c3d"):
        c3d_files.append(PATH + file)

success_gait_anal = []
failed_gait_anal = []

to_tilt = ["rshoulder_z", "rcrest_z", "rhip_z", "rknee_z", "rankle_z", "rmtp_z", "lshoulder_z", "lcrest_z", "lhip_z",
           "lknee_z", "lankle_z", "lmtp_z"]
to_shift = ["rshoulder_y", "rcrest_y", "rhip_y", "rknee_y", "rankle_y", "rmtp_y", "lshoulder_y", "lcrest_y", "lhip_y",
            "lknee_y", "lankle_y", "lmtp_y"]

shift_reference_marker = "lmtp_y"
tilt_reference_marker = "lmtp_z"

for file in c3d_files:

    kin_data = KinematicDataRun(file)
    kin_data.load_kinematics(correct_tilt=True,
                             correct_shift=True,
                             to_tilt=to_tilt,
                             to_shift=to_shift,
                             shift_reference_marker=shift_reference_marker,
                             tilt_reference_marker=tilt_reference_marker)
    kin_data.compute_gait_cycles_timestamp(left_marker="lmtp_z", right_marker="rmtp_z", recording_fs=200)
    success_gait_anal.append(file.split("/")[-1])
    #except:
    #    print("failed to create gait file for " + file + "\n please create gait file manually")
    #    failed_gait_anal.append(file)
    #    pass

print("*************************** REPORT ******************************* \n" +
      "Gait file successfully created for: " + ", ".join(success_gait_anal) + "\n" +
      "Gait file failed to be created for: " + ", ".join(failed_gait_anal) + "\n" +
      "******************************************************************")
