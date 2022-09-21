from kinematic_data import KinematicData
import os

PATH = "../temp_data/c3d/NWE00054/220915/"

c3d_files = []
for file in os.listdir(PATH):
    if file.endswith(".c3d"):
        c3d_files.append(file)

success_gait_anal = []
failed_gait_anal = []

for file in c3d_files:
    try:
        kin_data = KinematicData(file, PATH)
        kin_data.load_kinematics()
        kin_data.compute_gait_cycles_timestamp()
        kin_data.get_gait_param()
        kin_data.test_minEx3()
        success_gait_anal.append(file)
    except:
        print("failed to create gait file for " + file + "\n please create gait file manually")
        failed_gait_anal.append(file)
        pass

print("*************************** REPORT ******************************* \n" +
      "Gait file successfully created for: " + ", ".join(success_gait_anal) + "\n" +
      "Gait file failed to be created for: " + ", ".join(failed_gait_anal) + "\n" +
      "******************************************************************")
