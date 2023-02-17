from neurokin.ggait_legacy import GGait
import os
import logging
import matlab.engine

logger = logging.getLogger(__name__)

PATH = "C://Users//Elisa//Documents//GitHub//temp_data//c3d//deficit_comparison//6ohda//test//"
print_step_partition = False

c3d_files = []
for file in os.listdir(PATH):
    if file.endswith(".c3d"):
        c3d_files.append(file)

success_gait_anal = []
failed_gait_anal = []

for file in c3d_files:
    run = file.split(".")[0]
    try:
        kin_data = GGait(file, PATH, 200)
        kin_data.load_kinematics()
        kin_data.compute_gait_cycles_timestamp()
        if print_step_partition:
            kin_data.print_step_partition(output_folder=PATH, run=run)
        kin_data.get_gait_param()
        kin_data.test_minEx3()
        success_gait_anal.append(file)

    except matlab.engine.MatlabExecutionError as error:
        logger.error(error)
        print("failed to create gait file for " + file + "\n please create gait file manually")
        failed_gait_anal.append(file)




print("*************************** REPORT ******************************* \n" +
      "Gait file successfully created for: " + ", ".join(success_gait_anal) + "\n" +
      "Gait file failed to be created for: " + ", ".join(failed_gait_anal) + "\n" +
      "******************************************************************")