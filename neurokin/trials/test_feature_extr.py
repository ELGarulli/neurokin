from neurokin.kinematic_data import KinematicDataRun
from neurokin.neural_data import NeuralData
from neurokin.utils.neural import processing, neural_plot

GAIT_PATH = "./neurokin/test_data/"
NEURAL_PATH = "./temp_data/neural/220915/ENWE_00052-220915-153059/"
CONFIGPATH = "../../config.yaml"
GAIT_RECORDING_FS = 200
output_folder = "./"

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

step_left_marker = "lmtp"
step_right_marker = "rmtp"

file = "../test_data/runway03.c3d"

kin_data = KinematicDataRun(file, CONFIGPATH)  # creating a single run obj
kin_data.load_kinematics()

kin_data.compute_gait_cycles_bounds(left_marker=step_left_marker,  # computing left right bounds of steps
                                    right_marker=step_right_marker)
kin_data.print_step_partition(step_left_marker, step_right_marker,
                              output_folder)  # print step partition for inspection only

kin_data.get_c3d_compliance()

bodyparts_to_drop = [i[1] for i in kin_data.markers_df.columns.to_list()[::3] if i[1].startswith("*")]
kin_data.markers_df = kin_data.markers_df.drop(bodyparts_to_drop, axis=1, level=1, inplace=False)
kin_data.bodyparts =  [bp for bp in kin_data.bodyparts if bp not in bodyparts_to_drop]
kin_data.extract_features()

test = kin_data.get_binned_features()

print(kin_data.features_df.head(10))
