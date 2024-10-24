from neurokin.kinematic_data import KinematicDataRun
import pandas as pd


GAIT_PATH = "./neurokin/test_data/"
NEURAL_PATH = "./temp_data/neural/220915/ENWE_00052-220915-153059/"
CONFIGPATH = "../../config_elg.yaml"
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

#file = "../test_data/runway03.c3d"
file = "C:/Users/Elisa/Documents/GitHub/temp_data/c3d_fog/92/runway04.c3d"
dlc_file = "C:/Users/Elisa/Documents/GitHub/temp_data/CollectedData_Rafa.csv"

if __name__=="__main__":
    kin_data = KinematicDataRun(file, CONFIGPATH)  # creating a single run obj
    #kin_data.markers_df = pd.MultiIndex.from_frame(pd.read_csv(dlc_file, header=[0, 1, 2], index_col=[0]))
    kin_data.markers_df = pd.read_csv(dlc_file, header=[0, 1, 2], index_col=[0])

    kin_data.convert_DLC_like_to_df()

    #bodyparts_to_drop = [i[1] for i in kin_data.markers_df.columns.to_list()[::3] if i[1].startswith("*")]
    bodyparts_to_drop = ['Unnamed: 1_level_0_Unnamed: 1_level_1_Unnamed: 1_level_2', 'Unnamed: 2_level_0_Unnamed: 2_level_1_Unnamed: 2_level_2']
    kin_data.markers_df = kin_data.markers_df.drop(bodyparts_to_drop, axis=1, inplace=False)
    #kin_data.markers_df.columns.names = ["scorer", "bodyparts", "coords"]
    kin_data.bodyparts = [bp for bp in kin_data.bodyparts if bp not in bodyparts_to_drop]
    kin_data.extract_features()

    test = kin_data.get_binned_features()
    step_height = kin_data.get_trace_height(marker="lmtp", axis="z")
    step_length = kin_data.get_step_fwd_movement_on_bins(marker="lmtp", axis="y")

    #kin_data.features_df = pd.concat((kin_data.features_df, step_height, step_length), axis=1)

    print(kin_data.features_df.head(10))