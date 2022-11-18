import os
from matplotlib import pyplot as plt
from kinematic_data import KinematicDataRun
from neural_data import NeuralData

from neurokino_class import Neurokino



GAIT_PATH = "C:\\Users\\Elisa\\Documents\\GitHub\\temp_data\\c3d\\NWE00052\\220915\\"
NEURAL_PATH = "C:\\Users\\Elisa\\Documents\\GitHub\\temp_data\\neural\\220915\\ENWE_00052-220915-153059\\"
CONFIGPATH = "..\\config.yaml"
output_folder = "..\\demo_output/"

nperseg = 2**10
noverlap = int(nperseg/8)

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
kin_data.load_kinematics(correct_tilt=True,         # loading data and tilt-shift correcting
                         correct_shift=True,
                         to_tilt=to_tilt,
                         to_shift=to_shift,
                         shift_reference_marker=shift_reference_marker,
                         tilt_reference_marker=tilt_reference_marker)

kin_data.compute_gait_cycles_bounds(left_marker=step_left_marker,   # computing left right bounds of steps
                                    right_marker=step_right_marker)


neural_data = NeuralData(path=NEURAL_PATH)
neural_data.load_tdt_data(stream_name="NPr1", sync_present=True, stim_stream_name="Wav1")
neural_data.pick_sync_data(0)


neurokino = Neurokino(neural_object=neural_data, kinematic_object=kin_data)
avg_spectrogram, freq, t = neurokino.get_steps_neural_correlates(channel=6, side="l", nperseg=nperseg, noverlap=noverlap)
extent = min(t), max(t), freq[0], freq[-1]

plt.imshow(avg_spectrogram,  extent=extent, #vmin=, vmax=,cmap,
                         origin='upper')