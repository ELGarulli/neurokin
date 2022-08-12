from kinematic_data import KinematicData


kin_data = KinematicData("runway01.c3d", "../temp_data/")
kin_data.load_kinematics()
kin_data.compute_gait_cycles_timestamp()
kin_data.get_gait_param()
print("aight")