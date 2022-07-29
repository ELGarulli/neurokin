def load_kinematics(run_id):
    folder = LOAD_KIN_FOLDER + EXPDATE
    file = folder + "/KIN_" + EXPDATE + "_" + run_id + ".pkl"
    with open(file, 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict


def load_steps(run_id):
    folder = LOAD_KIN_FOLDER + EXPDATE
    file = folder + "/STEPS_" + EXPDATE + "_" + run_id + ".pkl"
    with open(file, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def load_first_frame(run_id):
    folder = LOAD_KIN_FOLDER + EXPDATE
    file = folder + "/FFRAME_" + EXPDATE + "_" + run_id + ".csv"
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            ff = int(float(rows[0]))
    return ff


def load_fs(run_id):
    folder = LOAD_KIN_FOLDER + EXPDATE
    file = folder + "/FS_" + EXPDATE + "_" + run_id + ".csv"
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)

        for rows in reader:
            fs = int(float(rows[0]))
    return fs
