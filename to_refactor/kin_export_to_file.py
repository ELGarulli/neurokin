def export_kinematic(run_id, kin_data):
    folder = LOAD_KIN_FOLDER + EXPDATE
    filename = folder + "/KIN_" + EXPDATE + "_" + run_id + ".pkl"
    CHECK_FOLDER = os.path.isdir(folder)
    if not CHECK_FOLDER:
        os.makedirs(folder)
    with open(filename, 'wb') as f:
        pickle.dump(kin_data, f)
    return


def export_steps(run_id, steps):
    folder = LOAD_KIN_FOLDER + EXPDATE
    filename = folder + "/STEPS_" + EXPDATE + "_" + run_id + ".pkl"
    CHECK_FOLDER = os.path.isdir(folder)
    if not CHECK_FOLDER:
        os.makedirs(folder)
    with open(filename, 'wb') as f:
        pickle.dump(steps, f)
    return


def export_frame(run_id, first_frame):
    folder = LOAD_KIN_FOLDER + EXPDATE
    filename = folder + "/FFRAME_" + EXPDATE + "_" + run_id + ".csv"
    CHECK_FOLDER = os.path.isdir(folder)
    if not CHECK_FOLDER:
        os.makedirs(folder)
    with open(filename, 'w') as f:
            f.write(str(first_frame))
    return


def export_fs(run_id, fs):
    folder = LOAD_KIN_FOLDER + EXPDATE
    filename = folder + "/FS_" + EXPDATE + "_" + run_id + ".csv"
    CHECK_FOLDER = os.path.isdir(folder)
    if not CHECK_FOLDER:
        os.makedirs(folder)
    with open(filename, 'w') as f:
        f.write(str(fs))
    return
