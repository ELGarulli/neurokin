#from utils.import_export._import import (load_neural, load_kinematics, load_fs, load_steps, load_first_frame,
#                                      load_neural_epochs)
#from utils.import_export._export import (export_kinematic, export_frame, export_steps, export_fs, export_neural,
#                                         export_neural_epochs)
#from ggait import get_gait_cycle_bounds
#from kinematics import import_toe_z_data


def get_gait_data(eng, gait_files, pathname, load=False):
    """
    bypasses matlab gui and accesses gait information from c3d file --> saves information in a dictionary
    :param eng: matlab engine API (requires matlab.engine.shareEngine command in MATLAB)
    :param gait_files: dictionary of file names for a certain recording session
    :param load: True if data has already been processed and can be accessed from dictionaries within Python
    :return: gait_data dictionary
    """
    gait_dict = {}
    first_frames = {}
    parse_dict = {}
    for file_name in gait_files.keys():

        if load:
            gait_dict[file_name] = load_kinematics(file_name)
            parse_dict[file_name] = load_steps(file_name)
            first_frames[file_name] = load_first_frame(file_name)
            sample_rate = load_fs(file_name)
            print("Loaded kinematics from " + file_name)
        else:
            h = eng.minEx_0(gait_files[file_name], pathname, nargout=1)
            h = eng.minEx_1(h, nargout=1)
            s = True
            left_steps_dict = get_gait_cycle_bounds(h, "Data_L", s)
            right_steps_dict = get_gait_cycle_bounds(h, "Data_R", s)

            h = check_step_detection(eng=eng, h=h, left_steps_dict=left_steps_dict, right_steps_dict=right_steps_dict,
                                     s=s)

            DATAFILE = h['FILENAME']
            full = pathname + DATAFILE
            print("Used file: " + full)

    gait_data = {'gait_dict': gait_dict,
                 'first_frame': first_frames,
                 'parse_dict': parse_dict,
                 "left_steps": left_steps_dict, #TODO fix loading of the steps if file exists already
                 "right_steps": right_steps_dict,
                 'sample_rate': sample_rate}

    return gait_data


def check_step_detection(eng, h, left_steps_dict, right_steps_dict, s):
    """

    :param eng: matlab engine API
    :param h: objects
    :param steps_dict: dictionary of steps from detected events
    :param s:
    :return:
    """
    uneven_r = 0
    uneven_l = 0
    # TODO s = tkinter.messagebox.askyesno(title="Rerun event detection?", message="Are you happy with the results?")
    for i in range(len(left_steps_dict['toe_off'][0]) - 1):
        if left_steps_dict['toe_off'][0][i] == left_steps_dict['heel_strike'][0][i]:
            if i != 0:
                left_steps_dict['heel_strike'][0][i] = (
                        left_steps_dict['maximas'][i] + (left_steps_dict['heel_strike']
                                                         [0][i - 1] - left_steps_dict['maximas'][i - 1]))
            else:
                if left_steps_dict['toe_off'][0][0] < right_steps_dict['toe_off'][0][0]:
                    left_steps_dict['heel_strike'][0][i] = right_steps_dict['toe_off'][0][i]
                else:
                    left_steps_dict['heel_strike'][0][i] = right_steps_dict['toe_off'][0][i + 1]

    for j in range(len(right_steps_dict['toe_off'][0]) - 1):
        if right_steps_dict['toe_off'][0][j] == right_steps_dict['heel_strike'][0][j]:
            if j != 0:
                right_steps_dict['heel_strike'][0][j] = (
                        right_steps_dict['maximas'][j] + (right_steps_dict['heel_strike']
                                                          [0][j - 1] - right_steps_dict['maximas'][j - 1]))
            else:
                if right_steps_dict['toe_off'][0][0] < left_steps_dict['toe_off'][0][0]:
                    right_steps_dict['heel_strike'][0][j] = left_steps_dict['toe_off'][0][j]
                else:
                    right_steps_dict['heel_strike'][0][j] = left_steps_dict['toe_off'][0][j + 1]

    h = eng.minEx_2(h, h['Data_L'], h['Data_R'],
                    h['TIME'],
                    left_steps_dict['toe_off'],
                    right_steps_dict['toe_off'],
                    left_steps_dict['heel_strike'],
                    right_steps_dict['heel_strike'],
                    uneven_l,
                    uneven_r,
                    nargout=1)

    if not s:
        steps_dict = get_gait_cycle_bounds(h, "Data_R", "Data_L", s)
        h = eng.minEx_2(h, h['Data_L'], h['Data_R'],
                        h['TIME'],
                        left_steps_dict['toe_off'],
                        steps_dict['toe_off'],
                        left_steps_dict['heel_strike'],
                        steps_dict['heel_strike'],
                        uneven_l,
                        uneven_r,
                        nargout=1)
    return h
