from utils.import_export._import import (load_neural, load_kinematics, load_fs, load_steps, load_first_frame,
                                         load_neural_epochs)
from utils.import_export._export import (export_kinematic, export_frame, export_steps, export_fs, export_neural,
                                         export_neural_epochs)

def get_gait_data(eng, gait_files, load=False):
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
            h = eng.minEx_0(gait_files[file_name], PATHNAME, nargout=1)
            h = eng.minEx_1(h, nargout=1)
            s = True
            steps_dict = events_detected(h, "Data_R", "Data_L", s)
            #steps_dict = event_int(h,  "Data_R", "Data_L", s, trend=True)

            h = check_step_detection(eng=eng, h=h, steps_dict=steps_dict, s=s)

            kin_data = import_kin_data(h)
            first_frame = h['first_frame']
            DATAFILE = h['FILENAME']
            full = PATHNAME + DATAFILE
            print("Used file: " + full)

            kin_dict = parse_kin(kin_data, steps_dict['left_toe_off'],
                                 steps_dict['left_heel_strike'],
                                 steps_dict['right_toe_off'],
                                 steps_dict['right_heel_strike'],
                                 steps_dict['maximas'])

            gait_dict[file_name] = kin_dict
            parse_dict[file_name] = steps_dict
            first_frames[file_name] = first_frame
            sample_rate = h['freq']
            export_kinematic(file_name, kin_dict)
            export_frame(file_name, first_frame)
            export_steps(file_name, steps_dict)
            export_fs(file_name, sample_rate)

    gait_data = {'gait_dict': gait_dict, 'first_frame': first_frames, 'parse_dict': parse_dict,
                 'sample_rate': sample_rate}

    return gait_data


def check_step_detection(eng, h, steps_dict, s):
    """

    :param eng: matlab engine API
    :param h: objects
    :param steps_dict: dictionary of steps from detected events
    :param s:
    :return:
    """

    # TODO s = tkinter.messagebox.askyesno(title="Rerun event detection?", message="Are you happy with the results?")
    for i in range(len(steps_dict['left_toe_off'][0])-1):
        if steps_dict['left_toe_off'][0][i] == steps_dict['left_heel_strike'][0][i]:
            if i != 0:
                steps_dict['left_heel_strike'][0][i] = (steps_dict['maximas']['Data_L'][i]+(steps_dict['left_heel_strike']
                                                                                          [0][i-1]-steps_dict['maximas']
                                                                                          ['Data_L'][i-1]))
            else:
                if steps_dict['left_toe_off'][0][0] < steps_dict['right_toe_off'][0][0]:
                    steps_dict['left_heel_strike'][0][i] = steps_dict['right_toe_off'][0][i]
                else:
                    steps_dict['left_heel_strike'][0][i] = steps_dict['right_toe_off'][0][i+1]

    for j in range(len(steps_dict['right_toe_off'][0])-1):
        if steps_dict['right_toe_off'][0][j] == steps_dict['right_heel_strike'][0][j]:
            if j != 0:
                steps_dict['right_heel_strike'][0][j] = (steps_dict['maximas']['Data_R'][j]+(steps_dict['right_heel_strike']
                                                                                          [0][j-1]-steps_dict['maximas']
                                                                                          ['Data_R'][j-1]))
            else:
                if steps_dict['right_toe_off'][0][0] < steps_dict['left_toe_off'][0][0]:
                    steps_dict['right_heel_strike'][0][j] = steps_dict['left_toe_off'][0][j]
                else:
                    steps_dict['right_heel_strike'][0][j] = steps_dict['left_toe_off'][0][j+1]

    h = eng.minEx_2(h, h['Data_L'], h['Data_R'],
                    h['TIME'],
                    steps_dict['left_toe_off'],
                    steps_dict['right_toe_off'],
                    steps_dict['left_heel_strike'],
                    steps_dict['right_heel_strike'],
                    steps_dict['uneven_l'],
                    steps_dict['uneven_r'], nargout=1)

    if not s:
        steps_dict = events_detected(h, "Data_R", "Data_L", s)
        h = eng.minEx_2(h, h['Data_L'], h['Data_R'],
                        h['TIME'],
                        steps_dict['left_toe_off'],
                        steps_dict['right_toe_off'],
                        steps_dict['left_heel_strike'],
                        steps_dict['right_heel_strike'],
                        steps_dict['uneven_l'],
                        steps_dict['uneven_r'], nargout=1)
    return h
