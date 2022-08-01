import pandas as pd
from constants.kinematics import KINEMATICS_PARAMS_TO_DROP
from typing import Dict
from numpy.typing import ArrayLike


def import_toe_z_data(h)-> Dict[str, ArrayLike]:
    mtp_l = h['mkr']['LMTP']['z']
    mtp_r = h['mkr']['RMTP']['z']
    kin_data = {
        "mtp_l": mtp_l,
        "mtp_r": mtp_r
    }
    return kin_data


def import_ggait_data(text_file) -> Dict[str, ArrayLike]:
    """
    Returns a dictionary with only the wanted gait attributes
    :param text_file:
    :param write_to_csv:
    :return:
    """
    ggait_file = pd.read_csv(text_file)
    df = pd.read_csv(ggait_file, delimiter='\t')
    df = df.to_dict('series')
    for key in KINEMATICS_PARAMS_TO_DROP:
        del df[key]
    return df

