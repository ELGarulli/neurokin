import pandas as pd


def import_kin_data(h): #, tsv_data -> Dict[str, np.ndarray]:
    mtp_l = h['mkr']['LMTP']['z']
    mtp_r = h['mkr']['RMTP']['z']
    kin_data = {
        "mtp_l": mtp_l,
        "mtp_r": mtp_r
    }
    return kin_data


def import_ggait_data(text_file, write_to_csv): # -> Dict[str, np.ndarray]: (r before strings to make raw string)
    read_file = pd.read_csv(text_file)
    read_file.to_csv(write_to_csv)
    df = pd.read_csv('runway10_GAIT_SUM.csv', delimiter='\t')
    df = df.to_dict('series')
    list_of_keys = ["GAIT_TYPE", "LIMB", "ANIMAL", "CONDITION 1", "CONDITION 2", "LIMB SIDE", "SPEED", "G ONSET",
                    "G END",
                    "T ONSET", "T END", "Rt=0 HINDLIMBS", "FLIGHT", "STANCE ONE LIMB", "STANCE TWO LIMBS",
                    "STANCE THREE LIMBS", "STANCE FOUR LIMBS", "Rt=0 HL IPSIFL", "Rt=0 HL CONFL", "FootLADDER_h",
                    "FootLADDER_v", "Perc_FootLADDER", "MIN ELE 5", "MAX ELE 5", "MAX JOINT 4", "MAX FOOTROT",
                    "MIN JOINT 4", "AMP ELE 5", "AMP JOINT 4", "AMP FOOTROT", "MIN SPEEDJOINT 4",
                    "MAX SPEEDJOINT 4", "AMP SPEEDJOINT 4", "PC1", "PH5 PH4", "R FOOT - TOE", "R ANKLE-MTP",
                    "FOOT-TOE timingMIN", "FOOT-TOE timingMAX", "Yfor STANCE", "Zfor STANCE", "Xfor STANCE", "BWS",
                    "LIMB_lag", "LIMB_R", "HIP lag", "HIP R", "KNEE lag", "KNEE R", "ANKLE lag", "ANKLE R", "MTP lag",
                    "MTP R", "STANCE", "STEP HEIGHT", "T DRAG END", "PC2", "PC3", "PC4", "PCA5",
                    "LAG HINDLIMBS", "LAG HL IPSIFL", "LAG HL CONFL", "PH5", "AMP5", "LAG FOOT - TOE", "LAG ANKLE-MTP",
                    "Kin AVE", "Xfor singleSTANCE", "Yfor singleSTANCE", "Zfor singleSTANCE", "Stance IPSIFORE",
                    "Swing IPSIFORE", "Stance CONTRAFORE",
                    "Swing CONTRAFORE"]  # excl. "MIN FOOTROT","PC1","PC2", "PC3", "PC4",
    for key in list_of_keys:
        del df[key]
    return df #returns a dictionary with only the wanted gait attributes

