from typing import List
import pandas as pd
import numpy as np

"""
to do: 
include axis_flag into df creation -> 3 or 1 column needed?
look for prettier solutions -> dlc2kin has a function for creating empty dfs

"""
class DLC_conversion():
        def concat_dlc_df_to_df(df, dlc_df):
            return pd.concat([df, dlc_df], axis=1)

        def remove_scorer_index(df: pd.DataFrame)-> None:
            sliced_df = df[df.keys()[0][0]]
            return sliced_df

        def add_scorer_index(df: pd.DataFrame, marker_id_filter: List[str], axis_flag: str)-> None:
            expanded_df = create_empty_df(scorer = 'feature', bodyparts = marker_id_filter, frames_no = np.shape(df)[0])
            count = 0
            for bp in marker_id_filter:
                for a in axis:
                    df['scorer', bp, a] = df[:, count]
                    count += 1
            return expanded_df

        def create_empty_df(scorer, bodyparts, frames_no, axis_flag: str):
            """
            Creates empty dataframe to receive 3d data from c3d file
            Parameters
            ----------
            scorer: string
                mock data scorer
            bodyparts: list
                bodyparts that will be in the dataframe
            frames_no: int
                number of frames of the recording
            Outputs
            -------
            df: empty dataframe with shape compatible with dlc2kinematics
            """

            df = None
            a = np.full((frames_no, 3), np.nan)
            for bodypart in bodyparts:
                pdindex = pd.MultiIndex.from_product(
                    [[scorer], [bodypart], ["x", "y", "z"]],
                    names=["scorer", "bodyparts", "coords"],
                )

                frame = pd.DataFrame(a, columns=pdindex, index=range(0, frames_no))
                df = pd.concat([frame, df], axis=1)
            return df

