import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from neurokin.utils.features.core import FeatureExtraction, DefaultParams


def _get_interval_border_idxs(
    all_matching_idxs: np.ndarray,
    df: pd.DataFrame,
    colname: str,
    min_interval_duration: Optional[float] = None,
    max_interval_duration: Optional[float] = None,
    fps: int = 80,
) -> List[Tuple[int, int]]:
    """
    Function written by Dennis Segebarth
    Input: indices of a boolean series that are the same between bodyparts
    (e.g. all bodyparts are immobile/freezing)
    Output: list of tuples with start and end index of the intervals where all bodyparts
     are immobile/freezing
    """
    interval_border_idxs = []

    # check if there are any intervals
    if all_matching_idxs.shape[0] >= 1:
        # get start and end indices of intervals
        step_idxs = np.where(np.diff(all_matching_idxs) > 1)[0]
        step_end_idxs = np.concatenate(
            [step_idxs, np.array([all_matching_idxs.shape[0] - 1])]
        )
        step_start_idxs = np.concatenate([np.array([0]), step_idxs + 1])
        interval_start_idxs = all_matching_idxs[step_start_idxs]
        interval_end_idxs = all_matching_idxs[step_end_idxs]
        # loop over all intervals
        for start_idx, end_idx in zip(interval_start_idxs, interval_end_idxs):
            interval_frame_count = (end_idx + 1) - start_idx
            interval_duration = interval_frame_count * (1 / fps)
            if (min_interval_duration != None) and (max_interval_duration != None):
                append_interval = (
                    min_interval_duration <= interval_duration <= max_interval_duration
                )
            elif min_interval_duration != None:
                append_interval = min_interval_duration <= interval_duration
            elif max_interval_duration != None:
                append_interval = interval_duration <= max_interval_duration
            else:
                append_interval = True
            if append_interval:
                interval_border_idxs.append((start_idx, end_idx))
                # write bout_duration in df
                df.loc[
                    start_idx:end_idx, (colname + "_bout_duration")
                ] = interval_duration
                df.loc[start_idx:end_idx, colname] = True
    return interval_border_idxs, df


class ImmobilityAndFreezing(FeatureExtraction):
    """
    Classifies bodyparts as immobile if the bodyparts speed is below a speed threshold
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of m should be computed
    Output: df with joint angle data for input markers
    """

    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {
            "immobility_threshold": 3,
            "markers_for_immobility": ["Snout", "TailBase"],
            "minimum_duration_immobility": 0.1,
            "minimum_duration_freezing": 0.5,
            "fps": 80,
        }
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {
            "immobility_threshold": int,
            "markers_for_immobility": [str],
            "minimum_duration_immobility": int,
            "minimum_duration_freezing": int,
            "fps": int,
        }
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        # Check whether immobility is already calculated
        if ("scorer", "subject", "immobility") in marker_df.columns:
            pass
        else:
            # Check whether speed is calculated for all markers critical for immobility
            if 'speed' in marker_df.loc[:, ('scorer', params['markers_for_immobility'])].columns.all():

                valid_idxs_per_marker_id = []
                for bodypart_id in params["markers_for_immobility"]:
                    temp_df = self._copy_filtered_columns_of_df(
                        df_to_filter=marker_df,
                        marker_id_filter=bodypart_id,
                        coords_filter=["speed"],
                    )
                    # remove multiindex to make it easier to work with
                    temp_df = temp_df.droplevel([0, 1], axis=1)

                    # create new col for immobility, set val to false
                    temp_df["immobility"] = False
                    # filter for rows in which the speed is less than the immobility threshold.
                    temp_df.loc[
                        (temp_df["speed"] < params["immobility_threshold"]),
                        "immobility",
                    ] = True

                    valid_idxs_per_marker_id.append(
                        temp_df.loc[temp_df["immobility"] == True].index.values
                    )
                shared_valid_idxs_for_all_markers = valid_idxs_per_marker_id[0]
                if len(valid_idxs_per_marker_id) > 1:
                    for next_set_of_valid_idxs in valid_idxs_per_marker_id[1:]:
                        shared_valid_idxs_for_all_markers = np.intersect1d(
                            shared_valid_idxs_for_all_markers, next_set_of_valid_idxs
                        )

                # To Do: merging immobility bouts

                # create col in df for immobility
                immobility_and_freezing_df = self._copy_filtered_columns_of_df(
                    df_to_filter=marker_df,
                    marker_id_filter=source_marker_ids,
                    coords_filter=["speed"],
                )

                immobility_and_freezing_df["immobility"] = False

                # calculate bout duration and create col in df for it
                (
                    immobility_interval_border_idxs,
                    immobility_and_freezing_df,
                ) = _get_interval_border_idxs(
                    all_matching_idxs=shared_valid_idxs_for_all_markers,
                    min_interval_duration=params["minimum_duration_immobility"],
                    fps=params["fps"],
                    df=immobility_and_freezing_df,
                    colname="immobility",
                )

                # create col in df for freezing
                immobility_and_freezing_df["freezing"] = False

                # calculate bout duration and create col in df for freezing
                (
                    freezing_interval_border_idxs,
                    immobility_and_freezing_df,
                ) = _get_interval_border_idxs(
                    all_matching_idxs=shared_valid_idxs_for_all_markers,
                    min_interval_duration=params["minimum_duration_freezing"],
                    fps=params["fps"],
                    df=immobility_and_freezing_df,
                    colname="freezing",
                )

                # transform dfs into multiindex df

                # immobility column (boolean)
                immobility_col_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="immobility",
                    data=immobility_and_freezing_df.loc[:, "immobility"],
                )

                # immobility bout duration
                immobility_bout_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="immobility_bout_duration",
                    data=immobility_and_freezing_df.loc[:, "immobility_bout_duration"],
                )

                # freezing column (boolean)
                freezing_col_df = self.convert_singleindex_to_multiindex_df(
                    scorer="scorer",
                    bodypart="subject",
                    axis="freezing",
                    data=immobility_and_freezing_df.loc[:, "freezing"],
                )

                # freezing bout duration
                if immobility_and_freezing_df['freezing_bout_duration'].any():
                    freezing_bout_df = self.convert_singleindex_to_multiindex_df(
                        scorer="scorer",
                        bodypart="subject",
                        axis="freezing_bout_duration",
                        data=immobility_and_freezing_df.loc[:, "freezing_bout_duration"],
                )
                else:
                    print('No freezing bouts detected')

                final_immobility_df = pd.concat(
                    [
                        immobility_col_df,
                        immobility_bout_df,
                        freezing_col_df,
                        freezing_bout_df,
                    ],
                    axis=1,
                )
                return final_immobility_df
            #
            #
            except ValueError:
                raise ValueError(
                    "Be sure to extract speed for all bodyparts that you want to use for immobility!"
                )


# class GaitDisruptionBouts(FeatureExtraction):
#     """
#     Classifies bodyparts as immobile if the bodyparts speed is below a speed threshold
#     Input: df with positon data (i.e. DLC output), source_marker_ids: List of m should be computed
#     Output: df with joint angle data for input markers
#     """
#
#     @property
#     def input_type(self) -> str:
#         return "markers"
#
#     @property
#     def default_values(self) -> Dict[str, Any]:
#         default_values = {
#             "minimum_duration_immobility_for_gait_disruption": 0.1,
#             "fps": 80,
#             "min_amount_step_cycles_before_immobility": 3,
#         }
#         return default_values
#
#     @property
#     def default_value_types(self) -> Dict[str, List[type]]:
#         default_types = {
#             "minimum_duration_immobility_for_gait_disruption": int,
#             "fps": int,
#             "min_amount_step_cycles_before_immobility": int,
#         }
#         return default_types
#
#     def _run_feature_extraction(
#         self,
#         source_marker_ids: List[str],
#         marker_df: pd.DataFrame,
#         params: Dict[str, Any],
#     ) -> pd.DataFrame:
#         # Check whether immobility is already calculated
#         if ("scorer", "subject", "immobility") in marker_df.columns:
#             pass
#         else:
#             try:
#                 # slice marker df
#                 # check conditions: step cycles, immobility time
#                 filtered_df = self._copy_filtered_columns_of_df(
#                     df_to_filter=marker_df,
#                     marker_id_filter="subject",
#                     coords_filter=["immobility_bout_duration", "gait cycle"],
#                 )
#                 # remove multiindex to make it easier to work with
#                 filtered_df = filtered_df.droplevel([0, 1], axis=1)
#
#                 # create new col for immobility, set val to false
#                 filtered_df["gait_disruption"] = False
#
#                 #idx_conditions_fulfilled  # check for number of gait cycles + check for immobility duration
#
#                 #filtered_df.loc[idx_conditions_fulfilled, "gait_disruption"] = True
#                 pass
#             except ValueError:
#                 raise ValueError(
#                     "Be sure to extract speed for all bodyparts that you want to use for immobility!"
#                 )
