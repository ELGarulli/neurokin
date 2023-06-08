import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import math


def interpolate_low_likelihood_intervals(
    df: pd.DataFrame,
    marker_ids: List[str],
    max_interval_length: int,
    framerate: float,
) -> pd.DataFrame:
    """
    Interpolate low likelihood intervals in a dataframe of marker coordinates.
    Input: dataframe of marker coordinates, list of marker ids, max interval length in seconds, framerate
    Output: dataframe of marker coordinates with interpolated low likelihood intervals
    """
    interpolated_df = df.copy()
    for marker_id in marker_ids:
        outlier_series = interpolated_df["scorer"][marker_id]["x"].notna()
        (
            low_likelihood_interval_border_idxs,
            all_low_likelihood_interval_border_idxs,
        ) = get_low_likelihood_interval_border_idxs(
            outlier_series=outlier_series,
            max_interval_length=max_interval_length,
            framerate=framerate,
        )
        for start_idx, end_idx in all_low_likelihood_interval_border_idxs:
            if end_idx + 1 < interpolated_df["scorer"][marker_id]["x"].shape[0]:
                interpolated_df["scorer"][marker_id]["x"][
                    start_idx : end_idx + 1
                ] = np.NaN
                interpolated_df["scorer"][marker_id]["y"][
                    start_idx : end_idx + 1
                ] = np.NaN
                interpolated_df["scorer"][marker_id]["z"][
                    start_idx : end_idx + 1
                ] = np.NaN
        for start_idx, end_idx in low_likelihood_interval_border_idxs:
            if (start_idx - 1 >= 0) and (
                end_idx + 2 < interpolated_df["scorer"][marker_id]["x"].shape[0]
            ):
                if (
                    not math.isnan(
                        interpolated_df["scorer"][marker_id]["x"][end_idx + 2]
                    )
                ) and (
                    not math.isnan(
                        interpolated_df["scorer"][marker_id]["x"][start_idx - 1]
                    )
                ):
                    interpolated_df["scorer"][marker_id]["x"][
                        start_idx - 1 : end_idx + 2
                    ] = interpolated_df["scorer"][marker_id]["x"][
                        start_idx - 1 : end_idx + 2
                    ].interpolate()

                    interpolated_df["scorer"][marker_id]["y"][
                        start_idx - 1 : end_idx + 2
                    ] = interpolated_df["scorer"][marker_id]["y"][
                        start_idx - 1 : end_idx + 2
                    ].interpolate()

                    interpolated_df["scorer"][marker_id]["z"][
                        start_idx - 1 : end_idx + 2
                    ] = interpolated_df["scorer"][marker_id]["z"][
                        start_idx - 1 : end_idx + 2
                    ].interpolate()

    return interpolated_df


def get_low_likelihood_interval_border_idxs(
    outlier_series: pd.Series,
    framerate: float,
    max_interval_length: int,
) -> List[Tuple[int, int]]:
    """
    Get indices of low likelihood intervals in a series of booleans.
    Input: series of booleans of nan values, framerate, max interval length in seconds
    Output: list of tuples of start and end indices of low likelihood intervals
    """
    bad_likelihood_idxs = np.where(outlier_series.values == False)[0]
    short_low_likelihood_interval_border_idxs = get_interval_border_idxs(
        all_matching_idxs=bad_likelihood_idxs,
        framerate=framerate,
        max_interval_duration=max_interval_length * framerate,
    )
    all_low_likelihood_interval_border_idxs = get_interval_border_idxs(
        all_matching_idxs=bad_likelihood_idxs,
        framerate=framerate,
        max_interval_duration=outlier_series.shape[0] * framerate,
    )
    return (
        short_low_likelihood_interval_border_idxs,
        all_low_likelihood_interval_border_idxs,
    )


def get_interval_border_idxs(
    all_matching_idxs: np.ndarray,
    framerate: float,
    min_interval_duration: Optional[float] = None,
    max_interval_duration: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """
    Get border indices of intervals in a series of indices.
    """
    interval_border_idxs = []
    if all_matching_idxs.shape[0] >= 1:
        step_idxs = np.where(np.diff(all_matching_idxs) > 1)[0]
        step_end_idxs = np.concatenate(
            [step_idxs, np.array([all_matching_idxs.shape[0] - 1])]
        )
        step_start_idxs = np.concatenate([np.array([0]), step_idxs + 1])
        interval_start_idxs = all_matching_idxs[step_start_idxs]
        interval_end_idxs = all_matching_idxs[step_end_idxs]
        for start_idx, end_idx in zip(interval_start_idxs, interval_end_idxs):
            interval_frame_count = (end_idx + 1) - start_idx
            interval_duration = interval_frame_count * framerate
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
    return interval_border_idxs


def get_max_odd_n_frames_for_time_interval(fps: int, time_interval=0.5) -> int:
    """
    Get the maximum number of frames that fit into a time interval.
    """

    assert type(fps) == int, '"fps" has to be an integer!'
    frames_per_time_interval = fps * time_interval
    if frames_per_time_interval % 2 == 0:
        max_odd_frame_count = frames_per_time_interval - 1
    elif frames_per_time_interval == int(frames_per_time_interval):
        max_odd_frame_count = frames_per_time_interval
    else:
        frames_per_time_interval = int(frames_per_time_interval)
        if frames_per_time_interval % 2 == 0:
            max_odd_frame_count = frames_per_time_interval - 1
        else:
            max_odd_frame_count = frames_per_time_interval
    assert (
        max_odd_frame_count > 0
    ), f"The specified time interval is too short to fit an odd number of frames"
    return int(max_odd_frame_count)
