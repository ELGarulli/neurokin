import pandas as pd
from scipy.signal import savgol_filter


def apply_savgol_filter(
    df: pd.DataFrame,  # DataFrame to smooth
    window_length: int,  # Odd integer (!) of sliding window size in frames to consider for smoothing
    polyorder: int = 3,  # Order of the polynom used for the savgol filter
) -> pd.DataFrame:
    """
    Smoothes the DataFrame with a savgol filter
    Note: window_length has to be an odd integer and bigger than the polyorder (default: 3)!
    """

    final_smoothed_df = df.copy()
    for column in df.columns:
        smoothed_df = df.loc[:, column].copy()
        smoothed_df = savgol_filter(
            x=smoothed_df, window_length=window_length, polyorder=polyorder, axis=0
        )
        final_smoothed_df.loc[:, column] = smoothed_df
    return final_smoothed_df
