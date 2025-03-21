import numpy as np
import pandas as pd


def parse_df_features_for_binning(df_markers, df_features):
    df_markers_no_scorer = df_markers.copy().droplevel("scorer", axis=1)
    cols = ['_'.join(col) for col in df_markers_no_scorer.columns.values]
    df_grouped = df_features.copy()
    df_grouped = df_grouped.droplevel("scorer", axis=1)
    df_grouped.columns = df_grouped.columns.to_flat_index()
    df_grouped.columns = ['_'.join(col) for col in df_grouped.columns.values]
    #test_names = [col.replace(" ", "_") for col in df_grouped.columns.values]
    df_grouped.drop(cols, inplace=True, axis=1)

    return df_grouped

def get_easy_metrics_on_bins(df_grouped, window, overlap):

    means = df_grouped.rolling(window=window, step=overlap).mean().add_suffix("_mean")
    stds = df_grouped.rolling(window=window, step=overlap).std().add_suffix("_std")
    maxs = df_grouped.rolling(window=window, step=overlap).max().add_suffix("_max")
    mins = df_grouped.rolling(window=window, step=overlap).max().add_suffix("_min")
    metrics = pd.concat((means, stds, maxs, mins), axis=1)
    return metrics


def get_step_height_on_bins(df_markers, marker, window, overlap, axis):
    df_markers_no_scorer = df_markers.copy().droplevel("scorer", axis=1)
    trace_column = df_markers_no_scorer[marker, axis]
    metric = trace_column.rolling(window=window, step=overlap).apply(get_step_height)
    pd_metric = metric.to_frame(name=marker+"_height")
    return pd_metric


def get_step_fwd_movement_on_bins(df_markers, marker, window, overlap, axis):
    df_markers_no_scorer = df_markers.copy().droplevel("scorer", axis=1)
    trace_column = df_markers_no_scorer[marker, axis]
    metric = trace_column.rolling(window=window, step=overlap).apply(get_step_len)
    pd_metric = metric.to_frame(name=marker+"_fwd_move")
    return pd_metric


def get_step_height(trace):
    trace = trace + np.min(trace)
    return np.absolute(np.max(trace) - np.min(trace))


def get_step_len(trace):
    return trace.values[-1] - trace.values[0]
