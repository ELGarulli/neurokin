import pandas as pd


def get_easy_metrics_on_bins(df_markers, df_features, window, overlap):
    df_ = df_features.columns.drop(df_markers.columns.tolist())

    means = df_.rolling(window=window, step=overlap).mean()
    return means
