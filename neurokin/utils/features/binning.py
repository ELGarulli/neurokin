import pandas as pd


def get_easy_metrics_on_bins(df_markers, df_features, window, overlap):
    df_markers_no_scorer = df_markers.copy().droplevel("scorer", axis=1)
    cols = ['_'.join(col) for col in df_markers_no_scorer.columns.values]
    df_grouped = df_features.copy()
    df_grouped = df_grouped.droplevel("scorer", axis=1)
    df_grouped.columns = df_grouped.columns.to_flat_index()
    df_grouped.columns = ['_'.join(col) for col in df_grouped.columns.values]

    df_grouped.drop(cols, inplace=True, axis=1)

    means = df_grouped.rolling(window=window, step=overlap).mean().add_suffix("_mean")
    stds = df_grouped.rolling(window=window, step=overlap).std().add_suffix("_std")
    maxs = df_grouped.rolling(window=window, step=overlap).max().add_suffix("_max")
    mins = df_grouped.rolling(window=window, step=overlap).max().add_suffix("_min")
    metrics = pd.concat((means, stds, maxs, mins), axis=1)
    return metrics
