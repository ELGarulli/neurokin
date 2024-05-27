import numpy as np
from scipy import stats
from neurokin.utils.experiments.neural_states_helper import compute_ci

def plot_psd_single_state(ax, df,  group, condition, state, freqs, color, idx_min, idx_max, stat="std", linewidth=3):
    """

    :param ax:
    :param group:
    :param condition:
    :param state:
    :param freqs:
    :param idx_min:
    :param idx_max:
    :param color:
    :param stat:
    :param linewidth:
    :return:
    """

    f = freqs[idx_min:idx_max]
    data = df[(df["group"] == group) & (df["condition"]==condition)][state]
    data_mean = data.mean(axis=0)[idx_min:idx_max]
    ax.plot(f, data_mean, color=color, linewidth=linewidth)
    add_shades(ax, f, data, idx_min, idx_max, color, stat=stat)


def add_shades(ax, f, df, idx_min, idx_max, color, stat="std"):
    """
    Helper function to plot shades.
    :param ax: the ax to plot on
    :param f: frequencies to use on the x-axis
    :param df: the data to be plot (array like of states)
    :param idx_min: minimum frequency index to plot
    :param idx_max: maximum frequency index to plot
    :param color: color to plot the shade in, with 0.3 alpha factor
    :param stat: which statistics to use for shades. "std" for standard deviation,
                "sem" for standard error of mean, "ci" for confidence interval
    :return:
    """

    data = np.array(df.values.tolist())
    if stat == "sem":
        s = data.sem(axis=0)
    elif stat == "ci":
        s = data.apply(compute_ci, axis=0)
    elif stat == "std":
        s = data.std(axis=0)
    else:
        raise ValueError(f"The statistic value {stat} is unsupported, please choose between std, sem or ci")

    lower = data.mean(axis=0) - s
    upper = data.mean(axis=0) + s
    ax.fill_between(f, lower[idx_min:idx_max], upper[idx_min:idx_max], alpha=0.3, color=color)

def _plot_psd_single_state(ax, group_cond_dataset, freqs, idx_min, idx_max, color, stat="std", linewidth=3):
    """
    Helper function to plot the mean of psds and the respective shade, indicating a statistical interval
    :param ax: the ax to plot on
    :param group_cond_dataset: the data to be plot (array like of states)
    :param freqs: frequencies to use on the x-axis
    :param idx_min: minimum frequency index to plot
    :param idx_max: maximum frequency index to plot
    :param color: color to plot the mean in. The shade will be the same color, with 0.3 alpha factor
    :param stat: which statistics to use for shades. "std" for standard deviation,
                "sem" for standard error of mean, "ci" for confidence interval
    :param linewidth: how thick to have the mean line
    :return:
    """
    f = freqs[idx_min:idx_max]
    mean_data = np.mean(group_cond_dataset, axis=0)[idx_min:idx_max]

    ax.plot(f, mean_data, color=color, linewidth=linewidth)
    add_shades(ax, f, group_cond_dataset, idx_min, idx_max, color, stat=stat)


def _add_shades(ax, f, data, idx_min, idx_max, color, stat="std"):
    """
    Helper function to plot shades.
    :param ax: the ax to plot on
    :param f: frequencies to use on the x-axis
    :param data: the data to be plot (array like of states)
    :param idx_min: minimum frequency index to plot
    :param idx_max: maximum frequency index to plot
    :param color: color to plot the shade in, with 0.3 alpha factor
    :param stat: which statistics to use for shades. "std" for standard deviation,
                "sem" for standard error of mean, "ci" for confidence interval
    :return:
    """
    if stat == "sem":
        lower, upper = get_sem_bounds(data, idx_min, idx_max)
    elif stat == "ci":
        lower, upper = get_ci_bounds(data, idx_min, idx_max)
    else:
        lower, upper = get_std_bounds(data, idx_min, idx_max)
    ax.fill_between(f, lower, upper, alpha=0.3, color=color)


def get_ci_bounds(data, idx_min=None, idx_max=None):
    """
    Helper function to compute the confidence intervals on 2D array-like using z=1.96
    :param data: data to compute the confidence interval on
    :param idx_min: lower bound index if data needs parsing
    :param idx_max: upper bound index if data needs parsing
    :return: upper and lower bound arrays, matching the second dimension of the (parsed) array
    """
    std = np.std(data, axis=0)
    ci = 1.96 * std / np.sqrt(len(data))
    m = np.mean(data, axis=0)
    lower = m - ci
    upper = m + ci
    if idx_min is None:
        idx_min = 0
    if idx_max is None:
        idx_max = len(m)
    return lower[idx_min:idx_max], upper[idx_min:idx_max]


def get_std_bounds(data, idx_min=None, idx_max=None):
    """
    Helper function to compute the standard deviation on 2D array-like
    :param data: data to compute the confidence interval on
    :param idx_min: lower bound index if data needs parsing
    :param idx_max: upper bound index if data needs parsing
    :return: upper and lower bound arrays, matching the second dimension of the (parsed) array
    """
    std = np.std(data, axis=0)
    m = np.mean(data, axis=0)
    lower = m - std
    upper = m + std
    if idx_min is None:
        idx_min = 0
    if idx_max is None:
        idx_max = len(m)
    return lower[idx_min:idx_max], upper[idx_min:idx_max]


def get_sem_bounds(data, idx_min=None, idx_max=None):
    """
    Helper function to compute the standard error of mean on 2D array-like
    :param data: data to compute the confidence interval on
    :param idx_min: lower bound index if data needs parsing
    :param idx_max: upper bound index if data needs parsing
    :return: upper and lower bound arrays, matching the second dimension of the (parsed) array
    """
    sem = stats.sem(data, axis=0)
    m = np.mean(data, axis=0)
    lower = m - sem
    upper = m + sem
    if idx_min is None:
        idx_min = 0
    if idx_max is None:
        idx_max = len(m)
    return lower[idx_min:idx_max], upper[idx_min:idx_max]
