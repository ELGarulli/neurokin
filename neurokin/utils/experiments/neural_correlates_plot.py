import numpy as np
from scipy import stats


def plot_psd_single_state(ax, group_cond_dataset, freqs, idx_min, idx_max, color, stat, linewidth=3):
    f = freqs[idx_min:idx_max]
    mean_data = np.mean(group_cond_dataset, axis=0)[idx_min:idx_max]

    ax.plot(f, mean_data, color=color, linewidth=linewidth)
    add_shades(ax, f, group_cond_dataset, idx_min, idx_max, color, stat=stat)


def add_shades(ax, f, data, idx_min, idx_max, color, stat="std"):
    if stat == "sem":
        lower, upper = get_sem_bounds(data, idx_min, idx_max)
    elif stat == "ci":
        lower, upper = get_ci_bounds(data, idx_min, idx_max)
    else:
        lower, upper = get_std_bounds(data, idx_min, idx_max)
    ax.fill_between(f, lower, upper, alpha=0.3, color=color)


def get_ci_bounds(data, idx_min, idx_max):
    std = np.std(data, axis=0)
    ci = 1.96 * std / np.sqrt(len(data))
    m = np.mean(data, axis=0)
    lower = m - ci
    upper = m + ci
    return lower[idx_min:idx_max], upper[idx_min:idx_max]


def get_std_bounds(data, idx_min, idx_max):
    std = np.std(data, axis=0)
    m = np.mean(data, axis=0)
    lower = m - std
    upper = m + std
    return lower[idx_min:idx_max], upper[idx_min:idx_max]


def get_sem_bounds(data, idx_min, idx_max):
    sem = stats.sem(data, axis=0)
    m = np.mean(data, axis=0)
    lower = m - sem
    upper = m + sem
    return lower[idx_min:idx_max], upper[idx_min:idx_max]
