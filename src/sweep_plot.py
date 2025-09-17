# given a dirname, retrieve the list of updates and the list of result_dicts
import os
import os.path as osp

import matplotlib as mpl
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import argparse



W =  {'legend_name' : r'w only'              , 'color' : '#1f77b4', 'marker' : 'x', 'config_name': 'BroadcastLayer'}
WA = {'legend_name' : r'w & $d_\mathrm{axo}$', 'color' : '#ff7f0e', 'marker' : '*', 'config_name': 'DelayAxonal'}
WD = {'legend_name' : r'w & $d_\mathrm{den}$', 'color' : '#d62728', 'marker' : '.', 'config_name': 'DelayDendritic'}
WS = {'legend_name' : r'w & $d_\mathrm{syn}$', 'color' : '#9467bd', 'marker' : '2', 'config_name': 'DelaySynaptic'}

baseline_config = {"filename": "sweep_results_df.csv", "exp_name": "Baseline", "types": [W, WA, WD, WS], "linestyle": "-"}
err_range = (0.9, 15.0)
yticks = np.arange(int(err_range[0]), int(err_range[1]) + 1, 2)
metric_type = "median"

def get_categories_lists(df):
    categories = ['params', 'seed', 'metrics', 'metrics_epoch', 'parameters_epoch', 'output_epoch', 'metadata']
    params_list, seed_list, metrics_list, metrics_epoch_list, parameters_epoch_list, output_epoch_list, metadata_list = \
        ([col for col in df.columns if col.split('/')[0] == category] for category in categories)
    return params_list, seed_list, metrics_list, metrics_epoch_list, parameters_epoch_list, output_epoch_list, metadata_list

def plot_df(
    ax, df, param, metric, metric_type, sort_metric = 'final_val_err',
    plot_epoch = False, style='line', linestyle='-', color='b',
    label=None, marker = None, condition = lambda x:x,
    clip_on=True, skip_text=False, position=None
):
    assert metric_type in {'mean', 'median'}
    assert style in {'line', 'errorbar', 'barplot'}

    metric_full_name = f'{"metrics" if not plot_epoch else "metrics_epoch"}/{metric}_{metric_type}'
    assert metric_full_name in df.columns, f"Column {metric_full_name} not found in the dataframe"
    sort_metric_full_name = f'metrics/{sort_metric}_{metric_type}'
    assert sort_metric_full_name in df.columns, f"Column {sort_metric_full_name} not found in the dataframe"

    df = condition(df)
    df = df.sort_values(by=sort_metric_full_name)

    if plot_epoch :
        df = df.head(1)

        y = np.array(eval(df[metric_full_name].values[0]))
        x = np.arange(1, len(y) + 1)
        y_std = np.array(eval(df[f'metrics_epoch/{metric}_std'].values[0]))
        y_25 = np.array(eval(df[f'metrics_epoch/{metric}_25'].values[0]))
        y_75 = np.array(eval(df[f'metrics_epoch/{metric}_75'].values[0]))
        marker = None

    else:
        df = df.drop_duplicates(param, keep='first')
        df = df.sort_values(by=param)

        x = df[param].values
        y = df[metric_full_name].values
        y_std = df[f'metrics/{metric}_std'].values
        y_25 = df[f'metrics/{metric}_25'].values
        y_75 = df[f'metrics/{metric}_75'].values
        # y_0 = df[f'metrics/{metric}_0'].values
        # y_100 = df[f'metrics/{metric}_100'].values

    y_min = y - y_std if metric_type == 'mean' else y_25
    y_max = y + y_std if metric_type == 'mean' else y_75

    if style == 'line':
        ax.plot(x, 100*y, label=label, marker = marker, color=color, linestyle=linestyle, clip_on=clip_on, linewidth=1)
        ax.fill_between(x, 100*y_min, 100*y_max, alpha=0.3, color=color, clip_on=clip_on)
    elif style == 'errorbar':
        ax.errorbar(x, 100 * y, yerr=[100 * (y - y_min), 100 * (y_max - y)], fmt=marker, color=color, ecolor=color, elinewidth=2, capsize=3, label=label)


def plot_summary(
    ax, filename, param, metric, metric_type='mean', sort_metric='final_val_err',
    plot_epoch=False, style='line', linestyle='-', types=[W, WA, WD, WS],
    legend_prefix='', condition=lambda x: x, clip_on=True, skip_text=False):
    if isinstance(filename, pd.DataFrame):
        main_result_df = filename
    else:
        main_result_df = pd.read_csv(filename)
    for t in types:
        result_df = main_result_df[main_result_df.eq(t['config_name']).any(axis=1)]
        plot_df(ax, result_df, param, metric, metric_type, sort_metric= sort_metric,
                plot_epoch=plot_epoch, style=style, color=t['color'], label=legend_prefix+t['legend_name'],
                marker=t['marker'], condition=condition, linestyle=linestyle, clip_on=clip_on, skip_text=skip_text)


def plot_err_nhidden(ax, types = [W, WA, WD, WS], skip_ylabel=False):
    param_size = 'params/network_layout_layers_1_size'
    plot_summary(ax, filename=osp.join(base_path, baseline_config["filename"]), param=param_size,
                 metric='final_test_err', metric_type=metric_type, types=types, linestyle=baseline_config["linestyle"])

    ax.set_ylim(err_range)
    ax.set_yticks(yticks)
    ax.set_xlabel(r'# hidden nrns')
    if not skip_ylabel:
        ax.set_ylabel(r"test error [%]")
    else:
        ax.set_yticklabels([])
    ax.legend()
    ax.set_xlim(4, 31)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_err_nparams(ax, types = [W, WA, WD, WS], skip_ylabel=False):
    param_number = 'metrics/number_of_parameters_median'
    plot_summary(ax, filename=osp.join(base_path, baseline_config["filename"]), param=param_number,
                 metric='final_test_err', metric_type=metric_type, types=types, linestyle=baseline_config["linestyle"])

    ax.set_ylim(err_range)
    ax.set_yticks(yticks)
    ax.set_xlabel(r'# parameters')
    if not skip_ylabel:
        ax.set_ylabel(r"test error [%]")
    else:
        ax.set_yticklabels([])
    ax.legend()
    ax.set_xlim(20, 350)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



def plot_val_over_epoch(types = [W, WA, WD, WS], skip_ylabel=False):
    param_size = 'params/network_layout_layers_1_size'
    plot_summary(ax, filename=osp.join(base_path, baseline_config["filename"]), param=None, plot_epoch=True,
                 metric="val_err", metric_type = metric_type, style='line', linestyle='-', types = types,
                 legend_prefix ='', condition = lambda df: df[df[param_size] == 30])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("epochs")
    ax.set_ylabel(r"validation error [%]")

    ax.set_ylim(err_range)
    ax.set_yticks(yticks)
    ax.set_xlim(0, 300)
    ax.set_ylim(err_range)
    ax.legend()
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate sweep plots')
    parser.add_argument('--sweep_results_dirname', type=str, required=True,
                        help='Base path to the sweep results directory')
    parser.add_argument('--plot_err_by_epoch', action='store_true', help='When the error-by-epoch is included in the csv, plot error by epoch')
    args = parser.parse_args()
    
    base_path = args.sweep_results_dirname



    fig, ax = plt.subplots()
    plot_err_nhidden(ax, types = [W, WA], skip_ylabel=False)
    fig.show()
    fig.savefig(osp.join(base_path, 'nhidden_plot_axo.png'))

    fig, ax = plt.subplots()
    plot_err_nparams(ax, types = [W, WA], skip_ylabel=False)
    fig.show()
    fig.savefig(osp.join(base_path, 'nparams_plot_axo.png'))

    if args.plot_err_by_epoch:
        fig, ax = plt.subplots()
        plot_val_over_epoch(types = [W, WA], skip_ylabel=False)
        fig.show()
        fig.savefig(osp.join(base_path, 'val_over_epoch_axo.png'))

    plt.close(fig)
