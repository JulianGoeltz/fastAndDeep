#!python3
import argparse
import datetime
import json
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import os
import os.path as osp
import re
import sys
import time
import torch
import yaml
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('..')
import training
import utils

json_filename = '/jenkins/results/p_jg_FastAndDeep/data.json'
m_train = 'x'
m_test = '+'
m_testInference = '3'
ms_testInference = 5
max_allowed_error = 10

device = utils.get_default_device()


def get_data(dirname, dataset, datatype, criterion):
    # load stuff
    outputs = np.load(osp.join(dirname, f"{dataset}_{datatype}_spiketimes.npy"))
    labels = np.load(osp.join(dirname, f"{dataset}_{datatype}_labels.npy"))

    # correct class according to defined loss
    selected_classes = criterion.select_classes(torch.tensor(outputs))
    correct = torch.eq(torch.tensor(labels), selected_classes.detach().cpu()).sum().numpy()
    acc = correct / len(selected_classes)

    return acc


def write_new_data():
    path = "../../experiment_results/lastrun/epoch_300"
    dataset, neuron_params, network_layout, training_params = training.load_config(osp.join(path, "config.yaml"))
    criterion = utils.GetLoss(training_params,
                              network_layout['layer_sizes'][-1],
                              neuron_params['tau_syn'], device)
    # find out stats of last run
    pattern = 'the accuracy is ([0-9.]*)'
    inference_accs = []
    with open('../../../inference.out', 'r') as f:
        for line in f:
            if re.search(pattern, line):
                inference_accs.append(float(re.findall(pattern, line)[0]))
    if len(inference_accs) == 0:
        print("# file print for debug")
        with open('../../../inference.out', 'r') as f:
            for line in f:
                print(line)
        raise IOError("did not find any printed accuracies in the file, see printed contents above")

    data = {
        'accuracy_test': get_data(path, dataset, 'test', criterion),
        'accuracy_train': get_data(path, dataset, 'train', criterion),
        'accuracy_test_inference': inference_accs,
    }

    BUILD_NUMBER = os.environ.get("BUILD_NUMBER", "0")
    with open(osp.join(path, f"{dataset}_hw_licences.txt")) as f:
        data['HX'] = f.read()
    data['STAGE_NAME'] = os.environ.get("STAGE_NAME", "")
    data['BUILD_NUMBER'] = BUILD_NUMBER
    data['dataset'] = dataset
    if not osp.isfile(json_filename):
        with open(json_filename, 'w+') as f:
            json.dump({}, f)

    with open(json_filename, 'r') as f:
        all_data = json.load(f)

    all_data[BUILD_NUMBER] = {"date": time.time()}
    all_data[BUILD_NUMBER].update(data)

    with open(json_filename, 'w') as f:
        json.dump(all_data, f)


def plot_summary():
    with open(json_filename, 'r') as f:
        all_data = json.load(f)

    # plot all the last runs
    parser = argparse.ArgumentParser()
    # all those are expected to be counted from the end
    parser.add_argument('--firstBuild', default=30, type=int)
    parser.add_argument('--lastBuild', default=0, type=int)
    parser.add_argument('--dataset', default='yin_yang', type=str)
    parser.add_argument('--setup', default='all', type=str)
    parser.add_argument("--nolegend", help="not plot legend",
                        default=False, action='store_true',)
    parser.add_argument("--reduced_xticks", help="not plot all xticks",
                        default=False, action='store_true',)
    parser.add_argument(
        '--filename',
        default="jenkinssummary_{dataset}.png",
        type=str)

    args = parser.parse_args()

    # getting correctly sorted subset of builds
    builds = np.array(sorted(
        [int(i) for i in all_data.keys() if (
            all_data[i]['dataset'] == args.dataset and
            (args.setup == 'all' or all_data[i]['HX'] == args.setup)
        )
        ]
    )[-args.firstBuild:])
    if args.lastBuild != 0:
        builds = builds[:-args.lastBuild]
    xvals = np.arange(len(builds))
    all_setups = list(np.unique([all_data[str(buildNo)]['HX'] for buildNo in builds]))
    for buildNo in builds:
        all_data[str(buildNo)]['error_train'] = 100 - 100 * all_data[str(buildNo)]['accuracy_train']
        all_data[str(buildNo)]['error_test'] = 100 - 100 * all_data[str(buildNo)]['accuracy_test']
        if 'accuracy_test_inference' in all_data[str(buildNo)]:
            all_data[str(buildNo)][
                'error_test_inference'
            ] = 100 - np.array(all_data[str(buildNo)]['accuracy_test_inference'])

    print(f"plotting {len(builds)} builds: {builds}")

    fig, ax = plt.subplots(1, 1, figsize=((6, 4.5)))
    # plotting
    if args.setup == 'all':
        for i, setup in enumerate(all_setups):
            indices = [all_data[str(buildNo)]['HX'] == setup for buildNo in builds]
            ax.plot(xvals[indices], [all_data[str(buildNo)]['error_train'] for buildNo in builds[indices]],
                    label="train set", ls='', marker=m_train, color=f"C{i}")
            ax.plot(xvals[indices], [all_data[str(buildNo)]['error_test'] for buildNo in builds[indices]],
                    label="test set", ls='', marker=m_test, color=f"C{i}")

            for j, buildNo in enumerate(builds):
                if (
                    all_data[str(buildNo)]['HX'] == setup and
                    'error_test_inference' in all_data[str(buildNo)] and
                    len(all_data[str(buildNo)]['error_test_inference']) > 0
                ):
                    ax.plot(xvals[j],
                            [all_data[str(buildNo)]['error_test_inference']],
                            label="test set inference", ls='', marker=m_testInference,
                            ms=ms_testInference, color=f"C{i}")

    else:
        ax.plot(xvals, [all_data[str(buildNo)]['error_train'] for buildNo in builds],
                label="train set", color='black', ls='', marker=m_train)
        ax.plot(xvals, [all_data[str(buildNo)]['error_test'] for buildNo in builds],
                label="test set", color='black', ls='', marker=m_test)
        for j, buildNo in enumerate(builds):
            if (
                'error_test_inference' in all_data[str(buildNo)] and
                len(all_data[str(buildNo)]['error_test_inference']) > 0
            ):
                ax.plot(xvals[j],
                        [all_data[str(buildNo)]['error_test_inference']],
                        label="test set inference", ls='', marker=m_testInference,
                        ms=ms_testInference, color=f"black")

    # formatting
    ax.set_yscale('log')
    ax.set_ylabel('error [%]')
    if args.setup == 'all':
        ax.set_title("train and test errors")
    else:
        ax.set_title(f"train and test errors (on {args.setup})")

    ax.axhline(1, color='grey', lw=0.5)
    ax.axhline(5, color='grey', lw=0.5)
    ax.axhline(10, color='grey', lw=0.5)

    # fail line
    ax.axhline(max_allowed_error, color='grey', ls=':')

    if not args.nolegend:
        plt.rcParams['legend.handlelength'] = 1
        plt.rcParams['legend.handleheight'] = 1.125

        legend1 = ax.legend(
            handles=[
                mlines.Line2D([], [], color='black', lw=0, marker=m_test, label='test'),
                mlines.Line2D([], [], color='black', lw=0, marker=m_train, label='train'),
                mlines.Line2D([], [], color='black', lw=0, marker=m_testInference,
                              label='test inference'),
                mlines.Line2D([], [], color='grey', lw=1, ls=':', marker='', label='max error for success'),
            ],
            loc='lower center',
            # fontsize='large',
            frameon=True, facecolor="lightgray")
        ax.add_artist(legend1)
        if args.setup == 'all':
            legend2 = ax.legend(
                handles=[
                    # mpatches.Patch(color=f"C{i}", label=setup)
                    mpatches.Patch(color=f"C{i}", label=setup)
                    for i, setup in enumerate(all_setups)
                ],
                loc='lower left',
                fontsize="small", frameon=True, facecolor="lightgray")
            ax.add_artist(legend2)

    ax.set_ylim(1.5, 25)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([], minor=True)
    ytickminors = list(range(3, 10)) + [15]
    ytickminorlabels = [3, 5, max_allowed_error, 15]  # [1, 5, 10, 30]
    ax.set_yticks(ytickminors, minor=True)
    ax.set_yticklabels([i if i in ytickminorlabels else "" for i in ytickminors], minor=True)
    ax.set_yticks([10])
    ax.set_yticklabels([10])

    if not args.reduced_xticks:
        ax.set_xticks(xvals)
        ax.set_xticklabels(
            ["#{} ({})".format(buildNo,
                               datetime.datetime.fromtimestamp(
                                   float(all_data[str(buildNo)]["date"])).strftime('%m-%d'),
                               )
             for buildNo in builds],
            rotation=-90, fontsize="small")
        if args.setup == 'all':
            for ticklabel, buildNo in zip(ax.get_xticklabels(), builds):
                index_of_setup = all_setups.index(all_data[str(buildNo)]['HX'])
                ticklabel.set_color(f"C{index_of_setup}")
    else:
        idcs = np.linspace(0, len(xvals) - 1, 10, dtype=int)
        ax.set_xticks(xvals[idcs])
        ax.set_xticklabels(
            ["#{}\n({})".format(buildNo,
                                datetime.datetime.fromtimestamp(
                                    float(all_data[str(buildNo)]["date"])).strftime('%m-%d'),
                                )
             for buildNo in builds[idcs]],
            # rotation=-90,
            fontsize="small")
    fig.tight_layout()  # rect=[0, 0.00, 1, 1.99])  # due to suptitle

    # saving
    fig.savefig(
        args.filename.format(dataset=args.dataset)
    )


if __name__ == '__main__':
    # ###################################
    write_new_data()
    # ###################################
    plot_summary()
