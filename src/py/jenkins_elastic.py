#!python3
import argparse
import datetime
import json
import matplotlib as mpl
import numpy as np
import os
import os.path as osp
import sys
import time
import yaml
mpl.use('Agg')
import matplotlib.pyplot as plt


json_filename = '/wang/data/bss2-testing/fast_and_deep/data.json'


def get_data(dirname, dataset, datatype):
    # load stuff
    outputs = np.load(osp.join(dirname, f"{dataset}_{datatype}_spiketimes.npy"))
    labels = np.load(osp.join(dirname, f"{dataset}_{datatype}_labels.npy"))
    # inputs = np.load(osp.join(dirname, f"{dataset}_{datatype}_inputs.npy"))
    num_classes = np.max(labels) + 1
    firsts = np.argmin(outputs, axis=1)

    # set firsts to -1 so that they cannot be counted as correct
    firsts[np.isnan(outputs[np.eye(num_classes, dtype=bool)[firsts]])] = -1
    firsts[np.isinf(outputs[np.eye(num_classes, dtype=bool)[firsts]])] = -1

    return np.mean(labels == firsts)


if __name__ == '__main__':
    # ###################################
    path = "../../experiment_results/lastrun/epoch_300"
    with open(osp.join(path, "config.yaml")) as f:
        dataset = yaml.safe_load(f)['dataset']
    # find out stats of last run
    data = {
        'accuracy_test': get_data(path, dataset, 'test'),
        'accuracy_train': get_data(path, dataset, 'train'),
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

    # ###################################
    # plot all the last runs
    parser = argparse.ArgumentParser()
    # all those are expected to be counted from the end
    parser.add_argument('--firstBuild', default=20, type=int)
    parser.add_argument('--lastBuild', default=0, type=int)
    parser.add_argument('--dataset', default='yin_yang', type=str)
    args = parser.parse_args()

    # getting correctly sorted subset of builds
    builds = sorted(
        [int(i) for i in all_data.keys() if all_data[i]['dataset'] == args.dataset]
    )[-args.firstBuild:]
    if args.lastBuild != 0:
        builds = builds[:-args.lastBuild]
    xvals = range(len(builds))

    print(f"plotting {len(builds)} builds: {builds}")

    fig, ax = plt.subplots(1, 1, figsize=((6, 4.5)))
    # plotting
    ax.plot(xvals, [100 - 100 * all_data[str(buildNo)]['accuracy_train'] for buildNo in builds],
            label="train set", ls='', marker='x')
    ax.plot(xvals, [100 - 100 * all_data[str(buildNo)]['accuracy_test'] for buildNo in builds],
            label="test set", ls='', marker='x')

    # formatting
    ax.set_yscale('log')
    ax.set_ylabel('error [%]')
    ax.set_title("train and test accuracies")
    ax.axhline(1, color='black')
    ax.axhline(5, color='black')
    ax.axhline(30, color='black')
    ax.legend()
    ax.set_yticks([1, 5, 10, 30])
    ax.set_yticklabels([1, 5, 10, 30])
    ax.set_xticks(xvals)
    ax.set_xticklabels(
        ["#{} ({})".format(buildNo,
                           datetime.datetime.fromtimestamp(
                               float(all_data[str(buildNo)]["date"])).strftime('%d-%m'))
         for buildNo in builds],
        rotation=-90)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # due to suptitle

    # saving
    fig.savefig(f"jenkinssummary_{dataset}.png")
