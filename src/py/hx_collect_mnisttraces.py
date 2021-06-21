#!python3
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import sys
import time
import torch
import yaml

import utils
import training
import datasets
import evaluation

if __name__ == '__main__':
    dataset_train = datasets.HicannMnist('train', late_at_inf=True)
    dataset_val = datasets.HicannMnist('val', late_at_inf=True)
    dataset_test = datasets.HicannMnist('test', late_at_inf=True)

    if len(sys.argv) < 3:
        raise IOError("arguments that are needed in order: dirname, epoch to start at, [optional instance of a digit]")

    filename = '16x16_mnist'
    n_classes = 10
    n_repetitions = 10
    n_instance = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    single_sim_time = 10
    time_shown = 50

    dirname = sys.argv[1]
    start_epoch = int(sys.argv[2])
    savepoints = [start_epoch, ]
    net = training.continue_training(dirname, filename, start_epoch, savepoints,
                                     dataset_train, dataset_val, dataset_test)

    net.hx_settings['single_simtime'] = single_sim_time

    print("### prepare data")
    dataset, neuron_params, training_params, network_layout = training.load_config(
        osp.join(dirname, f"epoch_{start_epoch}", "config.yaml"))
    training_params['batch_size_eval'] = 10000
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000, shuffle=False)
    for i, data in enumerate(loader_test):
        inputs, labels = data
    # get n_instance th instance of each class
    idxs = torch.tensor([np.where(labels == i)[0][n_instance] for i in range(10)])
    # idxs = torch.tensor(np.unique(labels.numpy(), return_index=True)[1])
    distinct_inputs = inputs[idxs]
    distinct_labels = labels[idxs]
    fn = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())

    print("### start recording")
    lst = {}
    for recorded in range(10):
        net.hx_record_neuron = 246 + recorded
        lst[recorded] = []
        for i in range(n_repetitions):
            outs = net(distinct_inputs)
            print(f"accuracy is {(1. * (torch.argmin(outs[0], axis=1) == distinct_labels)).mean()}")
            # print(distinct_inputs)
            # print(outs[0])
            times, volts = net.trace[:, 0] * 1e6, net.trace[:, 1]
            plt.plot(times, volts)
            plt.gca().set_title(f"neuron {net.hx_record_neuron}")
            plt.gcf().savefig("tmp_trace.png")
            plt.close(plt.gcf())
            lst[recorded].append((times, volts))
        lst[recorded] = np.array(lst[recorded])

    print("### start plotting")
    fig, axes = plt.subplots(5, 2, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        for recorded in range(len(lst)):
            for k, (times, volts) in enumerate(lst[recorded]):
                start = i * net.hx_settings["single_simtime"] * net.hx_settings["scale_times"] * 1e6
                mask = np.logical_and(times > start, times < start + time_shown)
                ax.plot(times[mask] - start, volts[mask], color=f"C{recorded}", lw=1,
                        alpha=1. if distinct_labels[i] == recorded else 0.3)
        for j, t in enumerate(distinct_inputs[i]):
            ax.axvline(t * net.hx_settings["taum"], ymax=0.1, color=f"C{j + 3}")

        ax.set_yticklabels([])
        ax.set_ylabel(f"{distinct_labels[i]}", fontweight='bold')
        ax.yaxis.set_label_coords(-0.04, 0.2)
        ax.yaxis.label.set_color(f"C{distinct_labels[i]}")

        # preview pic of input
        size = 0.06
        numberClasses = 10
        xl, yl, xh, yh = np.array(ax.get_position()).ravel()
        h = yh - yl
        y_start = yh - 0.035 - size * 0.5  # - 0.92 * h * ((i + 0.5) / numberClasses)
        x_start = xl - size * 0.5 - 0.03  # - 0.05 * ((i + 1) % 2
        ax1 = fig.add_axes([x_start, y_start, size, size])
        # ax1.axison = False
        ax1.imshow(distinct_inputs[i].reshape(16, 16),
                   cmap='gray')
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        # ax1.grid(b=True)
    fig.savefig(f"tmp_{fn}_trace.pdf")
    plt.close(fig)

    print("### save data")
    np.save(f"tmp_{fn}_traces.npy", lst)
    np.save(f"tmp_{fn}_inputs.npy", distinct_inputs)
    np.save(f"tmp_{fn}_labels.npy", distinct_labels)
    with open(f"tmp_{fn}_hxsetting.json", "w") as f:
        json.dump(net.hx_settings, f)

    values_taken = 700
    reshaped_trace = np.ones((n_classes, n_classes, n_repetitions, 2, values_taken)) * -300.
    for i_digit in range(n_classes):
        for i_recorded in range(n_classes):
            # for i_repetition, (times, volts) in enumerate(lst[recorded]):
            for i_repetition in range(n_repetitions):
                (times, volts) = lst[i_recorded][i_repetition]
                start = i_digit * net.hx_settings["single_simtime"] * net.hx_settings["scale_times"] * 1e6
                start_idx = np.argmax(times > start)
                remaining_samples = len(times[start_idx:start_idx + values_taken])
                reshaped_trace[i_digit, i_recorded, i_repetition, 0, :remaining_samples] = times[
                    start_idx:start_idx + values_taken] - start
                reshaped_trace[i_digit, i_recorded, i_repetition, 1, :remaining_samples] = volts[
                    start_idx:start_idx + values_taken]
    np.save(f"tmp_{fn}_othertrace.npy", reshaped_trace)
