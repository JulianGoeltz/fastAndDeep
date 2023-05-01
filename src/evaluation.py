#!python3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs
import numpy as np
import os
import os.path as osp
import time
import torch
import yaml

import training
import utils


def run_inference(dirname, filename, datatype, dataset, untrained, reference, device=None,
                  return_inputs=False, return_hidden=False, wholeset=False, net=None):
    if device is None:
        device = torch.device('cpu')
    if untrained:
        basename = filename + '_untrained'
    else:
        basename = filename
    _, neuron_params, network_layout, training_params = training.load_config(osp.join(dirname, "config.yaml"))
    criterion = utils.GetLoss(training_params,
                              network_layout['layer_sizes'][-1],
                              neuron_params['tau_syn'], device)

    saved_spikes_exist = os.path.exists(dirname + '/' + filename + '_{}_spiketimes.npy'.format(datatype))
    saved_inputs_exist = os.path.exists(dirname + '/' + filename + '_{}_inputs.npy'.format(datatype))

    if not return_hidden:
        if saved_spikes_exist and (not return_inputs or saved_inputs_exist):
            print('### Using pre-saved spiketimes for faster plot ###')
            outputs = torch.tensor(training.load_data(dirname, filename, '_{}_spiketimes.npy'.format(datatype)))
            labels = torch.tensor(training.load_data(dirname, filename, '_{}_labels.npy'.format(datatype)))
            selected_classes = criterion.select_classes(outputs)
            if return_inputs:
                inputs = torch.tensor(training.load_data(dirname, filename, '_{}_inputs.npy'.format(datatype)))
            else:
                inputs = None
            return outputs, selected_classes, labels, None, inputs

    print('### Running in inference mode ###')

    # print(training_params)
    if not reference:
        if training_params['use_hicannx'] and \
           os.environ.get('SLURM_HARDWARE_LICENSES') is None:
            print("#### to evaluate epochs on HX, execute on hardware (with 'srun --partition cube --wafer ...')")
            return (None, ) * (2 if not return_all else 4)
    if wholeset:
        batch_size = len(dataset)
    else:
        batch_size = training_params.get('batch_size_eval', len(dataset))

    loader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                         batch_size=batch_size)

    if net is None:
        if not untrained:
            net = utils.network_load(dirname, basename, device)
        else:
            if osp.isfile(dirname + "/" + basename + '_network.pt'):
                net = utils.network_load(dirname, basename, device)
            elif osp.isfile(dirname + '/../' + basename + '_network.pt'):
                net = utils.network_load(dirname + '/../', basename, device)
            else:
                raise IOError(f"No untrained network '{basename + '_network.pt'}' found in {dirname} or {dirname}/..")

    if training_params.get('use_forward_integrator', False):
        for layer in net.layers:
            layer.use_forward_integrator = True
            assert 'resolution' in training_params and 'sim_time' in training_params
            layer.sim_params['resolution'] = training_params['resolution']
            layer.sim_params['steps'] = int(np.ceil(training_params['sim_time'] / training_params['resolution']))
            print(layer.sim_params['steps'])
            layer.sim_params['decay_syn'] = float(np.exp(-training_params['resolution'] / neuron_params['tau_syn']))
            layer.sim_params['decay_mem'] = float(np.exp(-training_params['resolution'] / neuron_params['tau_mem']))
    # might use different device for analysis than training
    for i, bias in enumerate(net.biases):
        net.biases[i] = utils.to_device(bias, device)
    for layer in net.layers:
        layer.device = device

    with torch.no_grad():
        all_outputs = []
        all_labels = []
        all_inputs = []
        all_hiddens = []
        for i, data in enumerate(loader):
            inputs, labels = data
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float64)
            if not inputs.dtype == torch.float64:
                inputs = inputs.double()
            input_times = utils.to_device(inputs, device)
            outputs, hiddens = net(input_times)
            all_outputs.append(outputs)
            all_labels.append(labels)
            all_inputs.append(inputs)
            all_hiddens.append(hiddens)
            if 'mnist' in filename:
                print(f"\rinference ongoing, at {i} of {len(loader)} batches", end='')
        outputs = torch.stack([item for sublist in all_outputs for item in sublist])
        labels = np.array([item.item() for sublist in all_labels for item in sublist])
        inputs = torch.stack([item for sublist in all_inputs for item in sublist])
        hiddens = torch.stack([item for sublist in all_hiddens for item in sublist[0]])
        selected_classes = criterion.select_classes(outputs)

    print()
    return outputs, selected_classes, labels, hiddens, inputs


def plot_yyshape(ax, fillcolors=None):
    circle_centre_y = 0.15 + (2 - 0.15) / 2.
    circle_centre_x = 0.15 + np.array([1. / 4., 3. / 4.]) * (2 - 0.15)
    r_big = 1. * (2 - 0.15) / 2.
    r_small = 0.2 * (2 - 0.15) / 2.
    ec = "black"
    lw = 3
    full_circle = np.linspace(0, 2 * np.pi, 100)
    half_circle = np.linspace(0, np.pi, 100)

    if fillcolors is not None:
        ax.add_artist(mpatches.Wedge((circle_centre_x.mean(), circle_centre_y), r_big, 0, 180,
                                     color=fillcolors[1], zorder=-1))
        ax.add_artist(mpatches.Wedge((circle_centre_x.mean(), circle_centre_y), r_big, 180, 0,
                                     color=fillcolors[0], zorder=-1))

        ax.add_artist(mpatches.Wedge((circle_centre_x[0], circle_centre_y), r_big / 2., 180, 0,
                                     color=fillcolors[1], zorder=-1))
        ax.add_artist(mpatches.Wedge((circle_centre_x[1], circle_centre_y), r_big / 2., 0, 180,
                                     color=fillcolors[0], zorder=-1))

        ax.add_artist(plt.Circle((circle_centre_x[0], circle_centre_y), r_small,
                                 color=fillcolors[2], zorder=-1))
        ax.add_artist(plt.Circle((circle_centre_x[1], circle_centre_y), r_small,
                                 color=fillcolors[2], zorder=-1))

    ax.plot(circle_centre_x.mean() + r_big * np.cos(full_circle), circle_centre_y + r_big * np.sin(full_circle),
            color=ec, zorder=10, lw=lw)

    ax.plot(circle_centre_x[0] + r_small * np.cos(full_circle), circle_centre_y + r_small * np.sin(full_circle),
            color=ec, zorder=10, lw=lw)
    ax.plot(circle_centre_x[1] + r_small * np.cos(full_circle), circle_centre_y + r_small * np.sin(full_circle),
            color=ec, zorder=10, lw=lw)

    ax.plot(circle_centre_x[0] + r_big / 2. * np.cos(half_circle), circle_centre_y - r_big / 2. * np.sin(half_circle),
            color=ec, zorder=10, lw=lw)
    ax.plot(circle_centre_x[1] + r_big / 2. * np.cos(half_circle), circle_centre_y + r_big / 2. * np.sin(half_circle),
            color=ec, zorder=10, lw=lw)

    # use with plot_yy(ax, ["C0", "C1", "C2"])


def confusion_matrix(datatype, dataset, dirname='tmp', filename='', untrained=False, show=False, reference=False,
                     device=None, net=None):
    if device is None:
        device = torch.device('cpu')
    outputs, selected_classes, labels, _, _  = run_inference(dirname, filename, datatype, dataset,
                                                             untrained, reference, device, net=net)
    if outputs is None and labels is None:
        return
    # this can not run inference on hicannx currently
    num_labels = len(np.unique(labels))
    if reference:
        selected_classes = outputs.argmax(1)
    correct_per_pattern = np.zeros((num_labels, num_labels))
    for label in range(num_labels):
        idcs = (labels == label)
        classifications = selected_classes[idcs].reshape((-1, 1)) == torch.arange(num_labels).reshape((1, -1))
        correct_per_pattern[label] = classifications.sum(axis=0) / idcs.sum()
    accuracy = torch.eq(selected_classes, labels).detach().numpy().astype(float).mean()
    print(f"accuracy {accuracy}")
    if untrained:
        path = dirname + '/' + filename + '_confusion_matrix_{}_UNTRAINED.png'.format(datatype)
    else:
        path = dirname + '/' + filename + '_confusion_matrix_{}.png'.format(datatype)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(num_labels))
    ax.set_xticklabels(dataset.class_names)
    ax.set_yticks(np.arange(num_labels))
    ax.set_yticklabels(dataset.class_names)
    if untrained:
        plt.title('{0} dataset UNTRAINED (acc: {1})'.format(datatype, np.around(accuracy, 3)))
    else:
        plt.title('{0} dataset (acc: {1})'.format(datatype, np.around(accuracy, 3)))
    color_map = ax.imshow(correct_per_pattern)
    color_map.set_cmap("Blues_r")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    for (j, i), label in np.ndenumerate(correct_per_pattern):
        ax.text(i, j, np.around(label, decimals=2), ha='center', va='center')
    fig.colorbar(color_map)
    plt.savefig(path)

    if show:
        plt.show()
    plt.close(fig)
    return


def sorted_outputs(datatype, dataset, dirname='tmp', filename='', untrained=False, show=False, reference=False,
                   device=None, net=None):
    if device is None:
        device = torch.device('cpu')
    outputs, selected_classes, labels, _, _ = run_inference(dirname, filename, datatype, dataset,
                                                            untrained, reference, device, net=net)
    if outputs is None and labels is None:
        return
    num_labels = len(np.unique(labels))
    # sort for earliest spike
    outputs_sorted = [[] for i in range(num_labels)]
    for pattern in range(len(outputs)):
        true_label = labels[pattern]
        outputs_sorted[true_label].append(np.array(outputs[pattern].detach().cpu()))
    fig, axes = plt.subplots(num_labels, 1, sharex=True, figsize=(10, 10))
    if untrained:
        fig.suptitle('Output times for each class: {} dataset UNTRAINED'.format(datatype))
    else:
        fig.suptitle('Output times for each class: {} dataset'.format(datatype))
    for i in range(num_labels):
        to_plot = np.array(outputs_sorted[i])
        if reference:
            mins = np.max(to_plot, axis=1)
        else:
            mins = np.min(to_plot, axis=1)
        indices = np.argsort(mins)
        to_plot = to_plot[indices]
        for j in range(num_labels):
            if j == i:
                alpha = 1.
            else:
                alpha = 0.5
            xs = np.arange(len(to_plot[:, j]))
            ys = to_plot[:, j]
            axes[i].plot(xs, ys, label='neuron {}'.format(j), alpha=alpha)
            axes[i].legend()
            axes[i].set_ylim(0.2, 3.5)
            axes[i].set_xlabel('example (sorted by earliest spiketime)')
            axes[i].set_ylabel('spiketime')
    if untrained:
        path = dirname + '/' + filename + '_output_times_sorted_{}_UNTRAINED.png'.format(datatype)
    else:
        path = dirname + '/' + filename + '_output_times_sorted_{}.png'.format(datatype)
    plt.savefig(path)
    plt.close(fig)
    # sort for correct spiketime
    outputs_sorted = [[] for i in range(num_labels)]
    for pattern in range(len(outputs)):
        true_label = labels[pattern]
        outputs_sorted[true_label].append(np.array(outputs[pattern].detach().cpu()))
    fig, axes = plt.subplots(num_labels, 1, sharex=True, figsize=(10, 10))
    if untrained:
        fig.suptitle('Output times for each class: {} dataset UNTRAINED'.format(datatype))
    else:
        fig.suptitle('Output times for each class: {} dataset'.format(datatype))
    for i in range(num_labels):
        to_plot = np.array(outputs_sorted[i])
        indices = np.argsort(to_plot[:, i])

        for j in range(num_labels):
            if j == i:
                alpha = 1.
            else:
                alpha = 0.5
            xs = np.arange(len(to_plot[:, j]))
            ys = to_plot[:, j][indices]
            axes[i].plot(xs, ys, label='neuron {}'.format(j), alpha=alpha)
            axes[i].legend()
            axes[i].set_ylim(0.2, 3.5)
            axes[i].set_xlabel('example (sorted by correct spiketime)')
            axes[i].set_ylabel('spiketime')
    if untrained:
        path = dirname + '/' + filename + '_output_times_sorted2_{}_UNTRAINED.png'.format(datatype)
    else:
        path = dirname + '/' + filename + '_output_times_sorted2_{}.png'.format(datatype)
    plt.savefig(path)
    plt.close(fig)
    if show:
        plt.show()
    return


def loss_accuracy(title, dirname='tmp', filename='', show=False, reference=False):
    train_losses = training.load_data(dirname, filename, '_train_losses.npy')
    train_accuracy = training.load_data(dirname, filename, '_train_accuracies.npy')
    val_losses = training.load_data(dirname, filename, '_val_losses.npy')
    val_accuracy = training.load_data(dirname, filename, '_val_accuracies.npy')
    fig, axes = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})  # figsize=(12,12))
    fig.suptitle(title)

    axes[0].plot(train_accuracy, label=f"training (final {train_accuracy[-1]})")
    axes[0].plot(val_accuracy, label=f"validation (final {val_accuracy[-1]})")
    axes[0].legend()
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(-0.1, 1.1)
    # ax_2 = axes[0].twinx()
    axes[1].axhline(0.01, color="grey", alpha=0.3)
    axes[1].axhline(0.05, color="grey", alpha=0.3)
    axes[1].plot(range(1, len(train_losses) + 1), 1 - train_accuracy, color="C0", label="training")
    axes[1].plot(1 - val_accuracy, color="C1", label="validation")
    axes[1].set_ylabel('error')
    axes[1].set_yscale('log')

    axes[2].plot(train_losses, label='training set')
    axes[2].plot(val_losses, label='validation set')
    # print(train_losses)
    # print(val_losses)
    # axes[2].legend()
    axes[2].set_ylabel("loss")
    axes[2].set_yscale('log')
    # axes[1].set_ylim(8e-1, 9e-1)
    axes[2].set_xlabel('epoch')

    if title != "":
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # due to suptitle
    else:
        fig.tight_layout()

    path = dirname + '/' + filename + '_loss_accuracy.png'
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)


def weight_histograms(dirname='tmp', filename='', show=False, device=None):
    if device is None:
        device = torch.device('cpu')
    net = utils.network_load(dirname, filename, device)
    if net.use_hicannx and hasattr(net, '_ManagedConnection'):
        net._ManagedConnection.__exit__()
    if osp.isfile(dirname + "/" + filename + '_untrained_network.pt'):
        net_untrained = utils.network_load(dirname, filename + '_untrained', device)
    elif osp.isfile(dirname + '/../' + filename + '_untrained_network.pt'):
        net_untrained = utils.network_load(dirname + '/../', filename + '_untrained', device)
    else:
        print("*" * 30)
        print(f"No untrained network '{filename}_untrained_network.pt' found in {dirname} or {dirname}/..")
        print("*" * 30)
        return
    # this can not run inference on hicannx currently
    num_layers = len(net.layers)
    fig, axes = plt.subplots(num_layers, 1, sharex=True, figsize=(10, 10))
    fig.suptitle('Weight histograms after training')
    for i in range(num_layers):
        weights = net.layers[i].weights.data.detach().cpu().numpy()
        weights_untrained = net_untrained.layers[i].weights.data.detach().cpu().numpy()
        axes[i].xaxis.set_tick_params(which='both', labelbottom=True)
        axes[i].hist(weights_untrained.flatten(), density=True, rwidth=0.9, label='initial',
                     alpha=0.4, color='C1')
        axes[i].hist(weights.flatten(), density=True, rwidth=0.9, label='layer {0}'.format(i),
                     alpha=0.7, color='C0')
        axes[i].legend()
        axes[i].set_xlabel('weights')
    path = dirname + '/' + filename + '_weight_hist.png'
    plt.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return


def weight_matrix(dirname='tmp', filename='', device=None):
    if device is None:
        device = torch.device('cpu')
    net = utils.network_load(dirname, filename, device)
    # this can not run inference on hicannx currently
    num_layers = len(net.layers)
    for i in range(num_layers):
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle('Weight matrix after training: layer {0}'.format(i))
        weights = net.layers[i].weights.data.detach().cpu().numpy()
        print(weights.shape)
        color_map = axes.imshow(weights, aspect='auto', origin='lower')
        color_map.set_cmap("RdBu")
        fig.colorbar(color_map)
        path = dirname + '/' + filename + '_weight_matrix_layer_{0}.png'.format(i)
        plt.savefig(path)
    plt.close(fig)
    return


def yin_yang_classification(datatype, dataset, dirname='tmp', filename='', reference=False, untrained=False,
                            show=False, device=None, net=None):
    def color_from_class(label):
        if label == 0:
            return 'C0'
        elif label == 1:
            return 'C3'
        elif label == 2:
            return 'C2'
        else:
            return 'pink'
    if device is None:
        device = torch.device('cpu')

    outputs, selected_classes, labels, _, inputs = run_inference(
        dirname, filename, datatype, dataset, untrained, reference,
        device, return_inputs=True, net=net)
    if outputs is None and labels is None:
        return
    if reference:
        selected_classes = outputs.argmax(1)

    colors = [color_from_class(sc) for sc in selected_classes]
    reduced_inputs = np.array([[item[0], item[1]] for item in inputs])
    wrongs = torch.logical_not(torch.eq(torch.tensor(labels), selected_classes.detach().cpu())).numpy()
    acc = (len(colors) - wrongs.sum()) / len(colors)
    print(f"accuracy {acc}")
    plt.figure(figsize=(10, 10))
    plt.title('Classification result {0} set (accuracy: {1})'.format(datatype, np.around(acc, 3)))
    plt.scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=colors, marker='o', s=150, edgecolor='black', alpha=0.7)
    plt.scatter(reduced_inputs[:, 0][wrongs], reduced_inputs[:, 1][wrongs], c='black',
                marker='x', s=130, lw=3, label='wrong class')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('input time 1')
    plt.ylabel('input time 2')
    plt.legend()
    if untrained:
        path = dirname + '/' + filename + '_classification_{}_UNTRAINED.png'.format(datatype)
    else:
        path = dirname + '/' + filename + '_classification_{}.png'.format(datatype)
    plt.savefig(path)
    plt.close(plt.gcf())


def yin_yang_spiketimes(datatype, dataset, dirname='tmp', filename='', reference=False, untrained=False, show=False,
                        device=None, net=None):
    if device is None:
        device = torch.device('cpu')

    outputs, selected_classes, labels, _, inputs = run_inference(
        dirname, filename, datatype, dataset,
        untrained, reference, device, return_inputs=True, net=net)
    if outputs is None and labels is None:
        return
    outputs = outputs.detach().cpu().numpy()
    reduced_inputs = np.array([[item[0], item[1]] for item in inputs])
    cm = plt.cm.get_cmap('RdYlBu')
    norm = plt.Normalize(0., 3.)
    fig, axes = plt.subplots(nrows=1, ncols=4, gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}, figsize=(20, 7))
    plt.suptitle('Spiketimes per label neuron for {} set'.format(datatype))
    sc = axes[0].scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=outputs[:, 0], cmap=cm, marker='o',
                         s=150, edgecolor='black', alpha=0.7, norm=norm)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel('input time 1')
    axes[0].set_ylabel('input time 2')
    axes[0].set_title('Neuron 0')
    sc = axes[1].scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=outputs[:, 1], cmap=cm, marker='o',
                         s=150, edgecolor='black', alpha=0.7, norm=norm)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlabel('input time 1')
    axes[1].set_title('Neuron 1')
    sc = axes[2].scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=outputs[:, 2], cmap=cm, marker='o',
                         s=150, edgecolor='black', alpha=0.7, norm=norm)
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].set_xlabel('input time 1')
    axes[2].set_title('Neuron 2')
    if untrained:
        path = dirname + '/' + filename + '_spiketimes_{}_UNTRAINED.png'.format(datatype)
    else:
        path = dirname + '/' + filename + '_spiketimes_{}.png'.format(datatype)
    cb = fig.colorbar(sc, cax=axes.flat[3])
    cb.set_label('Output time', rotation=90)
    plt.savefig(path)
    plt.close(fig)


def yin_yang_spiketime_diffs(datatype, dataset, dirname='tmp', filename='', reference=False, untrained=False,
                             show=False, device=None, net=None):
    if device is None:
        device = torch.device('cpu')

    outputs, selected_classes, labels, _, inputs = run_inference(
        dirname, filename, datatype, dataset, untrained, reference,
        device, return_inputs=True, net=net)
    if outputs is None and labels is None:
        return
    outputs = outputs.detach().cpu().numpy()
    output_diffs = []
    firsts = []
    for sample in outputs:
        min_time = np.min(sample)
        diffs = sample - min_time
        output_diffs.append(diffs)
        firsts.append(min_time)
    output_diffs = np.array(output_diffs)
    reduced_inputs = np.array([[item[0], item[1]] for item in inputs])
    cm = plt.cm.get_cmap('RdYlBu')
    norm = plt.Normalize(0., 2.1)
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(22, 5.5),
                             gridspec_kw={"width_ratios": [0.98, 0.05, 0.04, 0.98, 0.98, 0.98, 0.05]})
    plt.suptitle('Spiketimes - earliest spike per label neuron for {} set'.format(datatype))
    sc = axes[0].scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=firsts, cmap=cm, marker='o',
                         s=150, edgecolor='black', alpha=0.7, norm=norm)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel('input time 1')
    axes[0].set_ylabel('input time 2')
    axes[0].set_title('Earliest spiketime')
    cb = fig.colorbar(sc, cax=axes.flat[1])
    cb.set_label('Earliest spike', rotation=90)
    axes[2].axis('off')
    cm = plt.cm.get_cmap('RdYlBu')
    sc = axes[3].scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=output_diffs[:, 0], cmap=cm, marker='o',
                         s=150, edgecolor='black', alpha=0.7, norm=norm)
    axes[3].set_aspect('equal', adjustable='box')
    axes[3].set_xlabel('input time 1')
    axes[3].set_ylabel('input time 2')
    axes[3].set_title('Neuron 0')
    sc = axes[4].scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=output_diffs[:, 1], cmap=cm, marker='o',
                         s=150, edgecolor='black', alpha=0.7, norm=norm)
    axes[4].set_aspect('equal', adjustable='box')
    axes[4].set_xlabel('input time 1')
    axes[4].set_title('Neuron 1')
    sc = axes[5].scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=output_diffs[:, 2], cmap=cm, marker='o',
                         s=150, edgecolor='black', alpha=0.7, norm=norm)
    axes[5].set_aspect('equal', adjustable='box')
    axes[5].set_xlabel('input time 1')
    axes[5].set_title('Neuron 2')
    if untrained:
        path = dirname + '/' + filename + '_spiketime_diffs_{}_UNTRAINED.png'.format(datatype)
    else:
        path = dirname + '/' + filename + '_spiketime_diffs_{}.png'.format(datatype)
    cb = fig.colorbar(sc, cax=axes.flat[6])
    cb.set_label('Output time - earliest spike', rotation=90)
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def yin_yang_hiddentimes(datatype, dataset, dirname='tmp', filename='', reference=False, untrained=False, show=False,
                         device=None, net=None):
    if device is None:
        device = torch.device('cpu')
    print('### Running in inference mode for yin_yang hidden times plot ###')
    outputs, selected_classes, labels, hiddens, inputs = run_inference(
        dirname, filename, datatype, dataset, untrained, reference,
        device, return_hidden=True, return_inputs=True, net=net)
    if outputs is None and labels is None:
        return
    hiddens = hiddens.detach().cpu().numpy()
    reduced_inputs = np.array([[item[0], item[1]] for item in inputs])
    cm = plt.cm.get_cmap('RdYlBu')
    norm = plt.Normalize(0., 3.)
    gridsize = 6
    fig, axes = plt.subplots(nrows=gridsize, ncols=gridsize, figsize=(22, 18))
    plt.suptitle('Spiketimes per hidden neuron for {} set'.format(datatype))
    for i in range(gridsize):
        for j in range(gridsize):
            k = i * gridsize + j
            if k >= hiddens.shape[1] or np.all(np.isinf(hiddens[:, k])):
                continue
            sc = axes[i, j].scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], c=hiddens[:, k], cmap=cm, marker='o',
                                    s=100, edgecolor='black', alpha=0.7, norm=norm)
            axes[i, j].set_xlim(0, 2.1)
            axes[i, j].set_ylim(0, 2.1)
            axes[i, j].set_aspect('equal', adjustable='box')
            if i == 4:
                axes[i, j].set_xlabel('input time 1')
            if j == 0:
                axes[i, j].set_ylabel('input time 2')
            axes[i, j].set_title('Hidden neuron {0}'.format(k))
    if untrained:
        path = dirname + '/' + filename + '_hiddentimes_{}_UNTRAINED.png'.format(datatype)
    else:
        path = dirname + '/' + filename + '_hiddentimes_{}.png'.format(datatype)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cb = fig.colorbar(sc, cax=cbar_ax)
    cb.set_label('Output time', rotation=90)
    plt.savefig(path)
    plt.close(fig)


def summary_plot(title, dirname='tmp', filename='', show=False, reference=False, device=None, net=None):
    if device is None:
        device = torch.device('cpu')
    if net is None:
        net = utils.network_load(dirname, filename, device)

    train_losses = training.load_data(dirname, filename, '_train_losses.npy')
    train_accuracy = training.load_data(dirname, filename, '_train_accuracies.npy')
    val_losses = training.load_data(dirname, filename, '_val_losses.npy')
    val_accuracy = training.load_data(dirname, filename, '_val_accuracies.npy')
    val_labels = training.load_data(dirname, filename, '_val_labels.npy')
    weights = training.load_data(dirname, filename, '_label_weights_training.npy')
    val_outputs_sorted = training.load_data(dirname, filename, '_mean_val_outputs_sorted.npy')
    std_outputs_sorted = training.load_data(dirname, filename, '_std_val_outputs_sorted.npy')
    num_labels = len(np.unique(val_labels))

    fig = plt.figure(figsize=(12, 12))
    gs_main = mpl_gs.GridSpec(1, 3,
                              # wspace=0.1
                              )

    hspace = 0.1
    gs_left = mpl_gs.GridSpecFromSubplotSpec(3, 1, gs_main[0, 0],
                                             hspace=hspace)
    ax = fig.add_subplot(gs_left[0, 0])
    ax.set_title(f"progress of {title}")
    ax.plot(range(1, len(train_accuracy) + 1, 1), train_accuracy,
            label='training (final: {})'.format(np.around(train_accuracy[-1], 3)))
    ax.plot(val_accuracy, label='validation (final: {})'.format(np.around(val_accuracy[-1], 3)))
    ax.legend()
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.1, 1.1)
    ax = fig.add_subplot(gs_left[1, 0])
    ax.plot(range(1, len(train_losses) + 1, 1), train_losses, label='training set')
    ax.plot(val_losses, label='validation set')
    ax.set_ylabel("loss")
    ax.set_yscale('log')
    ax = fig.add_subplot(gs_left[2, 0])
    ax.axhline(0.30, color="black", alpha=0.4)
    ax.axhline(0.05, color="black", alpha=0.4)
    ax.axhline(0.01, color="black", alpha=0.4)
    ax.plot(range(1, len(train_losses) + 1, 1), train_accuracy * (-1) + 1, label='training')
    ax.plot(val_accuracy * (-1) + 1, label='validation')
    ax.set_ylabel("error")
    ax.set_yscale('log')
    ax.set_xlabel('epoch')

    gs_centre = mpl_gs.GridSpecFromSubplotSpec(num_labels, 1, gs_main[0, 1],
                                               hspace=hspace)
    for i in range(num_labels):
        ax = fig.add_subplot(gs_centre[i, 0])
        if reference:
            to_plot = weights[:, i, :]
            for j in range(weights.shape[2]):
                ax.plot(to_plot[:, j], label='hidden {}'.format(j))
        else:
            to_plot = weights[:, :, i]
            for j in range(weights.shape[1]):
                ax.plot(to_plot[:, j], label='hidden {}'.format(j))
        if i == 0:
            ax.set_title("weights to {} outputs".format(num_labels))
        if i == num_labels - 1:
            ax.set_xlabel('epoch')

        if net.rounding:
            ax.axhline(0, color="black")
            ax.axhline(net.rounding_precision, color="black")

        if 'clip_weights_max' in net.sim_params:
            ax.axhline(net.sim_params['clip_weights_max'], color="black")
            ax.axhline(-net.sim_params['clip_weights_max'], color="black")

    gs_right = mpl_gs.GridSpecFromSubplotSpec(num_labels, 1, gs_main[0, 2],
                                              hspace=hspace)
    for i in range(num_labels):
        ax = fig.add_subplot(gs_right[i, 0])
        for j in range(num_labels):
            xs = np.arange(len(val_outputs_sorted[i][:, j]))
            ys = val_outputs_sorted[i][:, j]
            stds = std_outputs_sorted[i][:, j]
            ax.plot(xs, ys, label='neuron {}'.format(j))
            ax.fill_between(xs, ys - stds, ys + stds, alpha=0.4)
        # if i == 0:
        #     axes[i].legend()
        # axes[i].set_xlabel('epoch')
        # ax.set_ylim(0.9, 4.1)
        if i == 0:
            ax.set_title('output times for {} patterns'.format(num_labels))
        if i == num_labels - 1:
            ax.set_xlabel('epoch')

    fig.tight_layout()
    path = dirname + '/' + filename + '_summary_plot.png'
    fig.savefig(path)
    plt.close(fig)
    if show:
        plt.show()


def spiketime_hist(datatype, dataset, dirname='tmp', filename='', show=False, reference=False,
                   device=None, net=None):
    if reference:
        print('Plot for reference not implemented')
        return
    if device is None:
        device = torch.device('cpu')

    print("### running in inference mode for spiketime hist ###")
    outputs, selected_classes, labels, hiddens, _ = run_inference(
        dirname, filename, datatype, dataset, False,
        reference, device, return_hidden=True, net=net)
    if outputs is None and labels is None:
        return
    # sort into "should be first" and "should be late"
    should_early = []
    should_late = []
    no_spike_early = 0
    no_spike_late = 0
    num_labels = len(np.unique(labels))
    for i, output in enumerate(outputs):
        val = output[labels[i]].detach().cpu()
        if not torch.isinf(val):
            should_early.append(val)
        else:
            no_spike_early += 1
        for j in range(num_labels):
            if not j == labels[i]:
                val = output[j].detach().cpu()
                if not torch.isinf(val):
                    should_late.append(val)
                else:
                    no_spike_late += 1
    non_inf_mask = torch.logical_not(torch.isinf(hiddens))
    hiddens_no_inf = hiddens[non_inf_mask]
    no_spike_hidden = torch.sum(torch.isinf(hiddens))
    # get untrained data
    print("### running in inference mode for spiketime hist untrained ###")
    outputs_u, selected_classes, labels_u, hiddens_u, _ = run_inference(
        dirname, filename, datatype, dataset, True,
        reference, device, return_hidden=True)
    should_early_u = []
    should_late_u = []
    no_spike_early_u = 0
    no_spike_late_u = 0
    for i, output in enumerate(outputs_u):
        val = output[labels_u[i]].detach().cpu()
        if not torch.isinf(val):
            should_early_u.append(val)
        else:
            no_spike_early_u += 1
        for j in range(num_labels):
            if not j == labels_u[i]:
                val = output[j].detach().cpu()
                if not torch.isinf(val):
                    should_late_u.append(val)
                else:
                    no_spike_late_u += 1
    non_inf_mask = torch.logical_not(torch.isinf(hiddens_u))
    hiddens_no_inf_u = hiddens_u[non_inf_mask]
    no_spike_hidden_u = torch.sum(torch.isinf(hiddens_u))
    path = dirname + '/' + filename + '_spiketime_hist_{}.png'.format(datatype)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    axes[1][1].set_xlabel('spiketime')
    axes[0][1].set_ylabel('occurence')
    axes[1][1].set_ylabel('occurence')
    axes[0][1].xaxis.set_tick_params(labelbottom=True)
    axes[1][0].set_xlabel('spiketime')
    axes[0][0].set_ylabel('occurence')
    axes[1][0].set_ylabel('occurence')
    axes[0][0].xaxis.set_tick_params(labelbottom=True)
    axes[0][0].set_xlim(0, 4.5)
    axes[0][1].set_xlim(0, 4.5)
    axes[1][0].set_xlim(0, 4.5)
    axes[1][1].set_xlim(0, 4.5)

    bins = np.histogram(np.hstack((should_early, should_late)), bins=20)[1]
    dist_bins = bins[1] - bins[0]
    pos_no_spike = 4.0
    bins_no_spike = [pos_no_spike - dist_bins, pos_no_spike, pos_no_spike + dist_bins]
    axes[0][1].set_title('untrained')
    axes[0][1].hist(should_early_u, bins, alpha=0.7, rwidth=0.9, label='correct label neuron')
    axes[0][1].hist(should_late_u, bins, alpha=0.7, rwidth=0.9, label='wrong label neurons')
    axes[0][1].hist([pos_no_spike] * no_spike_early_u, bins_no_spike, alpha=0.7, rwidth=0.9, color='C0')
    axes[0][1].hist([pos_no_spike] * no_spike_late_u, bins_no_spike, alpha=0.7, rwidth=0.9, color='C1')
    axes[0][1].legend()
    axes[0][0].set_title('untrained')
    axes[0][0].hist(hiddens_no_inf_u.cpu().detach().numpy().flatten(), bins, alpha=0.7, rwidth=0.9,
                    color='grey', label='hidden neurons')
    axes[0][0].hist([pos_no_spike] * no_spike_hidden_u, bins_no_spike, alpha=0.7, rwidth=0.9, color='grey')
    # axes[2].legend()
    axes[0][0].legend()
    axes[1][1].set_title('trained')
    axes[1][1].hist(should_early, bins, alpha=0.7, rwidth=0.9, label='correct label neuron')
    axes[1][1].hist(should_late, bins, alpha=0.7, rwidth=0.9, label='wrong label neurons')
    axes[1][1].hist([pos_no_spike] * no_spike_early, bins_no_spike, alpha=0.7, rwidth=0.9, color='C0')
    axes[1][1].hist([pos_no_spike] * no_spike_late, bins_no_spike, alpha=0.7, rwidth=0.9, color='C1')
    axes[1][1].legend()
    axes[1][0].set_title('trained')
    axes[1][0].hist(hiddens_no_inf.cpu().detach().flatten(), bins, alpha=0.7, rwidth=0.9,
                    color='grey', label='hidden neurons')
    axes[1][0].hist([pos_no_spike] * no_spike_hidden, bins_no_spike, alpha=0.7, rwidth=0.9, color='grey')
    axes[1][0].legend()

    ticks = list(np.arange(0.5, 4.5, 0.5))
    tick_labels = [str(i) for i in ticks]
    tick_labels[-1] = 'no spike'
    for ax in axes.flatten():
        ax.xaxis.set_ticks(ticks)
        ax.set_xticklabels(tick_labels)

    fig.tight_layout()
    plt.savefig(path)

    if show:
        plt.show()
    plt.close(fig)
    return


def compare_voltages(dirname, filename, dataset, device=None, return_all=False, net=None):
    """new plot: comparing un-/trained label voltages to specific patterns"""
    if device is None:
        device = torch.device('cpu')
    _, neuron_params, network_layout, training_params = training.load_config(osp.join(dirname, "config.yaml"))
    assert not training_params.get('use_hicannx', False), "for now only do software membranes"
    assert net is None, "with loaded network we need to take care -> done later"
    loader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                         batch_size=training_params.get('batch_size_eval', len(dataset)))
    training_params['use_forward_integrator'] = True

    all_spikes, all_membranes = {}, {}
    for trained in [True, False]:
        if trained:
            basename = filename
        else:
            basename = filename + "_untrained"
        net = utils.network_load(dirname, basename, device)

        for layer in net.layers:
            layer.use_forward_integrator = True
            assert 'resolution' in training_params and 'sim_time' in training_params
            layer.sim_params['resolution'] = training_params['resolution']
            layer.sim_params['steps'] = int(np.ceil(training_params['sim_time'] / training_params['resolution']))
            print(layer.sim_params['steps'])
            layer.sim_params['decay_syn'] = float(np.exp(-training_params['resolution'] / neuron_params['tau_syn']))
            layer.sim_params['decay_mem'] = float(np.exp(-training_params['resolution'] / neuron_params['tau_syn']))
        # might use different device for analysis than training
        for i, bias in enumerate(net.biases):
            net.biases[i] = utils.to_device(bias, device)
        for layer in net.layers:
            layer.device = device

        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, labels = data
                input_times = utils.to_device(torch.tensor(inputs, dtype=torch.float64), device)
                outputs, hiddens = net(input_times)
                break
            np.save(osp.join(dirname, filename + '_membrane_points.npy'), inputs)
            np.save(osp.join(dirname, filename + '_membrane_labels.npy'), labels)

            assert osp.isfile("membrane_trace.npy") and osp.isfile("membrane_spike.npy"), \
                "make sure plotting is enabled in utils"
            all_spikes[trained] = np.load("membrane_spike.npy")
            all_membranes[trained] = np.load("membrane_trace.npy")
            np.save(
                osp.join(dirname, filename + f"_membrane_{'trained' if trained else 'untrained'}_spikes.npy"),
                all_spikes[trained])
            np.save(
                osp.join(dirname, filename + f"_membrane_{'trained' if trained else 'untrained'}_traces.npy"),
                all_membranes[trained])

            if filename == "yin_yang":
                fig, ax = plt.subplots(1, 1)
                mask = list(range(10))
                for i in range(10):
                    ax.scatter(inputs[mask[i], 0], inputs[mask[i], 1],
                               color=f"C{i}",
                               edgecolor="black",
                               label=f"C{i}",
                               s=60
                               )
                plot_yyshape(ax, [f"C{i}" for i in range(3)])
                ax.set_aspect('equal', adjustable='box')
                ax.legend()
                fig.savefig(osp.join(dirname, filename + '_membrane_points.png'))
                plt.close(fig)
    fig, axes = plt.subplots(10, 2, figsize=(10, 16), sharex=True, sharey=True)
    # print(all_membranes)
    for i in range(10):
        axes[i, 0].plot(np.arange(net.layers[0].sim_params['steps']) * training_params['resolution'],
                        all_membranes[False][:, i, :])
        axes[i, 1].plot(np.arange(net.layers[0].sim_params['steps']) * training_params['resolution'],
                        all_membranes[True][:, i, :])

        axes[i, 0].axhline(neuron_params['leak'], color='black', ls=":", lw=0.4, alpha=0.3)
        axes[i, 0].axhline(neuron_params['threshold'], color='black', ls=":", lw=1, alpha=0.3)
        axes[i, 1].axhline(neuron_params['leak'], color='black', ls=":", lw=0.4, alpha=0.3)
        axes[i, 1].axhline(neuron_params['threshold'], color='black', ls=":", lw=1, alpha=0.3)
        for j, sp in enumerate(all_spikes[False][i]):
            axes[i, 0].axvline(sp, color=f"C{j}", ls="-.", ymin=0.9)
        for j, sp in enumerate(all_spikes[True][i]):
            axes[i, 1].axvline(sp, color=f"C{j}", ls="-.", ymin=0.9)

        axes[i, 0].set_ylim(
            (neuron_params['threshold'] - neuron_params['leak']) * np.array((-1, 1.1)) + neuron_params['leak'])
        axes[i, 1].set_ylim(
            (neuron_params['threshold'] - neuron_params['leak']) * np.array((-1, 1.1)) + neuron_params['leak'])

        axes[i, 0].set_ylabel(f"correct label {labels[i]}", fontweight='bold')
        axes[i, 0].yaxis.label.set_color(f"C{labels[i]}")
        # ax.set_xlim(-0.5, 1.0)
    axes[0, 0].set_title("label membranes, untrained network")
    axes[0, 1].set_title("trained network")
    axes[-1, 0].set_xlabel("time [taus]")
    axes[-1, 1].set_xlabel("time [taus]")
    fig.tight_layout()
    fig.savefig(osp.join(dirname, filename + '_membrane_traces.png'))
    plt.close(fig)
