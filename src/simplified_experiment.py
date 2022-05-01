#!python3
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from pprint import pprint
import sys
import time
import torch
import yaml

import training
import datasets
import evaluation
import utils


debug_plot = True

config_path = "../experiment_configs/bars_default.yaml"

dataset, neuron_params, network_layout, training_params = training.load_config(config_path)

multiply_input_layer = 1 if not training_params['use_hicannx'] else 5
dataset_train = datasets.BarsDataset(3, noise_level=0, multiply_input_layer=multiply_input_layer)
dataset_val = datasets.BarsDataset(3, noise_level=0, multiply_input_layer=multiply_input_layer)
dataset_test = datasets.BarsDataset(3, noise_level=0, multiply_input_layer=multiply_input_layer)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=training_params['batch_size'], shuffle=True)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=training_params.get('batch_size_eval', None), shuffle=False)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=training_params.get('batch_size_eval', None), shuffle=False)
filename = dataset

# prepare special hardware stuff
if training_params['use_hicannx']:
    assert os.environ.get('SLURM_HARDWARE_LICENSES') is not None
    with open('py/hx_settings.yaml') as f:
        hx_settings = yaml.load(f, Loader=yaml.SafeLoader)
    hx_setup_no = os.environ.get('SLURM_HARDWARE_LICENSES')
    if hx_setup_no not in hx_settings:
        raise OSError(f"Setup no {hx_setup_no} is not described in hx settings file, only {hx_settings.keys()}")
    print("Using hardware settings:")
    pprint(hx_settings[hx_setup_no])
    neuron_params = hx_settings[hx_setup_no]['neuron_params']

    network_layout['n_inputs'] = network_layout['n_inputs'] * multiply_input_layer
else:
    if os.environ.get('SLURM_HARDWARE_LICENSES') is not None:
        sys.exit("There are SLURM_HARDWARE_LICENSES available "
                 f"({os.environ.get('SLURM_HARDWARE_LICENSES')}), but 'use_hicannx' is False. \n"
                 "Either execute without hw resources, or set 'use_hicannx'")

config_name = osp.splitext(osp.basename(filename))
dirname = '{0}_{1:%Y-%m-%d_%H-%M-%S}'.format(config_name, datetime.datetime.now())
# net = training.train(training_params, network_layout, neuron_params,
#                      dataset_train, dataset_val, dataset_test, dirname, filename)

torch.manual_seed(training_params['torch_seed'])
np.random.seed(training_params['numpy_seed'])
device = torch.device('cpu')

# create sim params
sim_params = {k: training_params.get(k, False)
              for k in ['use_forward_integrator', 'resolution', 'sim_time',
                        'rounding_precision', 'use_hicannx', 'max_dw_norm',
                        'clip_weights_max']
              }
sim_params.update(neuron_params)

net = training.Net(network_layout, sim_params, device)

criterion = utils.LossFunction(network_layout['layer_sizes'][-1],
                               sim_params['tau_syn'], training_params['xi'],
                               training_params['alpha'], training_params['beta'], device)

if training_params['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=training_params['learning_rate'])
elif training_params['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=training_params['learning_rate'],
                                momentum=training_params['momentum'])
else:
    raise NotImplementedError(f"optimizer {training_params['optimizer']} not implemented")

scheduler = None

# define logging variables
weight_bumping_steps = []
progress_train_accuracy = np.full((training_params['epoch_number'] + 1), np.nan)
progress_val_accuracy = np.full((training_params['epoch_number'] + 1), np.nan)
progress_times_hidden = np.full(
    (training_params['epoch_number'] + 1, network_layout['layer_sizes'][0]), np.nan)
progress_times_label = np.full(
    (training_params['epoch_number'] + 1, network_layout['layer_sizes'][-1], network_layout['layer_sizes'][-1]),
    np.nan)
progress_times_label_std = np.full(
    (training_params['epoch_number'] + 1, network_layout['layer_sizes'][-1], network_layout['layer_sizes'][-1]),
    np.nan)


# first evaluation
def validate(net, loader):
    all_outputs, all_labels, all_hiddens, all_losses = [], [], [], []
    num_shown, num_correct = 0, 0
    with torch.no_grad():
        for (input_times, labels) in loader:
            outputs, hiddens = net(input_times)

            loss = criterion(outputs, labels) * len(labels)

            firsts = outputs.argmin(1)
            # set firsts to -1 so that they cannot be counted as correct
            nan_mask = torch.isnan(torch.gather(outputs, 1, firsts.view(-1, 1))).flatten()
            inf_mask = torch.isinf(torch.gather(outputs, 1, firsts.view(-1, 1))).flatten()
            firsts[nan_mask] = -1
            firsts[inf_mask] = -1
            num_correct += len(outputs[firsts == labels])
            num_shown += len(labels)

            all_outputs.append(outputs)
            all_hiddens.append(hiddens)
            all_labels.append(labels)
            all_losses.append(loss)
        loss = sum(all_losses) / float(num_shown)  # can't use simple mean because batches might have diff size
        accuracy = float(num_correct) / num_shown
        # flatten output and label lists
        outputs = torch.stack([item for sublist in all_outputs for item in sublist])
        hiddens = torch.stack([item for sublist in all_hiddens for item in sublist])
        labels = [item.item() for sublist in all_labels for item in sublist]
        return loss, accuracy, labels, outputs, hiddens

fig_accuracy, ax_accuracy = plt.subplots(1, 1)
fig_labeltimes, axes_labeltimes = plt.subplots(3, 1, sharex=True, sharey=True)
fig_labeltimes.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, 
                               wspace=0.4, hspace=0.4)
fig_accuracy.show()
fig_accuracy.canvas.draw()
fig_labeltimes.show()
fig_labeltimes.canvas.draw()
plt.ioff()


def evaluate_and_plot(epoch):
    # ### evaluation
    loss, accuracy, labels, outputs, hiddens = validate(net, loader_val)
    # generating mean times
    unique_labels, unique_inverses = np.unique(labels, return_inverse=True)
    for i, label in enumerate(unique_labels):
        times = outputs[unique_inverses == i]
        progress_times_label[epoch + 1, i] = times.mean(axis=0)
        progress_times_label_std[epoch + 1, i] = times.std(axis=0)

    # saving data
    progress_val_accuracy[epoch + 1] = accuracy

    if debug_plot:
        # ### plotting
        ax_accuracy.clear()
        ax_accuracy.plot(progress_val_accuracy * 100)
        ax_accuracy.set_xlim(0, training_params['epoch_number'] + 1)
        ax_accuracy.set_ylim(-5, 105)
        ax_accuracy.set_xlabel("epochs [1]")
        ax_accuracy.set_ylabel("accuracy [%]")
        fig_accuracy.canvas.draw()

        for i_pattern, ax in enumerate(axes_labeltimes):
            ax.clear()
            ax.plot(progress_times_label[:, i_pattern])
            for i_neuron in range(len(unique_labels)):
                ax.fill_between(
                    np.arange(training_params['epoch_number'] + 1),
                    progress_times_label[:, i_pattern, i_neuron] - progress_times_label_std[:, i_pattern, i_neuron],
                    progress_times_label[:, i_pattern, i_neuron] + progress_times_label_std[:, i_pattern, i_neuron],
                                alpha=0.4)
            ax.set_ylabel("label spike times \n[$\\tau_s$]")
            ax.set_title(f"{datasets.BarsDataset.class_names[i_pattern]}. patterns") 

        axes_labeltimes[-1].set_xlim(0, training_params['epoch_number'] + 1)
        axes_labeltimes[-1].set_ylim(0.1, 2)
        axes_labeltimes[-1].set_xlabel("epochs [1]")
        fig_labeltimes.canvas.draw()


bump_val = training_params['weight_bumping_value']
last_weights_bumped = -2  # means no bumping happened last time
last_learning_rate = 0  # for printing learning rate at beginning
# noisy_training = training_params.get('training_noise') not in (False, None)
assert training_params.get('training_noise') in (False, None)


# right before training loop plot evaluation of untrained network
evaluate_and_plot(-1)
# training loop
for epoch in range(training_params['epoch_number']):
    train_loss = []
    num_correct = 0
    num_shown = 0
    for j, data in enumerate(loader_train):
        input_times, labels = data
        # if not isinstance(inputs, torch.Tensor):
        #     inputs = torch.tensor(inputs, dtype=torch.float64)
        # if not inputs.dtype == torch.float64:
        #     inputs = inputs.double()
        # input_times = utils.to_device(inputs, device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        label_times, hidden_times = net(input_times)
        firsts = label_times.argmin(1)
        # Either do the backward pass or bump weights because spikes are missing
        last_weights_bumped, bump_val = training.check_bump_weights(
            net, hidden_times, label_times,
            training_params, epoch, j, bump_val, last_weights_bumped)
        if last_weights_bumped != -2:  # means bumping happened
            weight_bumping_steps.append(epoch * len(loader_train) + j)
        else:
            loss = criterion(label_times, labels)
            loss.backward()
            optimizer.step()
            # on hardware we need extra step to write weights
            train_loss.append(loss.item())
        net.write_weights_to_hicannx()

        # set inf and nan firsts to -1 so that they cannot be counted as correct
        nan_mask = torch.isnan(torch.gather(label_times, 1, firsts.view(-1, 1))).flatten()
        inf_mask = torch.isinf(torch.gather(label_times, 1, firsts.view(-1, 1))).flatten()
        firsts[nan_mask] = -1
        firsts[inf_mask] = -1
        num_correct += len(label_times[firsts == labels])
        num_shown += len(labels)
        progress_train_accuracy[epoch] = len(label_times[firsts == labels]) / len(labels)

    # end of epoch evaluation
    train_accuracy = num_correct / num_shown if num_shown > 0 else np.nan

    evaluate_and_plot(epoch)

with torch.no_grad():
    validate_loss, validate_accuracy, validate_outputs, validate_labels, _ = training.validation_step(
        net, criterion, loader_val, device)
print("train accuracy: {4:.3f}, validation accuracy: {1:.3f},"
      "trainings loss: {2:.5f}, validation loss: {3:.5f}".format(
          0, validate_accuracy,
          np.mean(train_loss) if len(train_loss) > 0 else np.NaN,
          validate_loss, train_accuracy),
      flush=True)

debug_plot = True
evaluate_and_plot(epoch)
