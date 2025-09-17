#!python3
from collections import defaultdict
import copy
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
import torch
import yaml

import networks
import utils

torch.set_default_dtype(torch.float64)


def running_mean(x, N=30):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def get_hx_settings() -> dict:
    with open('py/hx_settings.yaml') as f:
        hx_settings = yaml.load(f, Loader=yaml.SafeLoader)
    hx_setup_no = os.environ.get('SLURM_HARDWARE_LICENSES')
    if hx_setup_no in hx_settings:
        return hx_settings[hx_setup_no]
    elif 'DEFAULT' in hx_settings:
        # adapt calibration path to default one
        hx_settings['DEFAULT']['calibration'] = f"calibrations/{hx_setup_no}.npz"
        hx_settings['DEFAULT']['_created_from_default'] = True
        return hx_settings['DEFAULT']
    else:
        raise OSError(f"DEFAULT not defined and setup no {hx_setup_no} is not described"
                      f"in hx settings file, only {hx_settings.keys()}")


def load_data(dirname, filename, dataname):
    path = dirname + '/' + filename + dataname
    data = np.load(path, allow_pickle=True)
    return data


def load_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data['dataset_params'], data['default_neuron_params'], data['network_layout'], data['training_params']


def validation_step(net, criterion, loader, device, return_input=False):
    all_outputs = []
    all_labels = []
    all_inputs = []
    with torch.no_grad():
        losses = []
        num_correct = 0
        num_shown = 0
        for j, data in enumerate(loader):
            inputs, labels = data
            input_times = utils.to_device(inputs.clone().type(torch.float64), device)
            outputs, _ = net(input_times)
            selected_classes = criterion.select_classes(outputs)
            num_correct += len(outputs[selected_classes == labels])
            num_shown += len(labels)
            loss = criterion(outputs, labels, net) * len(labels)
            losses.append(loss)
            all_outputs.append(outputs)
            all_labels.append(labels)
            if return_input:
                all_inputs.append(input_times)

        loss = sum(losses) / float(num_shown)  # can't use simple mean because batches might have diff size
        accuracy = float(num_correct) / num_shown
        # flatten output and label lists
        outputs = [item for sublist in all_outputs for item in sublist]
        labels = [item.item() for sublist in all_labels for item in sublist]
        if return_input:
            inputs = (torch.stack([item for sublist in all_inputs for item in sublist])).detach().cpu().numpy()
            return loss, accuracy, outputs, labels, inputs
        else:
            return loss, accuracy, outputs, labels, None


def check_bump_weights(net, hidden_times, label_times, training_params, epoch, batch, bump_val, last_weights_bumped):
    """determines if spikes were lost, adapts bump_val and bumps weights

    only foremost layer gets bumped: if in an earlier layer spikes are missing,
    chances are that in subsequent layers there will be too little input and missing
    spikes as well.
    return value weights_bumped:
        positive integer is hidden id that needed bump,
        -1: label layer needed bump
        -2: no bumping needed
    """
    weights_bumped = -2
    # first go through hidden times in loop, then output below
    for i, times in enumerate(hidden_times):
        if len(times) == 0:
            continue
        # we want mean over batches and neurons
        denominator = times.shape[0] * times.shape[1]
        non_spikes = torch.isinf(times) + torch.isnan(times)
        num_nonspikes = float(non_spikes.bool().sum())
        if num_nonspikes / denominator > net.layers_def[i]['max_num_missing_spikes']:
            weights_bumped = i
            break
    else:
        # else after for only executed if no break happened
        i = -1
        denominator = label_times.shape[0] * label_times.shape[1]
        non_spikes = torch.isinf(label_times) + torch.isnan(label_times)
        num_nonspikes = float(non_spikes.bool().sum())
        if num_nonspikes / denominator > net.layers_def[-1]['max_num_missing_spikes']:
            weights_bumped = -1
    if weights_bumped != -2:
        if training_params['weight_bumping_exp'] and weights_bumped == last_weights_bumped:
            bump_val *= 2
        else:
            bump_val = training_params['weight_bumping_value']

        # method to perform some operation on delays in case of weight bumping
        # in limited experiments this was not useful, so commented out
        # if isinstance(net.layers[weights_bumped - 1], utils.DelayLayer):
        #     with torch.no_grad():
        #         net.layers[weights_bumped - 1]._delay_parameters.data = (
        #             utils.sigmoid_inverse(
        #                 torch.sigmoid(net.layers[weights_bumped - 1]._delay_parameters) * 0.999)
        #         )
        if training_params['weight_bumping_targeted']:
            # make bool and then int to have either zero or ones
            should_bump = non_spikes.sum(axis=0).bool().int()
            n_in = net.layers[i].weights.data.size()[0]
            bumps = torch.full_like(net.layers[i].weights.data, bump_val)
            bumps[torch.logical_not(should_bump).repeat(n_in, 1)] = 0

            net.layers[i].weights.data += bumps
        else:
            net.layers[i].weights.data += bump_val

        print("epoch {0}, batch {1}: missing {4} spikes, bumping weights by {2} (targeted_bump={3})".format(
            epoch, batch, bump_val, training_params['weight_bumping_targeted'],
            "label" if weights_bumped == -1 else "hidden"))
    return weights_bumped, bump_val


def save_untrained_network(dirname, filename, net):
    if (dirname is None) or (filename is None):
        return

    try:
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    except FileExistsError:
        print("Directory ", dirname, " already exists")
    if not dirname[-1] == '/':
        dirname += '/'
    # save network
    if net.substrate == 'sim':
        torch.save(net, dirname + filename + '_untrained_network.pt')
    elif net.substrate == 'hx':
        tmp_backend = net.hx_backend
        tmp_MC = net._ManagedConnection
        tmp_connection = net._connection
        del net.hx_backend
        del net._ManagedConnection
        del net._connection
        torch.save(net, dirname + filename + '_untrained_network.pt')
        net.hx_backend = tmp_backend
        net._ManagedConnection = tmp_MC
        net._connection = tmp_connection
        # save hardware licence to identify the used hicann
        with open(dirname + filename + '_hw_licences.txt', 'w') as f:
            f.write(os.environ.get('SLURM_HARDWARE_LICENSES'))
        # save fpga bitfile info
        with open(dirname + filename + '_fpga_bitfile.yaml', 'w') as f:
            f.write(tmp_connection.bitfile_info)
        # save current calib settings
        shutil.copy(osp.join('py', 'hx_settings.yaml'), dirname + '/hw_settings.yaml')
    elif net.substrate == 'hx_pynn':
        tmp_network = net.network
        del net.network
        torch.save(net, dirname + filename + '_untrained_network.pt')
        net.network = tmp_network
        # save hardware licence to identify the used hicann
        with open(dirname + filename + '_hw_licences.txt', 'w') as f:
            f.write(os.environ.get('SLURM_HARDWARE_LICENSES'))
        # save fpga bitfile info
        with open(dirname + filename + '_fpga_bitfile.yaml', 'w') as f:
            f.write(list(net.network.pynn.simulator.state.conn.bitfile_info.values())[0])
        # save current calib settings
        shutil.copy(osp.join('py', 'hx_settings.yaml'), dirname + '/hw_settings.yaml')
    else:
        raise NotImplementedError()
    return


def save_config(dirname, filename, dataset_params, default_neuron_params,
                network_layout, training_params, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    if not osp.isdir(dirname):
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    # save parameter configs
    with open(osp.join(dirname, 'config.yaml'), 'w') as f:
        yaml.dump({"dataset_params": dataset_params, "default_neuron_params": default_neuron_params,
                   "network_layout": network_layout, "training_params": training_params}, f)
    with open(osp.join(dirname, filename + '_gitsha.txt'), 'w') as f:
        try:
            f.write(subprocess.check_output(["git", "rev-parse", "HEAD"]).decode())
        except subprocess.CalledProcessError:
            print("Not a git repository, can't save git sha")
        except FileNotFoundError:
            print("git probably not installed, install it")
    return


def save_result_dict(dirname, filename, net, result_dict, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    try:
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    except FileExistsError:
        print("Directory ", dirname, " already exists")
    # save network
    if net.substrate == 'sim':
        torch.save(net, dirname + filename + '_network.pt')
    elif net.substrate == 'hx':
        tmp_backend = net.hx_backend
        tmp_MC = net._ManagedConnection
        tmp_connection = net._connection
        del net.hx_backend
        del net._ManagedConnection
        del net._connection
        torch.save(net, dirname + filename + '_network.pt')
        net.hx_backend = tmp_backend
        net._ManagedConnection = tmp_MC
        net._connection = tmp_connection
        # save hardware licence to identify the used hicann
        with open(dirname + filename + '_hw_licences.txt', 'w') as f:
            f.write(os.environ.get('SLURM_HARDWARE_LICENSES'))
        # save fpga bitfile info
        with open(dirname + filename + '_fpga_bitfile.yaml', 'w') as f:
            f.write(tmp_connection.bitfile_info)
        # save current calib settings
        with open(dirname + '/hx_settings.yaml', 'w') as f:
            yaml.dump({os.environ.get('SLURM_HARDWARE_LICENSES'): net.hx_settings}, f)
    elif net.substrate == 'hx_pynn':
        tmp_network = net.network
        del net.network
        torch.save(net, dirname + filename + '_network.pt')
        net.network = tmp_network
        # save hardware licence to identify the used hicann
        with open(dirname + filename + '_hw_licences.txt', 'w') as f:
            f.write(os.environ.get('SLURM_HARDWARE_LICENSES'))
        # save fpga bitfile info
        with open(dirname + filename + '_fpga_bitfile.yaml', 'w') as f:
            f.write(list(net.network.pynn.simulator.state.conn.bitfile_info.values())[0])
        # save current calib settings
        with open(dirname + '/hx_settings.yaml', 'w') as f:
            yaml.dump({os.environ.get('SLURM_HARDWARE_LICENSES'): net.hx_settings}, f)
    else:
        raise NotImplementedError()

    # save training result
    np.save(dirname + filename + '_parameters_training.npy',
            {k: {'name': v['name'], 'params': np.array(v['params'])}
             for k, v in result_dict['all_parameters'].items()})
    np.save(dirname + filename + '_train_losses.npy', result_dict['all_train_loss'])
    np.save(dirname + filename + '_train_accuracies.npy', result_dict['all_train_accuracy'])
    np.save(dirname + filename + '_val_losses.npy', result_dict['all_validate_loss'])
    np.save(dirname + filename + '_val_accuracies.npy', result_dict['all_validate_accuracy'])
    np.save(dirname + filename + '_mean_val_outputs_sorted.npy', result_dict['mean_validate_outputs_sorted'])
    np.save(dirname + filename + '_std_val_outputs_sorted.npy', result_dict['std_validate_outputs_sorted'])
    np.save(dirname + filename + '_weight_bumping_steps.npy', result_dict['weight_bumping_steps'])
    return


def save_result_spikes(dirname, filename, train_times, train_labels, train_inputs,
                       test_times, test_labels, test_inputs, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    # stunt to avoid saving tensors
    train_times = np.array([item.detach().cpu().numpy() for item in train_times])
    test_times = np.array([item.detach().cpu().numpy() for item in test_times])
    np.save(dirname + filename + '_train_spiketimes.npy', train_times)
    np.save(dirname + filename + '_train_labels.npy', train_labels)
    if train_inputs is not None:
        np.save(dirname + filename + '_train_inputs.npy', train_inputs)
    np.save(dirname + filename + '_test_spiketimes.npy', test_times)
    np.save(dirname + filename + '_test_labels.npy', test_labels)
    if test_inputs is not None:
        np.save(dirname + filename + '_test_inputs.npy', test_inputs)
    return


def save_optim_state(dirname, filename, optimizer, scheduler, np_rand_state, torch_rand_state, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    with open(dirname + filename + '_optim_state.yaml', 'w') as f:
        yaml.dump([optimizer.state_dict(), scheduler.state_dict()], f)
    # if saving a snapshot, save state of rngs
    torch.save(torch_rand_state, dirname + filename + '_torch_rand_state.pt')
    numpy_dict = {'first': np_rand_state[0],
                  'second': np_rand_state[1],
                  'third': np_rand_state[2],
                  }
    with open(dirname + filename + '_numpy_rand_state.yaml', 'w') as f:
        yaml.dump([numpy_dict], f)
    return


def setup_lr_scheduling(params, optimizer):
    if params['type'] is None:
        return None
    elif params['type'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=params['step_size'],
                                               gamma=params['gamma'])
    elif params['type'] == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=params['milestones'],
                                                    gamma=params['gamma'])
    elif params['type'] == 'ExponentialLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    gamma=params['gamma'])
    else:
        raise IOError('WARNING: Chosen scheduler unknown. Use StepLR or MultiStepLR or ExponentialLR')


def load_optim_state(dirname, filename, net, training_params):
    path = dirname + '/' + filename + '_optim_state.yaml'
    with open(path) as f:
        data = yaml.load_all(f, Loader=yaml.Loader)
        all_configs = next(iter(data))
    if training_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            [{"params": layer.parameters(),
              "lr": layer.lr if layer.lr is not None else training_params['learning_rate']}
             for layer in net.layers
            ],
            lr=training_params['learning_rate']
        )
    else:
        optimizer = torch.optim.SGD(
            [{"params": layer.parameters(),
              "lr": layer.lr if layer.lr is not None else training_params['learning_rate']}
             for layer in net.layers
            ],
            lr=training_params['learning_rate'],
            momentum=training_params['momentum']
        )
    optim_state = all_configs[0]
    optimizer.load_state_dict(optim_state)
    scheduler = setup_lr_scheduling(training_params['lr_scheduler'], optimizer)
    schedule_state = all_configs[1]
    scheduler.load_state_dict(schedule_state)
    path = dirname + '/' + filename + '_numpy_rand_state.yaml'
    try:
        with open(path) as f:
            data = yaml.load_all(f, Loader=yaml.Loader)
            all_configs = next(iter(data))
        numpy_dict = all_configs[0]
        numpy_rand_state = (numpy_dict['first'], numpy_dict['second'], numpy_dict['third'])
    except IOError:
        numpy_rand_state = None
    path = dirname + '/' + filename + '_torch_rand_state.pt'
    try:
        torch_rand_state = torch.load(path)
    except IOError:
        torch_rand_state = None
    return optimizer, scheduler, torch_rand_state, numpy_rand_state

def load_result_dict(dirname_long, filename):
    result_dict = {}
    result_dict['all_train_loss'] = list(load_data(dirname_long, filename, '_train_losses.npy'))
    result_dict['all_train_accuracy'] = list(load_data(dirname_long, filename, '_train_accuracies.npy'))
    result_dict['all_validate_loss'] = list(load_data(dirname_long, filename, '_val_losses.npy'))
    result_dict['all_validate_accuracy'] = list(load_data(dirname_long, filename, '_val_accuracies.npy'))
    all_parameters = load_data(dirname_long, filename, '_parameters_training.npy').item()
    result_dict['all_parameters'] = {k: {'name': v['name'], 'params': list(v['params'])} for k, v in all_parameters.items()}

    mean_validate_outputs_sorted = list(load_data(dirname_long, filename, '_mean_val_outputs_sorted.npy'))
    result_dict['mean_validate_outputs_sorted'] = [list(item) for item in mean_validate_outputs_sorted]
    std_validate_outputs_sorted = list(load_data(dirname_long, filename, '_std_val_outputs_sorted.npy'))
    result_dict['std_validate_outputs_sorted'] = [list(item) for item in std_validate_outputs_sorted]

    result_dict['weight_bumping_steps'] = list(load_data(dirname_long, filename, '_weight_bumping_steps.npy'))

    return result_dict

def apply_noise(input_times, noise_params, device):
    shape = input_times.size()
    noise = utils.to_device(torch.zeros(shape), device)
    noise.normal_(noise_params['mean'], noise_params['std_dev'])
    input_times = input_times + noise
    negative = input_times < 0.
    input_times[negative] *= -1
    return input_times


def run_epochs(e_start, e_end, net, criterion, optimizer, scheduler, device, trainloader, valloader,
               num_classes, all_parameters, all_train_loss, all_validate_loss, std_validate_outputs_sorted,
               mean_validate_outputs_sorted, tmp_training_progress, all_validate_accuracy,
               all_train_accuracy, weight_bumping_steps, training_params):
    bump_val = training_params['weight_bumping_value']
    last_weights_bumped = -2  # means no bumping happened last time
    last_learning_rate = 0  # for printing learning rate at beginning
    noisy_training = training_params.get('training_noise') not in (False, None)
    print_step = max(1, int(training_params['epoch_number'] * training_params['print_step_percent'] / 100.))

    for epoch in range(e_start, e_end, 1):
        train_loss = []
        num_correct = 0
        num_shown = 0
        for j, data in enumerate(trainloader):
            inputs, labels = data
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float64)
            if not inputs.dtype == torch.float64:
                inputs = inputs.double()
            input_times = utils.to_device(inputs, device)
            if noisy_training:
                input_times = apply_noise(input_times, training_params['training_noise'], device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            label_times, hidden_times = net(input_times)
            selected_classes = criterion.select_classes(label_times)
            # Either do the backward pass or bump weights because spikes are missing
            last_weights_bumped, bump_val = check_bump_weights(net, hidden_times, label_times,
                                                               training_params, epoch, j, bump_val, last_weights_bumped)
            live_plot = True
            weight_bumping_steps.append(last_weights_bumped)
            if last_weights_bumped == -2:  # means no bumping happened
                loss = criterion(label_times, labels, net)
                loss.backward()
                optimizer.step()
                net.delays_rectify()
                # on hardware we need extra step to write weights
                train_loss.append(loss.item())
            net.write_weights()
            num_correct += len(label_times[selected_classes == labels])
            num_shown += len(labels)
            tmp_training_progress.append(len(label_times[selected_classes == labels]) / len(labels))

            if live_plot and j % 100 == 0:
                fig, ax = plt.subplots(1, 1)
                tmp = 1. - running_mean(tmp_training_progress, N=30)
                ax.plot(np.arange(len(tmp)) / len(trainloader), tmp)
                ax.set_ylim(0.005, 1.0)
                ax.set_yscale('log')
                ax.set_xlabel("epochs")
                ax.set_ylabel("error [%] (running_mean of 30 batches)")
                ax.axhline(0.30)
                ax.axhline(0.05)
                ax.axhline(0.01)
                ax.set_yticks([0.01, 0.05, 0.1, 0.3])
                ax.set_yticklabels([1, 5, 10, 30])
                fig.savefig(f'live_accuracy_{os.environ.get("SLURM_HARDWARE_LICENSES")}.png')
                print("===========Saved live accuracy plot")
                plt.close(fig)

        if len(train_loss) > 0:
            all_train_loss.append(np.mean(train_loss))
        else:
            all_train_loss.append(np.nan)
        train_accuracy = num_correct / num_shown if num_shown > 0 else np.nan
        all_train_accuracy.append(train_accuracy)
        if scheduler is not None:
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            if not (last_learning_rate == lr):
                print('setting learning_rate to {0:.5f}'.format(lr))
            last_learning_rate = lr

        # evaluate on validation set
        with torch.no_grad():
            validate_loss, validate_accuracy, validate_outputs, validate_labels, _ = validation_step(
                net, criterion, valloader, device)

            # TODO: if one rewrites this without the loops for speedup
            tmp_class_outputs = [[] for i in range(num_classes)]
            for pattern in range(len(validate_outputs)):
                true_label = validate_labels[pattern]
                tmp_class_outputs[true_label].append(validate_outputs[pattern].cpu().detach().numpy())
            for i in range(num_classes):
                tmp_times = np.array(tmp_class_outputs[i])
                tmp_times[np.isinf(tmp_times)] = np.nan
                mask_notAllNan = np.logical_not(np.isnan(tmp_times)).sum(0) > 0
                mean_times = np.ones(tmp_times.shape[1:]) * np.nan
                std_times = np.ones(tmp_times.shape[1:]) * np.nan
                mean_times[mask_notAllNan] = np.nanmean(tmp_times[:, mask_notAllNan], 0)
                std_times[mask_notAllNan] = np.nanstd(tmp_times[:, mask_notAllNan], 0)
                mean_validate_outputs_sorted[i].append(mean_times)
                std_validate_outputs_sorted[i].append(std_times)

            all_validate_accuracy.append(validate_accuracy)
            for i, layer in enumerate([l for l in net.layers if l.monitor]):
                if isinstance(layer, utils.DelayLayer):
                    tmp_data = layer.effective_delays()
                else:
                    tmp_data = next(layer.parameters())
                all_parameters[i]['params'].append(
                    tmp_data.data.cpu().detach().numpy().copy()
                )
            all_validate_loss.append(validate_loss.data.cpu().detach().numpy())

        if (epoch % print_step) == 0:
            print("... {0:.0f}% done, train accuracy: {4:.3f}, validation accuracy: {1:.3f},"
                  "trainings loss: {2:.5f}, validation loss: {3:.5f}".format(
                      epoch * 100 / training_params['epoch_number'], validate_accuracy,
                      np.mean(train_loss) if len(train_loss) > 0 else np.nan,
                      validate_loss, train_accuracy),
                  flush=True)

        result_dict = {'all_parameters': all_parameters,
                       'all_train_loss': all_train_loss,
                       'all_validate_loss': all_validate_loss,
                       'std_validate_outputs_sorted': std_validate_outputs_sorted,
                       'mean_validate_outputs_sorted': mean_validate_outputs_sorted,
                       'tmp_training_progress': tmp_training_progress,
                       'all_validate_accuracy': all_validate_accuracy,
                       'all_train_accuracy': all_train_accuracy,
                       'weight_bumping_steps': weight_bumping_steps,
                       # 'all_hidden_weights': all_hidden_weights,
                       # 'all_label_weights': all_label_weights,
                       }
    return net, criterion, optimizer, scheduler, result_dict


def train(training_params, dataset_params, network_layout, default_neuron_params,
          dataset_train, dataset_val, dataset_test,
          foldername='tmp', filename=''):
    if not training_params['torch_seed'] is None:
        torch.manual_seed(training_params['torch_seed'])
    if not training_params['numpy_seed'] is None:
        np.random.seed(training_params['numpy_seed'])
    if 'optimizer' not in training_params.keys():
        training_params['optimizer'] = 'sgd'
    if 'enforce_cpu' in training_params.keys() and training_params['enforce_cpu']:
        device = torch.device('cpu')
    else:
        device = utils.get_default_device()
    if not device == 'cpu':
        torch.cuda.manual_seed(training_params['torch_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # save parameter config
    save_config(foldername, filename, dataset_params, default_neuron_params,
                network_layout, training_params)

    # set default values if not given
    for k in ['use_forward_integrator', 'resolution', 'sim_time',
              'rounding_precision', 'substrate', 'max_dw_norm',
              'clip_weights_max']:
        if k not in training_params:
            training_params[k] = False

    print('training_params')
    print(training_params)
    print('network_layout')
    print(network_layout)
    print('using optimizer {0}'.format(training_params['optimizer']))

    # setup saving of snapshots
    savepoints = training_params.get('epoch_snapshots', [])
    if not training_params['epoch_number'] in savepoints:
        savepoints.append(training_params['epoch_number'])
    print('Saving snapshots at epochs {}'.format(savepoints))

    print("loading data")
    loader_train = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_train, batch_size=training_params['batch_size'], shuffle=True), device)
    loader_val = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_val, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)
    loader_test = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_test, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)

    print("generating network")
    net = networks.get_network(
        default_neuron_params, network_layout,
        training_params, device)
    save_untrained_network(foldername, filename, net)

    num_classes = network_layout['layers'][-1]['size']
    print("loss function")
    criterion = utils.GetLoss(training_params, 
                              num_classes,
                              default_neuron_params['tau_syn'], device)

    if training_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            [{"params": layer.parameters(),
              "lr": layer.lr if layer.lr is not None else training_params['learning_rate']}
             for layer in net.layers
            ],
            lr=training_params['learning_rate']
        )
    elif training_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            [{"params": layer.parameters(),
              "lr": layer.lr if layer.lr is not None else training_params['learning_rate']}
             for layer in net.layers
            ],
            lr=training_params['learning_rate'],
            momentum=training_params['momentum']
        )
    else:
        raise NotImplementedError(f"optimizer {training_params['optimizer']} not implemented")
    scheduler = None
    if 'lr_scheduler' in training_params.keys():
        scheduler = setup_lr_scheduling(training_params['lr_scheduler'], optimizer)

    # evaluate on validation set before training
    all_parameters = {}
    all_train_loss = []
    all_validate_loss = []
    std_validate_outputs_sorted = [[] for i in range(num_classes)]
    mean_validate_outputs_sorted = [[] for i in range(num_classes)]
    tmp_training_progress = []
    all_validate_accuracy = []
    all_train_accuracy = []
    weight_bumping_steps = []
    # all_hidden_weights = []
    # all_label_weights = []
    print("initial validation started")
    with torch.no_grad():
        loss, validate_accuracy, validate_outputs, validate_labels, _ = validation_step(
            net, criterion, loader_val, device)
        tmp_class_outputs = [[] for i in range(num_classes)]
        for pattern in range(len(validate_outputs)):
            true_label = validate_labels[pattern]
            tmp_class_outputs[true_label].append(validate_outputs[pattern].cpu().detach().numpy())
        for i in range(num_classes):
            tmp_times = np.array(tmp_class_outputs[i])
            inf_mask = np.isinf(tmp_times)
            tmp_times[inf_mask] = np.nan
            mean_times = np.nanmean(tmp_times, 0)
            std_times = np.nanstd(tmp_times, 0)
            mean_validate_outputs_sorted[i].append(mean_times)
            std_validate_outputs_sorted[i].append(std_times)
        print('Initial validation accuracy: {:.3f}'.format(validate_accuracy))
        print('Initial validation loss: {:.3f}'.format(loss))
        all_validate_accuracy.append(validate_accuracy)
        for i, layer in enumerate([l for l in net.layers if l.monitor]):
            if isinstance(layer, utils.DelayLayer):
                tmp_data = layer.effective_delays()
            else:
                tmp_data = next(layer.parameters())
            all_parameters[i] = {
                'name' : layer.__class__.__name__,
                'params' : [tmp_data.data.cpu().detach().numpy().copy()]
            }
        all_validate_loss.append(loss.data.cpu().detach().numpy())

    print("training started")
    for e_start, e_end in zip([0] + savepoints[:-1], savepoints):
        print('Starting training from epoch {0} to epoch {1}'.format(e_start, e_end))
        net, criterion, optimizer, scheduler, result_dict = run_epochs(
            e_start, e_end, net, criterion,
            optimizer, scheduler, device, loader_train,
            loader_val, num_classes, all_parameters,
            all_train_loss, all_validate_loss,
            std_validate_outputs_sorted,
            mean_validate_outputs_sorted,
            tmp_training_progress, all_validate_accuracy,
            all_train_accuracy, weight_bumping_steps,
            training_params)
        print('Ending training from epoch {0} to epoch {1}'.format(e_start, e_end))
        all_parameters = result_dict['all_parameters']
        all_train_loss = result_dict['all_train_loss']
        all_validate_loss = result_dict['all_validate_loss']
        std_validate_outputs_sorted = result_dict['std_validate_outputs_sorted']
        mean_validate_outputs_sorted = result_dict['mean_validate_outputs_sorted']
        all_validate_accuracy = result_dict['all_validate_accuracy']
        all_train_accuracy = result_dict['all_train_accuracy']
        weight_bumping_steps = result_dict['weight_bumping_steps']
        tmp_training_progress = result_dict['tmp_training_progress']
        # all_hidden_weights = result_dict['all_hidden_weights']
        # all_label_weights = result_dict['all_label_weights']
        save_result_dict(foldername, filename, net, result_dict, epoch_dir=(True, e_end))

        # evaluate on test set
        if training_params['substrate'] == 'sim':
            return_input = False
        else:
            return_input = True
        # run again on training set (for spiketime saving)
        loss, final_train_accuracy, final_train_outputs, final_train_labels, final_train_inputs = validation_step(
            net, criterion, loader_train, device, return_input=return_input)
        loss, test_accuracy, test_outputs, test_labels, test_inputs = validation_step(
            net, criterion, loader_test, device, return_input=return_input)

        save_result_spikes(foldername, filename, final_train_outputs, final_train_labels, final_train_inputs,
                           test_outputs, test_labels, test_inputs, epoch_dir=(True, e_end))

        # each savepoint needs config to be able to run inference for eval
        save_config(foldername, filename, dataset_params, default_neuron_params,
                    network_layout, training_params, epoch_dir=(True, e_end))
        numpy_rand_state = np.random.get_state()
        torch_rand_state = torch.get_rng_state()
        save_optim_state(foldername, filename, optimizer, scheduler, numpy_rand_state,
                         torch_rand_state, epoch_dir=(True, e_end))
    print("Training finished")
    print('####################')
    print('Test accuracy: {}'.format(test_accuracy))

    #save_result_dict(foldername, filename, net, result_dict)

    return net


def continue_training(dirname, filename, start_epoch, savepoints, dataset_train, dataset_val, dataset_test,
                      net=None):
    dirname_long = dirname + '/epoch_{}/'.format(start_epoch)
    dataset_params, default_neuron_params, network_layout, training_params = load_config(
        osp.join(dirname_long, "config.yaml"))
    if not training_params['torch_seed'] is None:
        torch.manual_seed(training_params['torch_seed'])
    if not training_params['numpy_seed'] is None:
        np.random.seed(training_params['numpy_seed'])
    weight_bumping_steps = []
    tmp_training_progress = []
    all_train_loss = list(load_data(dirname_long, filename, '_train_losses.npy'))
    all_train_accuracy = list(load_data(dirname_long, filename, '_train_accuracies.npy'))
    all_validate_loss = list(load_data(dirname_long, filename, '_val_losses.npy'))
    all_validate_accuracy = list(load_data(dirname_long, filename, '_val_accuracies.npy'))
    all_parameters = load_data(dirname_long, filename, '_parameters_training.npy').item()
    all_parameters = {k: {'name': v['name'], 'params': list(v['params'])} for k, v in all_parameters.items()}

    mean_validate_outputs_sorted = list(load_data(dirname_long, filename, '_mean_val_outputs_sorted.npy'))
    mean_validate_outputs_sorted = [list(item) for item in mean_validate_outputs_sorted]
    std_validate_outputs_sorted = list(load_data(dirname_long, filename, '_std_val_outputs_sorted.npy'))
    std_validate_outputs_sorted = [list(item) for item in std_validate_outputs_sorted]

    if 'enforce_cpu' in training_params.keys() and training_params['enforce_cpu']:
        device = torch.device('cpu')
    else:
        device = utils.get_default_device()
    if not device == 'cpu':
        torch.cuda.manual_seed(training_params['torch_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('training_params')
    print(training_params)
    print('network_layout')
    print(network_layout)

    # setup saving of snapshots
    print('Saving snapshots at epochs {}'.format(savepoints))

    print("loading data")
    loader_train = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_train, batch_size=training_params['batch_size'], shuffle=True), device)
    loader_val = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_val, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)
    loader_test = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_test, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)

    if net is None:
        print("loading network")
        net = utils.network_load(dirname_long, filename, device)
    else:
        print("reusing network")
    if len(savepoints) == 1 and savepoints[0] == start_epoch:
        print("not doing anything with net, only returning it")
        return net

    num_classes = network_layout['layers'][-1]['size']
    print("loading optimizer and scheduler")
    criterion = utils.GetLoss(training_params,
                              num_classes,
                              default_neuron_params['tau_syn'], device)
    optimizer, scheduler, torch_rand_state, numpy_rand_state = load_optim_state(
        dirname_long, filename, net, training_params)

    # evaluate on validation set before training

    print("initial validation started")
    with torch.no_grad():
        loss, validate_accuracy, validate_outputs, validate_labels, _ = validation_step(
            net, criterion, loader_val, device)
        tmp_class_outputs = [[] for i in range(num_classes)]
        for pattern in range(len(validate_outputs)):
            true_label = validate_labels[pattern]
            tmp_class_outputs[true_label].append(validate_outputs[pattern].cpu().detach().numpy())
        for i in range(num_classes):
            tmp_times = np.array(tmp_class_outputs[i])
            inf_mask = np.isinf(tmp_times)
            tmp_times[inf_mask] = np.nan
            mean_times = np.nanmean(tmp_times, 0)
            std_times = np.nanstd(tmp_times, 0)
            mean_validate_outputs_sorted[i].append(mean_times)
            std_validate_outputs_sorted[i].append(std_times)
        print('Initial validation accuracy: {:.3f}'.format(validate_accuracy))
        print('Initial validation loss: {:.3f}'.format(loss))
        all_validate_accuracy.append(validate_accuracy)
        for i, layer in enumerate([l for l in net.layers if l.monitor]):
            if isinstance(layer, utils.DelayLayer):
                tmp_data = layer.effective_delays()
            else:
                tmp_data = next(layer.parameters())
            all_parameters[i]['params'].append(
                tmp_data.data.cpu().detach().numpy().copy()
            )
        all_validate_loss.append(loss.data.cpu().detach().numpy())

    # only seed after initial validation run
    if torch_rand_state is None:
        print("WARNING: Could not load torch rand state, will carry on without")
    else:
        torch.set_rng_state(torch_rand_state)
    if numpy_rand_state is None:
        print("WARNING: Could not load numpy rand state, will carry on without")
    else:
        np.random.set_state(numpy_rand_state)
    print("training started")
    for e_start, e_end in zip([start_epoch] + savepoints[:-1], savepoints):
        print('Starting training from epoch {0} to epoch {1}'.format(e_start, e_end))
        net, criterion, optimizer, scheduler, result_dict = run_epochs(
            e_start, e_end, net, criterion,
            optimizer, scheduler, device, loader_train,
            loader_val, num_classes, all_parameters,
            all_train_loss, all_validate_loss,
            std_validate_outputs_sorted,
            mean_validate_outputs_sorted,
            tmp_training_progress, all_validate_accuracy,
            all_train_accuracy, weight_bumping_steps,
            training_params)
        print('Ending training from epoch {0} to epoch {1}'.format(e_start, e_end))
        all_parameters = result_dict['all_parameters']
        all_train_loss = result_dict['all_train_loss']
        all_validate_loss = result_dict['all_validate_loss']
        std_validate_outputs_sorted = result_dict['std_validate_outputs_sorted']
        mean_validate_outputs_sorted = result_dict['mean_validate_outputs_sorted']
        all_validate_accuracy = result_dict['all_validate_accuracy']
        all_train_accuracy = result_dict['all_train_accuracy']
        weight_bumping_steps = result_dict['weight_bumping_steps']
        tmp_training_progress = result_dict['tmp_training_progress']
        save_result_dict(dirname, filename, net, result_dict, epoch_dir=(True, e_end))

        # evaluate on test set
        if training_params['substrate'] == 'sim':
            return_input = False
        else:
            return_input = True
        # run again on training set (for spiketime saving)
        loss, final_train_accuracy, final_train_outputs, final_train_labels, final_train_inputs = validation_step(
            net, criterion, loader_train, device, return_input=return_input)
        loss, test_accuracy, test_outputs, test_labels, test_inputs = validation_step(
            net, criterion, loader_test, device, return_input=return_input)

        save_result_spikes(dirname, filename, final_train_outputs, final_train_labels, final_train_inputs,
                           test_outputs, test_labels, test_inputs, epoch_dir=(True, e_end))

        # each savepoint needs config to be able to run inference for eval
        save_config(dirname, filename, dataset_params, default_neuron_params,
                    network_layout, training_params, epoch_dir=(True, e_end))
        numpy_rand_state = np.random.get_state()
        torch_rand_state = torch.get_rng_state()
        save_optim_state(dirname, filename, optimizer, scheduler, numpy_rand_state,
                         torch_rand_state, epoch_dir=(True, e_end))
    print("Training finished")

    print('####################')
    print('Test accuracy: {}'.format(test_accuracy))

    save_result_dict(dirname, filename, net, result_dict)

    return net
