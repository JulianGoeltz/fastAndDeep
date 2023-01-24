#!python3
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

import utils

torch.set_default_dtype(torch.float64)


def running_mean(x, N=30):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class Net(torch.nn.Module):
    def __init__(self, network_layout, sim_params, device):
        super(Net, self).__init__()
        self.n_inputs = network_layout['n_inputs']
        self.n_layers = network_layout['n_layers']
        self.layer_sizes = network_layout['layer_sizes']
        self.n_biases = network_layout['n_biases']
        self.weight_means = network_layout['weight_means']
        self.weight_stdevs = network_layout['weight_stdevs']
        self.device = device

        if 'bias_times' in network_layout.keys():
            if len(network_layout['bias_times']) > 0 and isinstance(network_layout['bias_times'][0], (list, np.ndarray)):
                self.bias_times = network_layout['bias_times']
            else:
                self.bias_times = [network_layout['bias_times']] * self.n_layers
        else:
            self.bias_times = []
        self.biases = []
        for i in range(self.n_layers):
            bias = utils.to_device(utils.bias_inputs(self.n_biases[i], self.bias_times[i]), device)
            self.biases.append(bias)
        self.layers = torch.nn.ModuleList()
        layer = utils.EqualtimeLayer(self.n_inputs, self.layer_sizes[0],
                                     sim_params, (self.weight_means[0], self.weight_stdevs[0]),
                                     device, self.n_biases[0])
        self.layers.append(layer)
        for i in range(self.n_layers - 1):
            layer = utils.EqualtimeLayer(self.layer_sizes[i], self.layer_sizes[i + 1],
                                         sim_params, (self.weight_means[i + 1], self.weight_stdevs[i + 1]),
                                         device, self.n_biases[i + 1])
            self.layers.append(layer)

        self.rounding_precision = sim_params.get('rounding_precision')
        self.rounding = self.rounding_precision not in (None, False)
        self.sim_params = sim_params
        self.use_hicannx = sim_params.get('use_hicannx', False)

        if self.use_hicannx:
            with open('py/hx_settings.yaml') as f:
                self.hx_settings = yaml.load(f, Loader=yaml.SafeLoader)[
                    os.environ.get('SLURM_HARDWARE_LICENSES')]

            self.hx_settings['retries'] = 5
            self.hx_settings['single_simtime'] = 30.
            self.hx_settings['intrinsic_timescale'] = 1e-6
            self.hx_settings['scale_times'] = self.hx_settings['taum'] * self.hx_settings['intrinsic_timescale']

            if self.rounding:
                self.rounding_precision = max(self.rounding,
                                              1. / self.hx_settings['scale_weights'])
            else:
                self.rounding_precision = 1. / self.hx_settings['scale_weights']
                self.rounding = True

            if 'clip_weights_max' in self.sim_params and self.sim_params['clip_weights_max'] not in (None, False):
                self.sim_params['clip_weights_max'] = min(self.sim_params['clip_weights_max'],
                                                          63 / self.hx_settings['scale_weights'])
            else:
                self.sim_params['clip_weights_max'] = 63 / self.hx_settings['scale_weights']

            self.init_hicannx(device)

        if self.rounding:
            print(f"#### Rounding the weights to precision {self.rounding_precision}")
        return

    def __del__(self):
        if self.use_hicannx and hasattr(self, '_ManagedConnection'):
            self._ManagedConnection.__exit__()

    def init_hicannx(self, device):
        assert np.all(np.array(self.n_biases[1:]) == 0), "for now, on HX no bias in any but first layer is possible"

        self.hx_record_neuron = None
        self.hx_record_target = "membrane"
        self.plot_rasterSimvsEmu = False
        self.plot_raster = False

        self.largest_possible_batch = 0
        self.fast_eval = False
        self._record_timings = False
        self._record_power = False

        import pylogging
        pylogging.reset()
        pylogging.default_config(
            level=pylogging.LogLevel.WARN,
            fname="",
            # level=pylogging.LogLevel.DEBUG,
            # format='%(levelname)-6s%(asctime)s,%(msecs)03d %(name)s  %(message)s',
            print_location=False,
            color=True,
            date_format="RELATIVE")

        # import modified backend based on strobe backend from SB and BC
        import fastanddeep.fd_backend
        import pyhxcomm_vx as hxcomm
        self._ManagedConnection = hxcomm.ManagedConnection()
        connection = self._ManagedConnection.__enter__()

        self.hx_backend = fastanddeep.fd_backend.FandDBackend(
            connection=connection,
            structure=[self.n_inputs + self.n_biases[0]] + self.layer_sizes,
            calibration=self.hx_settings['calibration'],
            synapse_bias=self.hx_settings['synapse_bias'],
        )

        self.hx_backend.configure()

        if 'calibration_custom' in self.hx_settings:
            self.hx_backend.config_postcalib(self.hx_settings['calibration_custom'])

        self.hx_lastsetweights = [torch.full(l.weights.data.shape, -64) for l in self.layers]
        self.write_weights_to_hicannx()
        return

    def stimulate_hx(self, inpt_batch):
        if self._record_timings:
            timer = utils.TIMER("==")
        num_batch, num_inp = inpt_batch.shape
        # in case we have a batch that is too long do slice consecutively
        if self.largest_possible_batch > 0 and num_batch > self.largest_possible_batch:
            return_value = [[]] * self.n_layers
            iters = int(np.ceil(num_batch / self.largest_possible_batch))
            print(f"Splitting up batch of size {num_batch} into {iters} "
                  f"batches of largest size {self.largest_possible_batch}")
            for i in range(iters):
                tmp = self.stimulate_hx(
                    inpt_batch[i * self.largest_possible_batch: (i + 1) * self.largest_possible_batch])
                for j, l in enumerate(tmp):
                    if i == 0:
                        return_value[j] = [l]
                    else:
                        return_value[j].append(l)
            return [torch.cat(l, dim=0) for l in return_value]

        # create one long spiketrain of batch
        spiketrain, simtime = utils.hx_spiketrain_create(
            inpt_batch.cpu().detach().numpy(),
            self.hx_settings['single_simtime'],
            self.hx_settings['scale_times'],
            np.arange(num_batch).reshape((-1, 1)).repeat(num_inp, 1),
            np.empty_like(inpt_batch.cpu(), dtype=int),
        )
        # remove infs from spiketrain
        spiketrain = utils.hx_spiketrain_purgeinf(spiketrain)
        if self._record_timings:
            timer.time("spiketrain creation&purging")
        # pass inputs to hicannx
        if self.hx_record_neuron is not None:
            self.hx_backend.set_readout(self.hx_record_neuron, target=self.hx_record_target)
        retries = self.hx_settings['retries']
        while retries > 0:
            if self._record_timings:
                timer.time("shit")
            spikes_all, trace = self.hx_backend.run(
                duration=simtime,
                input_spikes=[spiketrain],
                record_madc=(self.hx_record_neuron is not None),
                measure_power=self._record_power,
                fast_eval=self.fast_eval,
                record_timings=self._record_timings,
            )
            if self._record_timings:
                timer.time("hx_backend.run")
                print("==time on chip should be "
                      f"{self.hx_settings['single_simtime'] * self.hx_settings['scale_times'] * 1e4}")
            spikes_all = [s[0] for s in spikes_all]
            # repeat if sensibility check (first and last layer) not passed (if fast_eval just go ahead)
            if self.fast_eval or ((len(spikes_all[0]) == 0 or spikes_all[0][:, 0].max() < simtime) and
                                  (len(spikes_all[-1]) == 0 or spikes_all[-1][:, 0].max() < simtime)):
                if not self.fast_eval:
                    last_spike = max(spikes_all[0][:, 0]) if len(spikes_all[0]) > 0 else 0.
                    # print(f"last_spike occurs as {last_spike} for simtime {simtime}")
                    if simtime - last_spike > 0.001:
                        # in test we have runs without output spikes
                        if sys.argv[0][:5] != 'test_':
                            # raise Exception("seems to be that batch wasn't fully computed")
                            pass
                    # print(np.unique(spikes_l[:, 1]))
                    # sys.exit()
                break
            retries -= 1
        else:
            raise Exception("FPGA stalled and retries were exceeded")

        # save trace if recorded
        if self.hx_record_neuron is not None:
            # get rid of error values (FPGA fail or sth)
            mask_trace = (trace[:, 0] == 0)
            if mask_trace.sum() > 0:
                print(f"#### trace of neuron {self.hx_record_neuron} "
                      f"received {mask_trace.sum()} steps of value 0")
                trace = trace[np.logical_not(mask_trace)]
            self.trace = trace

        # disect spiketrains (with numba it looks a bit complicated)
        return_value = []
        if self._record_timings:
            timer.time("stuff")
        for i, spikes in enumerate(spikes_all):
            # if fast eval only label layer, otherwise all
            if not self.fast_eval or i == len(spikes_all) - 1:
                # need to explicitly sort
                spikes_t, spikes_id = spikes[:, 0], spikes[:, 1].astype(int)
                sorting = np.argsort(spikes_t)
                times_hw = torch.tensor(utils.hx_spiketrain_disect(
                    spikes_t[sorting], spikes_id[sorting], self.hx_settings['single_simtime'],
                    num_batch, self.layer_sizes[i],
                    np.full((num_batch, self.layer_sizes[i]), np.inf, dtype=float),
                    self.hx_settings['scale_times']))
                return_value.append(times_hw)
            else:
                return_value.append(torch.zeros(num_batch, self.layer_sizes[i]))
        if self._record_timings:
            timer.time("spiketrain disecting")
        return return_value

    def write_weights_to_hicannx(self):
        if not self.use_hicannx:
            if self.sim_params['clip_weights_max']:
                for i, layer in enumerate(self.layers):
                    maxweight = self.sim_params['clip_weights_max']
                    self.layers[i].weights.data = torch.clamp(layer.weights.data, -maxweight, maxweight)
            return

        maxweight = 63 / self.hx_settings['scale_weights']
        weights_towrite = []
        weights_changed = False
        for i in range(self.n_layers):
            # contain weights in range accessible on hw
            self.layers[i].weights.data = torch.clamp(self.layers[i].weights.data, -maxweight, maxweight)
            # prepare weights for writing
            w_tmp = self.round_weights(
                self.layers[i].weights.data, 1. / self.hx_settings['scale_weights']
            ).cpu().detach().numpy()
            w_tmp = (w_tmp * self.hx_settings['scale_weights']).astype(int)
            weights_towrite.append(w_tmp)
            if np.any(w_tmp != self.hx_lastsetweights[i]):
                weights_changed = True

        if weights_changed:
            self.hx_backend.write_weights(*weights_towrite)

    def forward(self, input_times):
        # When rounding we need to save and manipulate weights before forward pass, and after
        if self.rounding and not self.fast_eval:
            float_weights = []
            for layer in self.layers:
                float_weights.append(layer.weights.data)
                layer.weights.data = self.round_weights(layer.weights.data, self.rounding_precision)

        if not self.use_hicannx:
            hidden_times = []
            for i in range(self.n_layers):
                input_times_including_bias = torch.cat(
                    (input_times,
                     self.biases[i].view(1, -1).expand(len(input_times), -1)),
                    1)
                output_times = self.layers[i](input_times_including_bias)
                if not i == (self.n_layers - 1):
                    hidden_times.append(output_times)
                    input_times = output_times
                else:
                    label_times = output_times
            return_value = label_times, hidden_times
        else:
            if not self.fast_eval:
                input_times_including_bias = torch.cat(
                    (input_times,
                     self.biases[0].view(1, -1).expand(len(input_times), -1)),
                    1)
            else:
                input_times_including_bias = input_times

            if self._record_timings:
                timer = utils.TIMER()
            spikes_all_hw = self.stimulate_hx(input_times_including_bias)
            if self._record_timings:
                timer.time("net.stimulate_hx")

            # pass to layers pro forma to enable easy backward pass
            if not self.fast_eval:
                hidden_times = []
                for i in range(self.n_layers):
                    input_times_including_bias = torch.cat(
                        (input_times,
                         self.biases[i].view(1, -1).expand(len(input_times), -1)),
                        1)
                    output_times = self.layers[i](
                        input_times_including_bias,
                        output_times=utils.to_device(spikes_all_hw[i], self.device))
                    if not i == (self.n_layers - 1):
                        hidden_times.append(output_times)
                        input_times = output_times
                    else:
                        label_times = output_times
                return_value = label_times, hidden_times
            else:
                label_times = spikes_all_hw.pop(-1)
                return_value = label_times, spikes_all_hw

        if self.rounding and not self.fast_eval:
            for layer, floats in zip(self.layers, float_weights):
                layer.weights.data = floats

        return return_value

    def round_weights(self, weights, precision):
        return (weights / precision).round() * precision


def load_data(dirname, filename, dataname):
    path = dirname + '/' + filename + dataname
    data = np.load(path, allow_pickle=True)
    return data


def load_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data['dataset'], data['neuron_params'], data['network_layout'], data['training_params']


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
            loss = criterion(outputs, labels) * len(labels)
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
    for i, times in enumerate(hidden_times):
        # we want mean over batches and neurons
        denominator = times.shape[0] * times.shape[1]
        non_spikes = torch.isinf(times) + torch.isnan(times)
        num_nonspikes = float(non_spikes.bool().sum())
        if num_nonspikes / denominator > training_params['max_num_missing_spikes'][i]:
            weights_bumped = i
            break
    else:
        # else after for only executed if no break happened
        i = -1
        denominator = label_times.shape[0] * label_times.shape[1]
        non_spikes = torch.isinf(label_times) + torch.isnan(label_times)
        num_nonspikes = float(non_spikes.bool().sum())
        if num_nonspikes / denominator > training_params['max_num_missing_spikes'][-1]:
            weights_bumped = -1
    if weights_bumped != -2:
        if training_params['weight_bumping_exp'] and weights_bumped == last_weights_bumped:
            bump_val *= 2
        else:
            bump_val = training_params['weight_bumping_value']
        if training_params['weight_bumping_targeted']:
            # make bool and then int to have either zero or ones
            should_bump = non_spikes.sum(axis=0).bool().int()
            n_in = net.layers[i].weights.data.size()[0]
            bumps = should_bump.repeat(n_in, 1) * bump_val
            net.layers[i].weights.data += bumps
        else:
            net.layers[i].weights.data += bump_val

        # print("epoch {0}, batch {1}: missing {4} spikes, bumping weights by {2} (targeted_bump={3})".format(
        #     epoch, batch, bump_val, training_params['weight_bumping_targeted'],
        #     "label" if weights_bumped == -1 else "hidden"))
    return weights_bumped, bump_val


def save_untrained_network(dirname, filename, net):
    if (dirname is None) or (filename is None):
        return
    path = '../experiment_results/' + dirname
    try:
        os.makedirs(path)
        print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")
    if not path[-1] == '/':
        path += '/'
    # save network
    if not net.use_hicannx:
        torch.save(net, path + filename + '_untrained_network.pt')
    else:
        tmp_backend = net.hx_backend
        tmp_MC = net._ManagedConnection
        del net.hx_backend
        del net._ManagedConnection
        torch.save(net, path + filename + '_untrained_network.pt')
        net.hx_backend = tmp_backend
        net._ManagedConnection = tmp_MC
        # save hardware licence to identify the used hicann
        with open(path + filename + '_hw_licences.txt', 'w') as f:
            f.write(os.environ.get('SLURM_HARDWARE_LICENSES'))
        # save current calib settings
        shutil.copy(osp.join('py', 'hx_settings.yaml'), path + '/hw_settings.yaml')
    return


def save_config(dirname, filename, neuron_params, network_layout, training_params, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    dirname = '../experiment_results/' + dirname
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    if not osp.isdir(dirname):
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    # save parameter configs
    with open(osp.join(dirname, 'config.yaml'), 'w') as f:
        yaml.dump({"dataset": filename, "neuron_params": neuron_params,
                   "network_layout": network_layout, "training_params": training_params}, f)
    with open(osp.join(dirname, filename + '_gitsha.txt'), 'w') as f:
        try:
            f.write(subprocess.check_output(["git", "rev-parse", "HEAD"]).decode())
        except subprocess.CalledProcessError:
            print("Not a git repository, can't save git sha")
    return


def save_data(dirname, filename, net, label_weights, train_losses, train_accuracies, val_losses, val_accuracies,
              val_labels, mean_val_outputs_sorted, std_val_outputs_sorted, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    dirname = '../experiment_results/' + dirname
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
    if not net.use_hicannx:
        torch.save(net, dirname + filename + '_network.pt')
    else:
        tmp_backend = net.hx_backend
        tmp_MC = net._ManagedConnection
        del net.hx_backend
        del net._ManagedConnection
        torch.save(net, dirname + filename + '_network.pt')
        net.hx_backend = tmp_backend
        net._ManagedConnection = tmp_MC
        # save hardware licence to identify the used hicann
        with open(dirname + filename + '_hw_licences.txt', 'w') as f:
            f.write(os.environ.get('SLURM_HARDWARE_LICENSES'))
        # save current calib settings
        with open(dirname + '/hx_settings.yaml', 'w') as f:
            yaml.dump({os.environ.get('SLURM_HARDWARE_LICENSES'): net.hx_settings}, f)
    # save training result
    np.save(dirname + filename + '_label_weights_training.npy', label_weights)
    np.save(dirname + filename + '_train_losses.npy', train_losses)
    np.save(dirname + filename + '_train_accuracies.npy', train_accuracies)
    np.save(dirname + filename + '_val_losses.npy', val_losses)
    np.save(dirname + filename + '_val_accuracies.npy', val_accuracies)
    np.save(dirname + filename + '_val_labels.npy', val_labels)
    np.save(dirname + filename + '_mean_val_outputs_sorted.npy', mean_val_outputs_sorted)
    np.save(dirname + filename + '_std_val_outputs_sorted.npy', std_val_outputs_sorted)
    return


def save_result_spikes(dirname, filename, train_times, train_labels, train_inputs,
                       test_times, test_labels, test_inputs, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    dirname = '../experiment_results/' + dirname
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    # stunt to avoid saving tensors
    train_times = np.array([item.detach().cpu().numpy() for item in train_times])
    test_times = np.array([item.detach().cpu().numpy() for item in test_times])
    np.save(dirname + filename + '_train_spiketimes.npy', train_times)
    np.save(dirname + filename + '_train_labels.npy', train_labels)
    if not (train_inputs is None):
        np.save(dirname + filename + '_train_inputs.npy', train_inputs)
    np.save(dirname + filename + '_test_spiketimes.npy', test_times)
    np.save(dirname + filename + '_test_labels.npy', test_labels)
    if not (test_inputs is None):
        np.save(dirname + filename + '_test_inputs.npy', test_inputs)
    return


def save_optim_state(dirname, filename, optimizer, scheduler, np_rand_state, torch_rand_state, epoch_dir=(False, -1)):
    if (dirname is None) or (filename is None):
        return
    dirname = '../experiment_results/' + dirname
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
        optimizer = torch.optim.Adam(net.parameters(), lr=training_params['learning_rate'])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=training_params['learning_rate'],
                                    momentum=training_params['momentum'])
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


def apply_noise(input_times, noise_params, device):
    shape = input_times.size()
    noise = utils.to_device(torch.zeros(shape), device)
    noise.normal_(noise_params['mean'], noise_params['std_dev'])
    input_times = input_times + noise
    negative = input_times < 0.
    input_times[negative] *= -1
    return input_times


def run_epochs(e_start, e_end, net, criterion, optimizer, scheduler, device, trainloader, valloader,
               num_classes, all_weights, all_train_loss, all_validate_loss, std_validate_outputs_sorted,
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
            if last_weights_bumped != -2:  # means bumping happened
                weight_bumping_steps.append(epoch * len(trainloader) + j)
            else:
                loss = criterion(label_times, labels)
                loss.backward()
                optimizer.step()
                # on hardware we need extra step to write weights
                train_loss.append(loss.item())
            net.write_weights_to_hicannx()
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
                fig.savefig('live_accuracy.png')
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

            tmp_class_outputs = [[] for i in range(num_classes)]
            for pattern in range(len(validate_outputs)):
                true_label = validate_labels[pattern]
                tmp_class_outputs[true_label].append(validate_outputs[pattern].cpu().detach().numpy())
            for i in range(num_classes):
                tmp_times = np.array(tmp_class_outputs[i])
                tmp_times[np.isinf(tmp_times)] = np.NaN
                mask_notAllNan = np.logical_not(np.isnan(tmp_times)).sum(0) > 0
                mean_times = np.ones(tmp_times.shape[1:]) * np.NaN
                std_times = np.ones(tmp_times.shape[1:]) * np.NaN
                mean_times[mask_notAllNan] = np.nanmean(tmp_times[:, mask_notAllNan], 0)
                std_times[mask_notAllNan] = np.nanstd(tmp_times[:, mask_notAllNan], 0)
                mean_validate_outputs_sorted[i].append(mean_times)
                std_validate_outputs_sorted[i].append(std_times)

            all_validate_accuracy.append(validate_accuracy)
            all_weights.append(net.layers[-1].weights.data.cpu().detach().numpy().copy())
            all_validate_loss.append(validate_loss.data.cpu().detach().numpy())

        if (epoch % print_step) == 0:
            print("... {0}% done, train accuracy: {4:.3f}, validation accuracy: {1:.3f},"
                  "trainings loss: {2:.5f}, validation loss: {3:.5f}".format(
                      epoch * 100 / training_params['epoch_number'], validate_accuracy,
                      np.mean(train_loss) if len(train_loss) > 0 else np.NaN,
                      validate_loss, train_accuracy),
                  flush=True)

        result_dict = {'all_weights': all_weights,
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


def train(training_params, network_layout, neuron_params, dataset_train, dataset_val, dataset_test,
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
    if not isinstance(training_params['max_num_missing_spikes'], (list, tuple, np.ndarray)):
        training_params['max_num_missing_spikes'] = [
            training_params['max_num_missing_spikes']] * network_layout['n_layers']

    # save parameter config
    save_config(foldername, filename, neuron_params, network_layout, training_params)

    # create sim params
    sim_params = {k: training_params.get(k, False)
                  for k in ['use_forward_integrator', 'resolution', 'sim_time',
                            'rounding_precision', 'use_hicannx', 'max_dw_norm',
                            'clip_weights_max']
                  }
    sim_params.update(neuron_params)

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
    net = utils.to_device(
        Net(network_layout, sim_params, device),
        device)
    save_untrained_network(foldername, filename, net)

    print("loss function")
    criterion = utils.GetLoss(training_params, 
                              network_layout['layer_sizes'][-1],
                              sim_params['tau_syn'], device)

    if training_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=training_params['learning_rate'])
    elif training_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=training_params['learning_rate'],
                                    momentum=training_params['momentum'])
    else:
        raise NotImplementedError(f"optimizer {training_params['optimizer']} not implemented")
    scheduler = None
    if 'lr_scheduler' in training_params.keys():
        scheduler = setup_lr_scheduling(training_params['lr_scheduler'], optimizer)

    # evaluate on validation set before training
    num_classes = network_layout['layer_sizes'][-1]
    all_weights = []
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
            tmp_times[inf_mask] = np.NaN
            mean_times = np.nanmean(tmp_times, 0)
            std_times = np.nanstd(tmp_times, 0)
            mean_validate_outputs_sorted[i].append(mean_times)
            std_validate_outputs_sorted[i].append(std_times)
        print('Initial validation accuracy: {:.3f}'.format(validate_accuracy))
        print('Initial validation loss: {:.3f}'.format(loss))
        all_validate_accuracy.append(validate_accuracy)
        all_weights.append(net.layers[-1].weights.data.cpu().detach().numpy().copy())
        all_validate_loss.append(loss.data.cpu().detach().numpy())

    print("training started")
    for i, e_end in enumerate(savepoints):
        if i == 0:
            e_start = 0
        else:
            e_start = savepoints[i - 1]
        print('Starting training from epoch {0} to epoch {1}'.format(e_start, e_end))
        net, criterion, optimizer, scheduler, result_dict = run_epochs(
            e_start, e_end, net, criterion,
            optimizer, scheduler, device, loader_train,
            loader_val, num_classes, all_weights,
            all_train_loss, all_validate_loss,
            std_validate_outputs_sorted,
            mean_validate_outputs_sorted,
            tmp_training_progress, all_validate_accuracy,
            all_train_accuracy, weight_bumping_steps,
            training_params)
        print('Ending training from epoch {0} to epoch {1}'.format(e_start, e_end))
        all_weights = result_dict['all_weights']
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
        save_data(foldername, filename, net, all_weights, all_train_loss,
                  all_train_accuracy, all_validate_loss, all_validate_accuracy,
                  validate_labels, mean_validate_outputs_sorted, std_validate_outputs_sorted,
                  epoch_dir=(True, e_end))

        # evaluate on test set
        if training_params['use_hicannx']:
            return_input = True
        else:
            return_input = False
        # run again on training set (for spiketime saving)
        loss, final_train_accuracy, final_train_outputs, final_train_labels, final_train_inputs = validation_step(
            net, criterion, loader_train, device, return_input=return_input)
        loss, test_accuracy, test_outputs, test_labels, test_inputs = validation_step(
            net, criterion, loader_test, device, return_input=return_input)

        save_result_spikes(foldername, filename, final_train_outputs, final_train_labels, final_train_inputs,
                           test_outputs, test_labels, test_inputs, epoch_dir=(True, e_end))

        # each savepoint needs config to be able to run inference for eval
        save_config(foldername, filename, neuron_params, network_layout, training_params, epoch_dir=(True, e_end))
        numpy_rand_state = np.random.get_state()
        torch_rand_state = torch.get_rng_state()
        save_optim_state(foldername, filename, optimizer, scheduler, numpy_rand_state,
                         torch_rand_state, epoch_dir=(True, e_end))
    print("Training finished")
    print('####################')
    print('Test accuracy: {}'.format(test_accuracy))

    save_data(foldername, filename, net, all_weights, all_train_loss,
              all_train_accuracy, all_validate_loss, all_validate_accuracy,
              validate_labels, mean_validate_outputs_sorted, std_validate_outputs_sorted)

    return net


def continue_training(dirname, filename, start_epoch, savepoints, dataset_train, dataset_val, dataset_test,
                      net=None):
    dirname_long = dirname + '/epoch_{}/'.format(start_epoch)
    dataset, neuron_params, network_layout, training_params = load_config(osp.join(dirname_long, "config.yaml"))
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
    all_weights = list(load_data(dirname_long, filename, '_label_weights_training.npy'))
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

    if not isinstance(training_params['max_num_missing_spikes'], (list, tuple, np.ndarray)):
        training_params['max_num_missing_spikes'] = [
            training_params['max_num_missing_spikes']] * network_layout['n_layers']

    # create sim params
    sim_params = {k: training_params.get(k, False)
                  for k in ['use_forward_integrator', 'resolution', 'sim_time',
                            'rounding_precision', 'use_hicannx', 'max_dw_norm',
                            'clip_weights_max']
                  }
    sim_params.update(neuron_params)

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

    print("loading optimizer and scheduler")
    criterion = utils.GetLoss(training_params,
                              network_layout['layer_sizes'][-1],
                              sim_params['tau_syn'], device)
    optimizer, scheduler, torch_rand_state, numpy_rand_state = load_optim_state(
        dirname_long, filename, net, training_params)

    # evaluate on validation set before training
    num_classes = network_layout['layer_sizes'][-1]

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
            tmp_times[inf_mask] = np.NaN
            mean_times = np.nanmean(tmp_times, 0)
            std_times = np.nanstd(tmp_times, 0)
            mean_validate_outputs_sorted[i].append(mean_times)
            std_validate_outputs_sorted[i].append(std_times)
        print('Initial validation accuracy: {:.3f}'.format(validate_accuracy))
        print('Initial validation loss: {:.3f}'.format(loss))
        all_validate_accuracy.append(validate_accuracy)
        all_weights.append(net.layers[-1].weights.data.cpu().detach().numpy().copy())
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
    for i, e_end in enumerate(savepoints):
        if i == 0:
            e_start = start_epoch
        else:
            e_start = savepoints[i - 1]
        print('Starting training from epoch {0} to epoch {1}'.format(e_start, e_end))
        net, criterion, optimizer, scheduler, result_dict = run_epochs(
            e_start, e_end, net, criterion,
            optimizer, scheduler, device, loader_train,
            loader_val, num_classes, all_weights,
            all_train_loss, all_validate_loss,
            std_validate_outputs_sorted,
            mean_validate_outputs_sorted,
            tmp_training_progress, all_validate_accuracy,
            all_train_accuracy, weight_bumping_steps,
            training_params)
        print('Ending training from epoch {0} to epoch {1}'.format(e_start, e_end))
        all_weights = result_dict['all_weights']
        all_train_loss = result_dict['all_train_loss']
        all_validate_loss = result_dict['all_validate_loss']
        std_validate_outputs_sorted = result_dict['std_validate_outputs_sorted']
        mean_validate_outputs_sorted = result_dict['mean_validate_outputs_sorted']
        all_validate_accuracy = result_dict['all_validate_accuracy']
        all_train_accuracy = result_dict['all_train_accuracy']
        weight_bumping_steps = result_dict['weight_bumping_steps']
        tmp_training_progress = result_dict['tmp_training_progress']
        save_data(dirname, filename, net, all_weights, all_train_loss,
                  all_train_accuracy, all_validate_loss, all_validate_accuracy,
                  validate_labels, mean_validate_outputs_sorted, std_validate_outputs_sorted,
                  epoch_dir=(True, e_end))

        # evaluate on test set
        if training_params['use_hicannx']:
            return_input = True
        else:
            return_input = False
        # run again on training set (for spiketime saving)
        loss, final_train_accuracy, final_train_outputs, final_train_labels, final_train_inputs = validation_step(
            net, criterion, loader_train, device, return_input=return_input)
        loss, test_accuracy, test_outputs, test_labels, test_inputs = validation_step(
            net, criterion, loader_test, device, return_input=return_input)

        save_result_spikes(dirname, filename, final_train_outputs, final_train_labels, final_train_inputs,
                           test_outputs, test_labels, test_inputs, epoch_dir=(True, e_end))

        # each savepoint needs config to be able to run inference for eval
        save_config(dirname, filename, neuron_params, network_layout, training_params, epoch_dir=(True, e_end))
        numpy_rand_state = np.random.get_state()
        torch_rand_state = torch.get_rng_state()
        save_optim_state(dirname, filename, optimizer, scheduler, numpy_rand_state,
                         torch_rand_state, epoch_dir=(True, e_end))
    print("Training finished")

    print('####################')
    print('Test accuracy: {}'.format(test_accuracy))

    save_data(dirname, filename, net, all_weights, all_train_loss,
              all_train_accuracy, all_validate_loss, all_validate_accuracy,
              validate_labels, mean_validate_outputs_sorted, std_validate_outputs_sorted)

    return net
