#!/bin/python3
import copy
from inspect import indentsize
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import sys
import torch

import training
import utils


class Net(torch.nn.Module):
    def __init__(self, default_neuron_params, network_layout, training_params, device):
        super(Net, self).__init__()
        self.n_inputs = network_layout['n_inputs']
        self.layers_def = network_layout['layers']
        self.n_layers = len(self.layers_def)
        self.device = device

        # prepare the dicts:
        # the feat_in and out are tuples of the (expected) dimensions
        # of the spiketimes, without batch
        last_features_out = [self.n_inputs, 1]
        for i, layer_def in enumerate(self.layers_def):
            self.layers_def[i]['feat_in'] = copy.deepcopy(last_features_out)
            if layer_def['name'] == "NeuronLayer":
                self.layers_def[i]['feat_out'] = [layer_def['size'], 1]
            elif layer_def['name'] == "Biases":
                self.layers_def[i]['feat_out'] = copy.deepcopy(last_features_out)
                self.layers_def[i]['feat_out'][0] += len(layer_def['times'])
            elif layer_def['name'] == "BroadcastLayer" or layer_def['name'].startswith('Delay'):
                self.layers_def[i]['feat_out'] = [
                    self.layers_def[i]['feat_in'][0], None
                ]
                for j in range(i, len(self.layers_def)):
                    if self.layers_def[j]['name'] == 'NeuronLayer':
                        self.layers_def[i]['feat_out'][1] = self.layers_def[j]['size']
                        break
            elif layer_def['name'] == "Multiplex":
                self.layers_def[i]['feat_out'] = copy.deepcopy(last_features_out)
                self.layers_def[i]['feat_out'][0] *= layer_def['factor']
            else:
                raise NotImplementedError(
                    f"Layertype {layer_def['name']} not implemented yet")
            last_features_out = self.layers_def[i]['feat_out']
        self.layers = torch.nn.ModuleList()
        for i, layer_def in enumerate(self.layers_def):
            # decide whether broadcast, delay, or neurons or whatever :)
            if layer_def['name'] == 'BroadcastLayer':
                self.layers.append(utils.BroadcastLayer(
                    layer_def['feat_in'][1],
                    layer_def['feat_out'][1],
                ))
            elif layer_def['name'].startswith('Delay'):
                self.layers.append(utils.DelayLayer(
                    layer_def, layer_def['feat_out'][0], layer_def['feat_out'][1], device
                ))
            elif layer_def['name'] == 'NeuronLayer':
                self.layers.append(utils.NeuronLayer(
                    layer_def, default_neuron_params, training_params, device,
                ))
            elif layer_def['name'] == 'Biases':
                self.layers.append(utils.BiasesLayer(
                    layer_def, device,
                ))
            elif layer_def['name'] == 'Multiplex':
                self.layers.append(utils.MultiplexLayer(layer_def))

        self.delays_rectify()

        self.rounding_precision = training_params.get('rounding_precision')
        self.rounding = self.rounding_precision not in (None, False)
        self.training_params = training_params
        if not training_params['clip_weights_max']:
            for layer in self.layers:
                if isinstance(layer, utils.NeuronLayer):
                    if 'clip_weights_max' in layer.neuron_params:
                        training_params['clip_weights_max'] = True
                        break
        self.default_neuron_params = default_neuron_params
        self.substrate = training_params.get('substrate', 'sim')
        self.fast_eval = False

        if self.rounding:
            print(f"#### Rounding the weights to precision {self.rounding_precision}")
        return

    def delays_rectify(self):
        """Make sure that there are no negative delays.
        Called after optim step"""
        for layer in self.layers:
            if isinstance(layer, utils.DelayLayer):
                layer.delays_rectify()

    def write_weights(self):
        if self.training_params['clip_weights_max']:
            for i, layer in enumerate(self.layers):
                if not isinstance(layer, utils.NeuronLayer):
                    continue

                # get local value if exists, otherwise global one
                maxweight = layer.neuron_params.get(
                    'clip_weights_max', self.training_params['clip_weights_max'])
                if not isinstance(maxweight, bool):
                    self.layers[i].weights.data = torch.clamp(layer.weights.data, -maxweight, maxweight)
        return

    def forward(self, input_times):
        # When rounding we need to save and manipulate weights before forward pass, and after
        if self.rounding and not self.fast_eval:
            float_weights = []
            for layer in self.layers:
                if not isinstance(layer, utils.NeuronLayer):
                    float_weights.append(None)
                    continue
                float_weights.append(layer.weights.data)
                layer.weights.data = self.round_weights(layer.weights.data, self.rounding_precision)

        hidden_times = []
        n_batch = input_times.shape[0]
        for i, lay in enumerate(self.layers):
            input_times_including_bias = input_times
            output_times = self.layers[i](input_times_including_bias)
            if not i == (self.n_layers - 1):
                if isinstance(lay, utils.NeuronLayer):
                    hidden_times.append(output_times)
                else:
                    hidden_times.append([])
                input_times = output_times
            else:
                label_times = output_times
        return_value = label_times, hidden_times

        if self.rounding and not self.fast_eval:
            for layer, floats in zip(self.layers, float_weights):
                if not isinstance(layer, utils.NeuronLayer):
                    continue
                layer.weights.data = floats

        return return_value

    def round_weights(self, weights, precision):
        return (weights / precision).round() * precision


class NetOnHX(Net):
    def __init__(self, default_neuron_params,
                 network_layout, training_params, device):
        super().__init__(default_neuron_params,
                         network_layout, training_params, device)

        self.hx_settings = training.get_hx_settings()

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

        if 'clip_weights_max' in self.training_params and self.training_params['clip_weights_max'] not in (None, False):
            self.training_params['clip_weights_max'] = min(self.training_params['clip_weights_max'],
                                                           63 / self.hx_settings['scale_weights'])
        else:
            self.training_params['clip_weights_max'] = 63 / self.hx_settings['scale_weights']

        self.init_hicannx(device)

        if self.rounding:
            print(f"#### Rounding the weights to precision {self.rounding_precision}")

    def __del__(self):
        if hasattr(self, 'substrate') and self.substrate == 'hx' and hasattr(self, '_ManagedConnection'):
            self._ManagedConnection.__exit__()

    def init_hicannx(self, device):
        self.hx_record_neuron = None
        self.hx_record_target = "membrane"
        self.plot_rasterSimvsEmu = False
        self.plot_raster = False

        self.largest_possible_batch = 0
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
        self._connection = self._ManagedConnection.__enter__()

        for i, lay in enumerate(self.layers_def):
            if lay['name'] == 'Biases':
                assert i == 0, f"On substrate {self.substrate}, bias is only allowed at beginning"
            elif lay['name'] not in ['NeuronLayer', 'BroadcastLayer']:
                raise NotImplementedError()

        if not osp.isfile(self.hx_settings['calibration']):
            raise FileNotFoundError(
                f"Calibration for the current setup {os.environ.get('SLURM_HARDWARE_LICENSES')}"
                "has to be created first "
                f"(probably with 'python py/generate_calibration.py --output calibrations/{os.environ.get('SLURM_HARDWARE_LICENSES')}.npz')")

        num_biases = 0 if self.layers_def[0]['name'] != 'Biases' else len(self.layers_def[0]['times'])
        self.structure = [
            self.n_inputs + num_biases
        ] + [lay['size'] for lay in self.layers_def if lay['name'] == 'NeuronLayer']
        self.hx_backend = fastanddeep.fd_backend.FandDBackend(
            connection=self._connection,
            structure=self.structure,
            calibration=self.hx_settings['calibration'],
            synapse_bias=self.hx_settings['synapse_bias'],
        )

        self.hx_backend.configure()

        if 'calibration_custom' in self.hx_settings:
            self.hx_backend.config_postcalib(self.hx_settings['calibration_custom'])

        self.hx_lastsetweights = [torch.full(l.weights.data.shape, -64)
                                  for l in self.layers if isinstance(l, utils.NeuronLayer)]
        self.write_weights()
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
                    num_batch, self.structure[i + 1],
                    np.full((num_batch, self.structure[i + 1]), np.inf, dtype=float),
                    self.hx_settings['scale_times']))
                return_value.append(times_hw)
            else:
                return_value.append(torch.zeros(num_batch, self.structure[i + 1]))
        if self._record_timings:
            timer.time("spiketrain disecting")
        return return_value

    def write_weights(self):
        maxweight = 63 / self.hx_settings['scale_weights']
        weights_towrite = []
        weights_changed = False
        for i, lay in enumerate([lay for lay in self.layers if isinstance(lay, utils.NeuronLayer)]):
            # contain weights in range accessible on hw
            lay.weights.data = torch.clamp(lay.weights.data, -maxweight, maxweight)
            # prepare weights for writing
            w_tmp = self.round_weights(
                lay.weights.data, 1. / self.hx_settings['scale_weights']
            ).cpu().detach().numpy()
            w_tmp = (w_tmp * self.hx_settings['scale_weights']).astype(int)
            weights_towrite.append(w_tmp)
            if np.any(w_tmp != self.hx_lastsetweights[i]):
                weights_changed = True
                self.hx_lastsetweights[i] = w_tmp

        if weights_changed:
            self.hx_backend.write_weights(*weights_towrite)

    def forward(self, input_times):
        # When rounding we need to save and manipulate weights before forward pass, and after
        if self.rounding and not self.fast_eval:
            float_weights = {}
            for i, layer in enumerate(self.layers):
                if not isinstance(layer, utils.NeuronLayer):
                    continue
                float_weights[i] = layer.weights.data
                layer.weights.data = self.round_weights(layer.weights.data, self.rounding_precision)

        if not self.fast_eval:
            input_times_including_bias = input_times
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
            index_neuron_layer = 0
            for i in range(self.n_layers):
                input_times_including_bias = input_times
                if isinstance(self.layers[i], utils.NeuronLayer):
                    # print(spikes_all_hw[index_neuron_layer])
                    output_times = self.layers[i](
                        input_times_including_bias,
                        output_times=utils.to_device(spikes_all_hw[index_neuron_layer], self.device)
                    )
                    index_neuron_layer += 1
                elif isinstance(self.layers[i], utils.DelayLayer):
                    output_times = self.layers[i](
                        input_times_including_bias,
                        hardware_times=utils.to_device(spikes_all_hw[index_neuron_layer], self.device)
                    )
                    index_neuron_layer += 1
                else:
                    output_times = self.layers[i](
                        input_times_including_bias,
                    )
                if not i == (self.n_layers - 1):
                    if isinstance(self.layers[i], utils.NeuronLayer):
                        hidden_times.append(output_times)
                    else:
                        hidden_times.append([])
                    input_times = output_times
                else:
                    label_times = output_times
            return_value = label_times, hidden_times
        else:
            label_times = spikes_all_hw.pop(-1)
            return_value = label_times, spikes_all_hw

        if self.rounding and not self.fast_eval:
            for i, layer in enumerate(self.layers):
                if not isinstance(layer, utils.NeuronLayer):
                    continue
                layer.weights.data = float_weights[i]

        return return_value


class NetOnHX_PYNN(NetOnHX):
    def __init__(self, default_neuron_params,
                 network_layout, training_params, device):
        super().__init__(default_neuron_params,
                         network_layout, training_params, device)
        self.hx_settings['intrinsic_timescale'] = 1e-3
        self.hx_settings['single_simtime'] = 30.
        self.hx_settings['scale_times'] = self.hx_settings['taum'] * self.hx_settings['intrinsic_timescale']

    def __del__(self):
        if hasattr(self, 'network'):
            print("Deleting a delay_utils.Network from a networks.NetOnHX_PYNN")
            del(self.network)

    def init_hicannx(self, device):
        self.hx_record_neuron = None
        self.hx_record_target = "membrane"

        self.largest_possible_batch = 0
        self._record_timings = False
        self._record_power = False

        import delay_utils
        self.network = delay_utils.Network()

        self.structure = [self.n_inputs]
        self.structure_delays = []
        for i, lay in enumerate(self.layers_def):
            if lay['name'] == 'NeuronLayer':
                self.structure.append(lay['size'])
            elif lay['name'] == 'BroadcastLayer':
                self.structure_delays.append(False)
            elif lay['name'] == 'DelayAxonal':
                self.structure_delays.append(True)
            else:
                raise NotImplementedError(f"For substrate hx_pynn, layer {lay['name']} is not implemented")

        assert len(self.structure) == len(self.structure_delays) + 1, \
            "net cannot be represented with hx_pynn for now"

        assert len(self.structure) == 3, "for now, hx_pynn needs exactly one hidden layer"

        if self.hx_settings.get('_created_from_default', False):
            self.hx_settings['calibration'] = f"calibrations/{os.environ.get('SLURM_HARDWARE_LICENSES')}__adaptationDelays-adapted.pbin"
            print(f"Changing calibration path to '{self.hx_settings['calibration']}'.")

        if not osp.isfile(self.hx_settings['calibration']):
            raise FileNotFoundError(
                "Calibration and delay calibration for the current setup "
                f"{os.environ.get('SLURM_HARDWARE_LICENSES')} has to be created first "
                f"(probably with 'python delay_utils.py doAll')")
        self.network.setup(*self.structure,
                           self.hx_settings['calibration'],
                           self.structure_delays)

        self.hx_lastsetweights = [torch.full(l.weights.data.shape, -64)
                                  for l in self.layers if isinstance(l, utils.NeuronLayer)]
        self.hx_lastsetdelays = [torch.full(l._delay_parameters.data.shape, -64)
                                 for l in self.layers if isinstance(l, utils.DelayLayer)]
        self.write_weights()
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

        if self._record_timings:
            timer.time("prep")
        # create one big spike train
        spiketrain = inpt_batch.cpu().detach().numpy().copy()
        spiketrain += np.arange(num_batch).reshape((-1, 1)) * self.hx_settings['single_simtime']
        spiketrain *= self.hx_settings['scale_times']
        spiketrain = spiketrain.T
        if self._record_timings:
            timer.time("spiketrain creation")
        inp_delay_spikes, hidden_spikes, hidden_delay_spikes, out_spikes = self.network.stimulate(
            spiketrain,
            duration=num_batch * self.hx_settings['single_simtime'] * self.hx_settings['scale_times'],
            record_neuron=self.hx_record_neuron
        )
        if self._record_timings:
            timer.time("hx stimulation")
        
        # voltage
        if self.hx_record_neuron is not None:
            self.trace = self.network.v_mem
        if self._record_timings:
            timer.time("voltage")

        all_spikes = []
        for spikes, sz, doit in [
            (inp_delay_spikes, self.structure[0], self.structure_delays[0]),
            (hidden_spikes, self.structure[1], True),
            (hidden_delay_spikes, self.structure[1], self.structure_delays[1]),
            (out_spikes, self.structure[2], True)
        ]:
            if not doit:
                continue

            cleaned_spikes = np.full((num_batch, sz), np.inf, dtype=float)
            for i_neuron, spikes in enumerate(spikes):
                for i_batch in np.arange(num_batch):
                    spikes_in_batch = spikes[np.logical_and(
                        spikes > i_batch * self.hx_settings['single_simtime'] * self.hx_settings['scale_times'],
                        spikes < (i_batch + 1) * self.hx_settings['single_simtime'] * self.hx_settings['scale_times']
                    )]
                    if len(spikes_in_batch) > 0:
                        cleaned_spikes[
                            i_batch, i_neuron
                        ] = spikes_in_batch.min() - i_batch * self.hx_settings['single_simtime'] * self.hx_settings['scale_times']
            all_spikes.append(torch.tensor(cleaned_spikes))

        all_spikes = [spikes / self.hx_settings['scale_times'] for spikes in all_spikes]

        if self._record_timings:
            timer.time("spiketrain label cleaning")

        # debug plots of accuracy of delays
        if False:  # len(inpt_batch) == 200:
            if self.structure_delays[0]:
                inp_delays = (all_spikes[0].cpu() - inpt_batch.cpu())
                print(f"tensor size {inpt_batch.numel()}, "
                      f"inpt {torch.isinf(inpt_batch).sum()} infs, "
                      f"inpt delays {torch.isinf(inp_delays).sum()} "
                      f"infs {torch.isnan(inp_delays).sum()} nans")
                tmp = (
                    self.layers[0].effective_delays().detach().cpu().numpy().reshape(1, -1).repeat(200, axis=0)
                    - inp_delays.detach().cpu().numpy()
                )
                tmp[np.isinf(tmp)] = np.nan
                print(f"input delays: {np.nanmean(tmp):.3f}±{np.nanstd(tmp):.3f}")
                with np.printoptions(precision=3, suppress=True):
                    print(f"input delays percentiles: {np.nanpercentile(tmp, [0, 10, 25, 50, 75, 90, 100])}")

                fig, ax = plt.subplots(1, 1)
                for i_n in range(tmp.shape[1]):
                    tmptmp = tmp[:, i_n]
                    tmptmp = tmptmp[np.logical_not(np.isnan(tmptmp))]
                    tmptmp = tmptmp[np.logical_not(np.isinf(tmptmp))]
                    if len(tmptmp) > 0:
                        violin = ax.violinplot(tmptmp, [i_n], widths=1.0)
                        utils.make_violin_color(violin, "C1")
                tmptmp = tmp[np.logical_not(np.isnan(tmp))]
                tmptmp = tmptmp[np.logical_not(np.isinf(tmptmp))]
                violin = ax.violinplot(tmptmp, [-1], widths=1.0)
                utils.make_violin_color(violin, "C0")
                ax.set_xticks([-1] + list(range(tmp.shape[1])))
                ax.set_xticklabels(["all"] + list(range(tmp.shape[1])))
                ax.set_xlabel("neurons")
                ax.set_ylabel("missmatch delay (ideal - measured) [taus]")
                fig.tight_layout()
                fig.savefig('dbg_distr_delays_input.png')
                plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                ax.scatter(
                    np.arange(inp_delays.shape[1]).reshape((1, -1)) + np.linspace(-0.25, 0.25, len(inpt_batch)).reshape((-1, 1)),
                    inp_delays,
                    marker='+', color='C0', label="measured",
                )
                ax.scatter(
                    np.arange(inp_delays.shape[1]) - 0.1,
                    self.layers[0].effective_delays().detach().cpu().numpy(),
                    marker=5, color='black', label="target",
                )
                ax.legend(loc='upper right')
                ax.set_ylim(0, None)
                ax.set_xticks(np.arange(inp_delays.shape[1]))
                ax.set_xticklabels(
                    [f"#{i} ({int(n_infs):d})"
                     for i, n_infs in enumerate(list(torch.sum(1.0 * torch.isinf(inp_delays), axis=0)))],
                    rotation=45
                )
                ax2 = ax.twinx()
                ax2.bar(
                    np.arange(inp_delays.shape[1]),
                    torch.mean(1.0 * torch.isinf(inp_delays), axis=0),
                    alpha=0.3, color='C1',
                )
                ax2.set_ylabel('ratio of infs')
                ax2.set_ylim(0, 1.0)

                ax.set_ylabel('delay [tausm]')
                ax.set_xlabel('neurons (# infs)')
                fig.tight_layout()
                fig.savefig(f'dbg_delay_input_{os.environ.get("SLURM_HARDWARE_LICENSES")}.png')
                plt.close(fig)

            if self.structure_delays[1]:
                hidden_delays = (all_spikes[-2] - all_spikes[-3])
                print(f"tensor size {all_spikes[-2].numel()}, "
                      f"hidden {torch.isinf(all_spikes[-3]).sum()} infs, "
                      f"hidden delays {torch.isinf(hidden_delays).sum()} "
                      f"infs {torch.isnan(hidden_delays).sum()} nans")
                tmp = (
                    self.layers[2].effective_delays().detach().cpu().numpy().reshape(1, -1).repeat(200, axis=0)
                    - hidden_delays.detach().numpy()
                )
                tmp[np.isinf(tmp)] = np.nan
                print(f"hidden delays: {np.nanmean(tmp):.3f}±{np.nanstd(tmp):.3f}")
                with np.printoptions(precision=3, suppress=True):
                    print(f"hidden delays percentiles: {np.nanpercentile(tmp, [0, 10, 25, 50, 75, 90, 100])}")

                fig, ax = plt.subplots(1, 1)
                for i_n in range(tmp.shape[1]):
                    tmptmp = tmp[:, i_n]
                    tmptmp = tmptmp[np.logical_not(np.isnan(tmptmp))]
                    tmptmp = tmptmp[np.logical_not(np.isinf(tmptmp))]
                    if len(tmptmp) > 0:
                        violin = ax.violinplot(tmptmp, [i_n], widths=1.0)
                        utils.make_violin_color(violin, "C1")
                tmptmp = tmp[np.logical_not(np.isnan(tmp))]
                tmptmp = tmptmp[np.logical_not(np.isinf(tmptmp))]
                violin = ax.violinplot(tmptmp, [-1], widths=1.0)
                utils.make_violin_color(violin, "C0")
                ax.set_xticks([-1] + list(range(tmp.shape[1])))
                ax.set_xticklabels(["all"] + list(range(tmp.shape[1])))
                ax.set_xlabel("neurons")
                ax.set_ylabel("missmatch delay (ideal - measured) [taus]")
                fig.tight_layout()
                fig.savefig('dbg_distr_delays_hidden.png')
                plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                ax.scatter(
                    np.arange(hidden_delays.shape[1]).reshape((1, -1)) + np.linspace(-0.25, 0.25, len(inpt_batch)).reshape((-1, 1)),
                    hidden_delays,
                    marker='+', color='C0', label="measured",
                )
                ax.scatter(
                    np.arange(hidden_delays.shape[1]) - 0.1,
                    self.layers[2].effective_delays().cpu().detach().numpy(),
                    marker=5, color='black', label="target",
                )
                ax.legend(loc='upper right')
                ax.set_ylim(0, None)
                ax.set_xticks(np.arange(hidden_delays.shape[1]))
                ax.set_xticklabels(
                    [f"#{i} ({int(n_infs):d})"
                     for i, n_infs in enumerate(list(torch.sum(1.0 * torch.isinf(hidden_delays), axis=0)))],
                    rotation=45
                )
                ax2 = ax.twinx()
                ax2.bar(
                    np.arange(hidden_delays.shape[1]) - 0.1,
                    torch.mean( 1.0 * torch.isinf(hidden_delays), axis=0),
                    alpha=0.3, color='C1',
                )
                ax2.set_ylabel('ratio of infs')
                ax2.set_ylim(0, 1.0)

                ax.set_ylabel('delay [tausm]')
                ax.set_xlabel('neurons (# infs)')
                fig.tight_layout()
                fig.savefig(f'dbg_delay_hidden_{os.environ.get("SLURM_HARDWARE_LICENSES")}.png')
                plt.close(fig)

        if self._record_timings:
            timer.time("calculating actual delays")

        return all_spikes

    def write_weights(self):
        maxweight = 63 / self.hx_settings['scale_weights']
        weights_towrite = []
        weights_changed = False
        for i, lay in enumerate([lay for lay in self.layers if isinstance(lay, utils.NeuronLayer)]):
            # contain weights in range accessible on hw
            lay.weights.data = torch.clamp(lay.weights.data, -maxweight, maxweight)
            # prepare weights for writing
            w_tmp = self.round_weights(
                lay.weights.data, 1. / self.hx_settings['scale_weights']
            ).cpu().detach().numpy()
            w_tmp = (w_tmp * self.hx_settings['scale_weights']).astype(int)
            weights_towrite.append(w_tmp)
            if np.any(w_tmp != self.hx_lastsetweights[i]):
                weights_changed = True
                self.hx_lastsetweights[i] = w_tmp

        delays_towrite = [
            (None if not self.structure_delays[0] else
             self.layers[0].effective_delays().cpu().detach().numpy() * self.hx_settings['taum']),
            (None if not self.structure_delays[1] else
             self.layers[2].effective_delays().cpu().detach().numpy() * self.hx_settings['taum']),
        ]
        if delays_towrite is not [None, None]:
            weights_changed = True

        if weights_changed:
            self.network.write_params(weights_towrite, delays_towrite)


def get_network(default_neuron_params, network_layout,
               training_params, device):
    substrate = training_params.get('substrate')
    if substrate == 'sim':
        return Net(default_neuron_params, network_layout,
               training_params, device)
    elif substrate == 'hx':
        return NetOnHX(default_neuron_params, network_layout,
                       training_params, device)
    elif substrate == 'hx_pynn':
        return NetOnHX_PYNN(default_neuron_params, network_layout,
                            training_params, device)
    else:
        raise NotImplementedError(
            f"substrate {substrate} not implemented"
        )
