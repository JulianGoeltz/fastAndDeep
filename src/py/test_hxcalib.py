#!python3
"""testing usefulness of calibration for hicannx

in a standardised, central way calibrations and settings are checked
defined in /src/py/hx_settings.yaml
test is composed of:
* initialisation
* responsivity:
    * for selected neuron (do several in sequence) have some inputs with maximum weight
    * record trace and spikes
    * also do it for double # inputs, half weight
    * plot for all neurons against calculated membrane dynamic
* quiescence of all neurons for no input
    * separate for hidden layer, label layer
    * case1: presynaptic neurons don't fire, but maximum weight
    * case2: presynaptic spikes, but 0 weight
* fireability: all neurons can fire: maximum weights, see how many neurons don't fire at all
* (TODO) spike time distribution:
    * Set seed
    * get random spike distribution
    * get random weight matrix
    * for rate in range(0.0, 1.0, 0.1) :
        * percentage of neurons that have a spike
        * average spike time (mean, percentile(0, 5, 25, 50, 75, 95, 100), all) saved
"""
import datetime
import json
import matplotlib as mpl
mpl.use('agg')
import matplotlib.gridspec as mpl_gs
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
import sys
import torch
import unittest
import yaml

import training
import utils


# parameters that define tests
NUMBER_OF_NEURONS = 256
params = {
    'all_batchsize': 30,
    'dist_seed': 42,
    'resp_inputs': torch.tensor(range(0, 4)),
    'resp_relativeinputs': not False,
    'resp_weight_frac': 0.7,
    # 'resp_neurons': range(118, 138),
    'resp_neurons': range(0, 10),
    'resp_trace_subtract_rest': not True,
    'save_individual_json': False,
}
folder_base = "debug_calib"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, range):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class CalibTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("$$$$$$$$ initialisation")
        if os.environ.get('SLURM_HARDWARE_LICENSES') is None:
            raise OSError("This test has to be executed with hardware access, "
                          "tested with 'SLURM_HARDWARE_LICENSES' variable")
        with open('py/hx_settings.yaml') as f:
            hx_settings = yaml.load(f, Loader=yaml.SafeLoader)
        hx_setup = os.environ.get('SLURM_HARDWARE_LICENSES')
        try:
            self.hx_setup_no = int(hx_setup[1:3])
        except ValueError:
            raise OSError("Setup number is extracted from env variable SLURM_HARDWARE_LICENSES "
                          f"({hx_setup}), expected to be of the form "
                          "'W[setupno]F3'")

        if self.hx_setup_no not in hx_settings:
            raise OSError(f"Setup no {self.hx_setup_no} (from SLURM_HARDWARE_LICENSES variable "
                          f"'{hx_setup}') is not described in hx settings file")
        self.hx_settings = hx_settings[self.hx_setup_no]
        self.folder = f"{folder_base}_{self.hx_setup_no}"
        self.filename = f"{self.folder}/plot_test_calib_{self.hx_setup_no}_"\
            f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
        print(f"$$ Testing on setup {self.hx_setup_no} with the following settings (others derived, see Net.init)")
        pprint(self.hx_settings, indent=5)

        # outside params
        print("$$ using params:")
        pprint(params, indent=5)

        # create net
        self.sim_params = self.hx_settings['neuron_params']
        self.sim_params['use_hicannx'] = True
        self.sim_params['uniform_weights'] = False
        self.network_layout = {
            "bias_times": [],
            "layer_sizes": [256],
            "n_biases": [0, 0],
            "n_inputs": 256,
            "n_layers": 1,
            "weight_means": [0.1, ],
            "weight_stdevs": [0.4, ],
        }
        self.device = torch.device('cpu')
        self.net = training.Net(self.network_layout, self.sim_params, 0,  # last is dw norm
                                self.device)
        print("$$ network set up")

    def get_traces(self):
        results, medians = [], []
        for neuron in params['resp_neurons']:
            # set weights
            target_weight = 63. * params['resp_weight_frac'] / self.net.hx_settings['scale_weights']

            weights = torch.zeros(256, 256).double()
            if params['resp_relativeinputs']:
                weights[(params['resp_inputs'] + neuron) % NUMBER_OF_NEURONS,
                        neuron] = target_weight
            else:
                weights[(params['resp_inputs']) % NUMBER_OF_NEURONS,
                        neuron] = target_weight
            self.net.layers[0].weights.data = weights
            self.net.write_weights_to_hicannx()

            # set recording
            self.net.hx_record_neuron = neuron

            # create batch
            inputs = torch.ones((params['all_batchsize'], 256)) * float('inf')
            if params['resp_relativeinputs']:
                inputs[:, (params['resp_inputs'] + neuron) % NUMBER_OF_NEURONS] = 0.
            else:
                inputs[:, (params['resp_inputs']) % NUMBER_OF_NEURONS] = 0.

            # emulate
            spiketimes, _ = self.net(inputs)

            # handle traces
            trace = self.net.trace
            # calculate resting potential
            time_mask = np.logical_and(200e-6 < trace[:, 0],
                                       trace[:, 0] < 400e-6)
            average = np.median(trace[time_mask, 1])
            medians.append(average)
            print(f"$$ neuron {neuron} has resting median {average}")

            results.append((neuron, spiketimes.detach().numpy(), trace))
        return results, medians, inputs

    def test_1_responsivity(self):
        print("$$$$$$$$ test_responsivity")

        print("$$$$ data collection")
        print("$$ recording traces and spikes")
        self.net.hx_backend.set_spiking(True)
        results_sp, medians, inputs = self.get_traces()
        self.net.hx_backend.set_spiking(False)
        results_nosp, medians_nosp, inputs_nosp = self.get_traces()
        # unset recording
        self.net.recorded_hiddenneuron = None
        neuron_forcalc = params['resp_neurons'][-1]

        # calculate spikes
        print("$$ calculating spike times")
        tmp_params = {}
        tmp_params.update(self.sim_params)
        tmp_params['use_hicannx'] = False
        # TODO: take proper care of rounding
        self.calclayer = utils.EqualtimeLayer(
            self.network_layout['n_inputs'], self.network_layout['layer_sizes'][0],
            tmp_params, self.net.layers[0].weights.data,
            self.device, self.network_layout['n_biases'][0], 0.)
        hidden_times_calc = self.calclayer(inputs, None)
        hidden_train_calc, simtime = utils.hx_spiketrain_create(
            hidden_times_calc.detach().numpy(),
            self.net.hx_settings['single_simtime'],
            self.net.hx_settings['scale_times'],
            np.arange(params['all_batchsize']).reshape((-1, 1)).repeat(256, 1),
            np.empty_like(hidden_times_calc.detach().numpy(), dtype=int),
        )
        hidden_train_calc = hidden_train_calc[np.logical_not(np.isinf(hidden_train_calc[:, 0]))]

        # calculate volts
        print("$$ calculating voltage")
        weights = self.net.layers[0].weights.data
        tau_syn = self.sim_params['tau_syn']
        tau_m = tau_syn
        g_L = self.sim_params['g_leak']
        inputs[torch.isinf(inputs)] = 10.

        def tmp_volts(t):
            inpt = ((t - inputs) > 0) * (t - inputs) * torch.exp(-(t - inputs) / tau_syn)
            tmp = torch.matmul(inpt, weights)
            return tau_m / g_L * tmp[0, neuron_forcalc] + self.sim_params['leak']

        calc_times = torch.linspace(0., 6., 200)
        calc_volts = [tmp_volts(t) for t in calc_times]

        if params['save_individual_json']:
            print("$$$$ saving json")
            with open(self.filename + "_resp.json", "w") as f:
                json.dump({
                    'results_sp': results_sp,
                    'results_nosp': results_nosp,
                    'medians': medians,
                    'hidden_train_calc': hidden_train_calc,
                    'calc_times': calc_times,
                    'calc_volts': calc_volts,
                    'net.hx_settings': self.net.hx_settings,
                    'params': params,
                }, f, cls=NumpyEncoder)
            print("$$ saved as {self.filename}_resp.json")

        print("$$$$ plotting")
        # for weirdish reasons, we want broken axis, and the easiest seems to be the following way

        def plotit(ax, results):
            # plot recorded traces
            for i, (neuron, hidden_spikes, trace) in enumerate(results):
                if params['resp_trace_subtract_rest']:
                    volts = trace[:, 1] + np.mean(medians) - medians[i]
                else:
                    volts = trace[:, 1]
                ax.plot(trace[:, 0] / self.net.hx_settings['intrinsic_timescale'],
                        volts,
                        color=f"C{i%10}",
                        label=f"neuron {i}: {neuron}" if i < 10 else "",
                        )
            # plot calculated volts
            for i in range(params['all_batchsize']):
                ax.plot(
                    calc_times * self.hx_settings['taum'] + self.net.hx_settings['taum'] * 100 * i,
                    calc_volts, color="black")

            ax.axhline(self.sim_params['leak'], color='black')
            ax.axhline(self.sim_params['threshold'], color='black')

            # recorded spikes
            for i, (neuron, hidden_spikes, trace) in enumerate(results):
                for t in hidden_spikes[(hidden_spikes[:, 1] == neuron), 0]:
                    # print(f"rec spike at {t / self.net.hx_settings['intrinsic_timescale']}")
                    ax.axvline(t / self.net.hx_settings['intrinsic_timescale'],
                               color=f"C{i}" if i < 10 else 'grey')

            # calculated spikes
            for t in hidden_train_calc[(hidden_train_calc[:, 1] == neuron_forcalc), 0]:
                # print(f"calc spike at {t / self.net.hx_settings['intrinsic_timescale']}")
                ax.axvline(t / self.net.hx_settings['intrinsic_timescale'], color="black")

            ax.set_xlabel('[$\\mu s$]')

        # now use this function to do a broken axis
        fig = plt.figure()
        gs_main = mpl_gs.GridSpec(2, 1,
                                  height_ratios=[2.5, 1],
                                  hspace=0.3,
                                  # left=0.05, right=0.95, top=0.95, bottom=0.05
                                  )
        # raster plot
        ax = fig.add_subplot(gs_main[1, 0])
        ax.set_xlabel('neurons')
        ax.set_ylabel('spike time [$\\mu s$]')
        ax.set_ylim(0, 2 * self.hx_settings['taum'])
        # reference calc spike
        for t in hidden_train_calc[(hidden_train_calc[:, 1] == neuron_forcalc), 0]:
            if t / self.net.hx_settings['intrinsic_timescale'] < 100:
                ax.axhline(t / self.net.hx_settings['intrinsic_timescale'], color="black", alpha=0.5)
        for i, (neuron, spike_times, trace) in enumerate(results_sp):
            tmp_spikes = spike_times[:, neuron]
            for j, t in enumerate(tmp_spikes):
                if not np.isinf(t):
                    ax.plot(
                        [i + (j / params['all_batchsize']) - 0.5,
                         i + ((j + 1) / params['all_batchsize']) - 0.5],
                        [t * self.net.hx_settings['taum'], ] * 2,
                        color=f"C{i%10}",
                    )
        # axvline the relevant quadrant markers
        last_quadrant_marker = 0
        for i, (neuron, spike_times, trace) in enumerate(results_sp):
            for j in np.arange(4) * 64:
                if j > last_quadrant_marker and neuron > j:
                    ax.axvline(i - 0.5, color="black")
                    last_quadrant_marker = j
                    print(f"plotting marker {j} for {i}th neuron {neuron}")

        ax.set_xticks(range(len(params['resp_neurons'])))
        ax.set_xticklabels(params['resp_neurons'])
        ax.set_xlim((-1, len(params['resp_neurons'])))

        # full width title
        ax = fig.add_subplot(gs_main[0, 0])
        ax.set_title(f"meaned leaks: {params['resp_trace_subtract_rest']}, g_l is {g_L}\n")
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(labelbottom=True)

        # compare PSPs to spike times
        gs_aux = mpl_gs.GridSpecFromSubplotSpec(1, 2, gs_main[0, 0],
                                                # hspace=hspace
                                                )
        ax = fig.add_subplot(gs_aux[0, 0])
        plotit(ax, results_sp)
        offset = 0 * self.net.hx_settings['single_simtime'] * self.hx_settings['taum']
        ax.set_xlim(-10 + offset, 15 * self.hx_settings['taum'] + offset)
        ax.set_title('spiking enabled')

        ax = fig.add_subplot(gs_aux[0, 1])
        plotit(ax, results_nosp)
        offset = 0 * self.net.hx_settings['single_simtime'] * self.hx_settings['taum']
        ax.set_xlim(-10 + offset, 15 * self.hx_settings['taum'] + offset)
        ax.set_title('spiking disabled')

        fig.savefig(f"{self.folder}/test_responsivity.png")
        print(f"saved {self.folder}/test_responsivity.png")
        plt.close(fig)

    def test_2_quiescence(self):
        print("$$$$$$$$ test_quiescence")
        self.net.hx_backend.set_spiking(True)
        # zero weights
        weights = torch.zeros(256, 256).double()
        self.net.layers[0].weights.data = weights
        self.net.write_weights_to_hicannx()
        # create batch
        inputs = torch.zeros((params['all_batchsize'], 256))
        # emulate
        spike_times, _ = self.net(inputs)
        if torch.isnan(spike_times).sum() > 0:
            raise OSError("there was a nan spike time encountered, this is not allowed to happen")

        # eval
        num_spikes = torch.logical_not(torch.isinf(spike_times)).sum()

        self.assertEqual(
            num_spikes, 0,
            f"found {num_spikes} label spikes, where there should be none. "
            f"distribution: {torch.logical_not(torch.isinf(spike_times)).float().mean(axis=0)}"
        )

    def test_3_fireability(self):
        print("$$$$$$$$ test_fireability")
        self.net.hx_backend.set_spiking(True)
        # full weights
        weights = torch.ones(256, 256).double() * 63. / self.net.hx_settings['scale_weights']
        self.net.layers[0].weights.data = weights
        self.net.write_weights_to_hicannx()
        # create batch
        inputs = torch.zeros((params['all_batchsize'], 256))
        # emulate
        spike_times, _ = self.net(inputs)
        if torch.isnan(spike_times).sum() > 0:
            raise OSError("there was a nan spike time encountered, this is not allowed to happen")

        # eval
        num_nospikes = torch.isinf(spike_times).sum()
        # labels are more important, those first (at first assertfail it is stopped)
        if num_nospikes > 0:
            dist = torch.isinf(spike_times).float().mean(axis=0)
            self.assertFalse(
                torch.any(dist > 2. / params['all_batchsize']),
                f"found {num_nospikes} label neuorns without spikes, where there should be few, and only random"
                f"distribution: {dist}, some entries larger than 2 * 1/batch_size ({2. / params['all_batchsize']})"
            )

    def test_4_spike_time_distribution(self):
        print("$$$$$$$$ test_spike_time_distribution")
        self.net.hx_backend.set_spiking(True)

        np.random.seed(params['dist_seed'])

        # random inputs and weights
        print("$$ generating randomness")
        inputs = torch.tensor(np.random.rand(params['all_batchsize'], 256))
        weights = np.random.rand(256, 256) * 63. / self.net.hx_settings['scale_weights']
        mask = np.random.rand(256, 256)

        print("$$ getting data")
        steps = np.arange(0., 1.000001, 0.1)
        results = []
        percentiles_h, percentiles_l = [], []
        inf_rate = []
        for step in steps:
            # setting weights
            weight_tmp = torch.tensor(weights)
            weight_tmp[mask > step] = 0.
            self.net.layers[0].weights.data = weight_tmp
            self.net.write_weights_to_hicannx()

            # emulate
            with torch.no_grad():
                spike_times, _ = self.net(inputs)

            results.append(
                [step, spike_times])
            tmp = spike_times[torch.logical_not(torch.isinf(spike_times))].detach().numpy()
            percentiles_l.append(np.percentile(
                tmp if len(tmp) > 0 else [np.inf],
                [0, 25, 50, 75, 100]))
            inf_rate.append(torch.isinf(spike_times).float().mean())

        print("$$ plotting")
        percentiles_h, percentiles_l = np.array(percentiles_h), np.array(percentiles_l)
        # inf_rate_h, inf_rate_l = np.array(inf_rate_h), np.array(inf_rate_l)

        fig = plt.figure()
        gs_main = mpl_gs.GridSpec(1, 1,
                                  # height_ratios=[3, 1],
                                  hspace=0.3,
                                  # left=0.05, right=0.95, top=0.95, bottom=0.05
                                  )
        # raster plot
        print("$$$$ label")
        ax = fig.add_subplot(gs_main[0, 0])
        ax.plot(steps, percentiles_l[:, 2], color='C0')
        ax.fill_between(steps, percentiles_l[:, 1], percentiles_l[:, 3], color='C0', alpha=0.3)
        ax.fill_between(steps, percentiles_l[:, 0], percentiles_l[:, 4], color='C0', alpha=0.1)
        ax2 = ax.twinx()
        ax2.plot(steps, inf_rate, color='C1')
        ax.set_title("distribution in label layer")
        ax.set_ylim(0., 1.5)
        ax.set_xlabel("rate of weights set")
        ax.set_ylabel("distrib of spike times [taus]")
        ax2.set_ylabel("rate of inf spikes")
        ax2.set_ylim(0., 1.)

        fig.savefig(f"{self.folder}/test_distribution.png")
        print(f"saved {self.folder}/test_distribution.png")
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
