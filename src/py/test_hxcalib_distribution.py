#!python3
import datetime
import json
import matplotlib as mpl
mpl.use('agg')
import matplotlib.gridspec as mpl_gs
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from pprint import pprint
from scipy.optimize import curve_fit
import sys
import torch
import unittest
import warnings
import yaml

import pyfisch_vx_v3 as fisch
import pyhalco_hicann_dls_vx_v3 as halco
import pyhaldls_vx_v3 as haldls
import pystadls_vx_v3 as stadls
import training
import utils


# parameters that define tests
NUMBER_OF_NEURONS = 256
params = {
    'batchsize': 20,
    'neurons': range(0, 256),
    'inputs_mem': torch.tensor(range(0, 8)),
    # for double neurons we need input for both atomicNeurons, thus +128
    'inputs_syn': torch.tensor(list(range(0, 3)) + list(range(128, 131))),
    'mindiff_signal_mem': 20. / 1000.,
    'mindiff_signal_memreset': 15. / 1000.,
    'mindiff_signal_syn': -30. / 1000.,
    'repetitions': 10,
    'relativeinputs': True,
    'taumem_disable_syn_input': False,
    'target_weight_mem': 30.,
    'target_weight_syn': 60.,
}
diff_stepsize = 10
used_volt_steps = 800
debug = not True
debug_plot = True
folder_base = "debug_calib"
# plot_range_taus = (6, 8)
# plot_range_taum = (3.5, 5.5)

# plot_range_taus = (5.3, 7.5)
# plot_range_taum = (4.0, 8.0)
# plot_range_tauratio = (0.5, 1.15)

plot_range_taus = None
plot_range_taum = None
plot_range_tauratio = None


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, range):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def exponential_decay(t, tau, asymptotic, coefficient):
    return coefficient * np.exp(-t / tau) + asymptotic


def plotit(data, neurons, filename):
    all_neurons = data['all_neurons']
    info = data['hx_settings']
    lbl = f"setup{data['setup']}-calib{info['calibration'][-4:]}"
    masked = [i for i, n in all_neurons.items()
              if n['leak'] is None or n['tau_synexc'] is None or n['tau_syninh'] is None
              or n['_tau_m'] is None
              ]
    neurons = [n for n in all_neurons if n not in masked]
    print("the following neurons had at least one None: ")
    pprint({i: all_neurons[i] for i in masked})

    leaks = np.array([all_neurons[i]['leak'] for i in neurons])
    leak_stds = np.array([all_neurons[i]['leak_std'] for i in neurons])
    threshs = np.array([all_neurons[i]['threshold'] for i in neurons])
    thresh___leak = threshs - leaks
    tau_ms_alternative = np.array([all_neurons[i]['_tau_m'] for i in neurons])
    tau_synexcs = np.array([all_neurons[i]['tau_synexc'] for i in neurons])
    tau_syninhs = np.array([all_neurons[i]['tau_syninh'] for i in neurons])
    synexc_weights = np.array([all_neurons[i]['synexc_weight'] for i in neurons])
    syninh_weights = np.array([all_neurons[i]['syninh_weight'] for i in neurons])

    targets = {
        'leak': data["hx_settings"]["neuron_params"]["leak"],
        'threshold': data["hx_settings"]["neuron_params"]["threshold"],
        'taum': data["hx_settings"]["taum"],
        'taus': data['hx_settings']['taus'] if 'taus' in data['hx_settings'] else data['hx_settings']['taum'],
        'syn_weights': -len(params['inputs_syn']) / 2 * params['target_weight_syn'] / 1000.,
    }
    targets['tauratio'] = targets['taum'] / targets['taus']
    targets["th_min_le"] = targets["threshold"] - targets["leak"]

    # one plot
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 9))
        # voltages
        lines = axes[0].plot(neurons, threshs, label="thresh",
                             linestyle='None', marker='x', color='C0')
        lines += axes[0].plot(neurons, leaks, label="leak",
                              linestyle='None', marker='x', color='C1')
        axes[0].axhline(targets['leak'], color='C1')
        axes[0].axhline(targets['threshold'], color='C0')
        axes[0].legend(lines, [l.get_label() for l in lines])
        # voltages std
        lines = axes[1].plot(neurons, leak_stds, label="leak std",
                             linestyle='None', marker='x', color='C1')
        lines += axes[1].plot(neurons, thresh___leak, label="thresh-leak",
                              linestyle='None', marker='x', color='C2')
        axes[1].axhline(targets['threshold'] - targets['leak'], color='C2')
        axes[1].legend(lines, [l.get_label() for l in lines])
        # taus
        lines = axes[2].plot(neurons, tau_synexcs, label="tausynexc",
                             linestyle='None', marker='x', color='C4')
        lines += axes[2].plot(neurons, tau_syninhs, label="tausyninh",
                              linestyle='None', marker='x', color='C5')
        lines += axes[2].plot(neurons, tau_ms_alternative, label="taum",
                              linestyle='None', marker='x', color='C6')
        axes[2].axhline(targets['taus'], color='C4')
        axes[2].axhline(targets['taus'], ls="dashed", color='C5')
        axes[2].axhline(targets['taum'], ls="dotted", color='C6')
        axes[2].legend(lines, [l.get_label() for l in lines])

        lines = axes[3].plot(neurons, synexc_weights, label="syn weight exc",
                             linestyle="None", marker="x", color='C7')
        lines += axes[3].plot(neurons, syninh_weights, label="syn weight inh",
                              linestyle="None", marker="x", color='C8')
        axes[3].axhline(targets['syn_weights'], color='C7')
        axes[3].axhline(targets['syn_weights'], ls='dashed', color='C8')
        axes[3].legend(lines, [l.get_label() for l in lines])

        axes[0].set_title(lbl)
        fig.tight_layout()
        fig.savefig(filename + ".png")
        plt.close(fig)
        datas = [
            (leaks, "$E_l$", "$[a.u.]$", targets["leak"]),
            (threshs, "$\\theta$", "$[a.u.]$", targets["threshold"]),
            (thresh___leak, "$E_l - \\theta$", "$[a.u.]$", targets["th_min_le"]),
            (tau_synexcs, "$\\tau_{s, exc}$", "$[\\mu s]$", targets["taus"], plot_range_taus),
            (tau_ms_alternative, "$\\tau_m$ with reset", "$[\\mu s]$", targets["taum"], plot_range_taum),
            (tau_ms_alternative / tau_synexcs, "$\\tau_m/\\tau_{s, exc}$", "$[1]$",
             targets["tauratio"], plot_range_tauratio),
            (tau_syninhs, "$\\tau_{s, inh}$", "$[\\mu s]$", targets["taus"], plot_range_taus),
            None,  # (tau_ms, "$\\tau_m$", "$[\\mu s]$", targets["taum"], plot_range_taum),
            (tau_ms_alternative / tau_syninhs, "$\\tau_m/\\tau_{s, inh}$", "$[1]$",
             targets["tauratio"], plot_range_tauratio),
            (synexc_weights, "synexc_weights", "[LSB]", targets['syn_weights']),
            (syninh_weights, "syninh_weights", "[LSB]", targets['syn_weights']),
            None,
        ]
        fig, axes = plt.subplots(4, 3)
        for i, (data, ax) in enumerate(zip(datas, axes.flatten())):
            if data is None:
                ax.axis('off')
                continue
            ax.hist(data[0], label=lbl, rwidth=0.8)
            ax.set_xlabel(data[1] + " " + data[2])
            if len(data) > 3 and data[3] is not None:
                ax.axvline(data[3], color="C3")
            if len(data) > 4 and data[4] is not None:
                ax.set_xlim(data[4])
        axes[-1, 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.55),)
        fig.tight_layout()
        fig.savefig(filename + "_hist.png")
        plt.close(fig)


def plotmultiple(lst):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, axes = plt.subplots(3, 3)

        for i, data in enumerate(lst):
            all_neurons = data['all_neurons']
            info = data['hx_settings']
            lbl = f"setup{data['setup']}-calib{info['calibration'][-4:]}"
            masked = [i for i, n in all_neurons.items()
                      if n['leak'] is None or n['tau_synexc'] is None or n['tau_syninh'] is None
                      ]
            neurons = [n for n in all_neurons if n not in masked]
            print("the following neurons had at least one None: ")
            pprint({i: all_neurons[i] for i in masked})

            leaks = np.array([all_neurons[i]['leak'] for i in neurons])
            threshs = np.array([all_neurons[i]['threshold'] for i in neurons])
            thresh___leak = threshs - leaks
            tau_ms_alternative = np.array([all_neurons[i]['_tau_m'] for i in neurons])
            tau_synexcs = np.array([all_neurons[i]['tau_synexc'] for i in neurons])
            tau_syninhs = np.array([all_neurons[i]['tau_syninh'] for i in neurons])
            targets = {
                'leak': data["hx_settings"]["neuron_params"]["leak"],
                'threshold': data["hx_settings"]["neuron_params"]["threshold"],
                'taum': data["hx_settings"]["taum"],
                'taus': data['hx_settings']['taus'] if 'taus' in data['hx_settings'] else data['hx_settings']['taum']
            }
            targets['tauratio'] = targets['taum'] / targets['taus']
            targets["th_min_le"] = targets["threshold"] - targets["leak"]
            # two plot
            datas = [
                (leaks, "$E_l$", "$[a.u.]$", targets["leak"]),
                (threshs, "$\\theta$", "$[a.u.]$", targets["threshold"]),
                (thresh___leak, "$E_l - \\theta$", "$[a.u.]$", targets["th_min_le"]),
                (tau_synexcs, "$\\tau_{s, exc}$", "$[\\mu s]$", targets["taus"], plot_range_taus),
                None,  # (tau_ms, "$\\tau_m$", "$[\\mu s]$", targets["taum"], plot_range_taum),
                (tau_ms_alternative / tau_synexcs, "$\\tau_m/\\tau_{s, exc}$", "$[1]$",
                 targets["tauratio"], plot_range_tauratio),
                (tau_syninhs, "$\\tau_{s, inh}$", "$[\\mu s]$", targets["taus"], plot_range_taus),
                (tau_ms_alternative, "$\\tau_m$ with reset", "$[\\mu s]$", targets["taum"], plot_range_taum),
                (tau_ms_alternative / tau_syninhs, "$\\tau_m/\\tau_{s, inh}$", "$[1]$",
                 targets["tauratio"], plot_range_tauratio),
            ]
            for i, (data, ax) in enumerate(zip(datas, axes.flatten())):
                if data is None:
                    ax.axis('off')
                    continue
                ax.hist(data[0], label=lbl, rwidth=0.8, alpha=0.5)
                ax.set_xlabel(data[1] + " " + data[2])
                if len(data) > 3 and data[3] is not None:
                    ax.axvline(data[3], color="C1")
                if len(data) > 4 and data[4] is not None:
                    ax.set_xlim(data[4])

        axes[2, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),)
        fig.tight_layout()
        fig.savefig("comparison_hist.png")
        plt.close(fig)


class CalibParameterDistribution(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("\n$$$$$$$$ initialisation")
        self.state = "init"
        if os.environ.get('SLURM_HARDWARE_LICENSES') is None:
            raise OSError("This test has to be executed with hardware access, "
                          "tested with 'SLURM_HARDWARE_LICENSES' variable")
        with open('py/hx_settings.yaml') as f:
            hx_settings = yaml.load(f, Loader=yaml.SafeLoader)
        self.hx_setup_no = os.environ.get('SLURM_HARDWARE_LICENSES')

        if self.hx_setup_no not in hx_settings:
            raise OSError(f"Setup no {self.hx_setup_no} is not described in hx settings file, only {hx_settings.keys()}")
        self.hx_settings = hx_settings[self.hx_setup_no]
        print(f"$$ Testing on setup {self.hx_setup_no} with the following settings (others derived, see Net.init)")
        pprint(self.hx_settings, indent=5)

        # outside params
        print("$$ using params:")
        pprint(params, indent=5)

        self.folder = f"{folder_base}_{self.hx_setup_no}"
        self.filename = f"{self.folder}/plot_test_calibdistr_{self.hx_setup_no}_"\
            f"{datetime.datetime.now():%Y%m%d_%H%M%S}"

        self.device = torch.device('cpu')

        print(self.hx_settings)
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

        self.all_neurons = {neuron: {} for neuron in params['neurons']}

        # create net
        self.net = training.Net(self.network_layout, self.sim_params, self.device)
        print("setting net shift label neurons zero")
        self.net.hx_settings['label_neurons_shift'] = 0
        self.sim_params = self.hx_settings['neuron_params']

        if debug_plot and not osp.isdir(self.folder):
            os.mkdir(self.folder)

    @classmethod
    def tearDownClass(self):
        # pprint(self.all_neurons)
        with open(self.filename + ".json", 'w') as f:
            data = {"hx_settings": self.hx_settings,
                    "params": params,
                    "setup": self.hx_setup_no,
                    "all_neurons": self.all_neurons}
            json.dump(data, f, cls=CustomJSONEncoder)

        keys = list(self.all_neurons.values())[0].keys()
        for key in keys:
            lst = [i[key] for i in self.all_neurons.values() if i[key] is not None]
            print(f"{key} has mean {np.mean(lst):.3f}, std {np.std(lst):.3f}, "
                  f"relative error {np.std(lst) / np.mean(lst):.3f}, "
                  f"median {np.median(lst):.3f}")
        if 'threshold' in keys and 'leak' in keys:
            lst = [i['threshold'] - i['leak'] for i in self.all_neurons.values()
                   if i['threshold'] is not None and i['leak'] is not None]
            print(f"\nthresh-leak has mean {np.mean(lst):.3f}, std {np.std(lst):.3f}, "
                  f"relative error {np.std(lst) / np.mean(lst):.3f}, "
                  "percentile [0 10 50 90 100]: [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(
                      *np.percentile(lst, [0, 10, 50, 90, 100])))

        if len(sys.argv) == 1 or sys.argv[1] == "CalibParameterDistribution.manipulate_all":
            plotit(data, params['neurons'], self.filename)

    def run(self, result=None):
        """ Stop after first error """
        if not result.errors:
            super(CalibParameterDistribution, self).run(result)

    def help_dissecttrace(self, neuron, volts_data, times, signal_finder):
        volts = []

        if debug_plot:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                fig, ax = plt.subplots(1, 1)
                ax.plot(times, volts_data, label="volts")
                ax.plot(times[::diff_stepsize][:-1], np.diff(volts_data[::diff_stepsize]) * 20, label="diff")
                ax.plot(times[::diff_stepsize][:-1], signal_finder(volts_data), label="signal")
                ax.legend()
                fig.savefig(f"{self.folder}/debug_calib_distr_trace_neuron_{neuron}_{self.state}.png")
                plt.close(fig)

        while True:
            signals = signal_finder(volts_data)
            if len(volts_data) == 0:
                break
            signal = np.argmax(signals)
            initial_index = (signal - 2) * diff_stepsize
            if not signals[signal] or len(volts_data) - initial_index <= used_volt_steps:
                break
            if len(volts) == 0:
                times_trimmed = times[initial_index:initial_index + used_volt_steps] - times[initial_index]
                # leak_trimmed = leakdata[initial_index:initial_index + used_volt_steps, 0]
            # if len(psps) == 1:
            #     leak = np.mean(volts_data[initial_index - int(used_volt_steps / 2):initial_index])
            volts.append(volts_data[initial_index:initial_index + used_volt_steps])
            volts_data = volts_data[initial_index + used_volt_steps:]
        if len(volts) == 0:
            raise IOError("The voltage trace given to dissect did not have a signal, or too close to the end")
        volts = np.array(volts)
        return volts, times_trimmed

    def help_fitfunction(self, neuron, volts, times_trimmed, fn, fit_initial):
        try:
            volts_mean = np.mean(volts, axis=0)
        except Exception:
            if debug:
                print(f"volts has lengths {[len(i) for i in volts]}, retrying without zero-length entries")
            volts = [v for v in volts if len(v) > 0]
            try:
                volts_mean = np.mean(volts, axis=0)
            except Exception:
                print(f"volts has lengths {[len(i) for i in volts]}")
                raise
        volts_for_fit = volts_mean[diff_stepsize * 4:]
        # volt_std = np.std(volts, axis=0) + 1e-3
        times_for_fit = times_trimmed[diff_stepsize * 4:]

        if debug:
            print("fitting")
        fit_result = curve_fit(
            fn,  # function
            times_for_fit, volts_for_fit,  # x and y
            p0=fit_initial,
            # sigma=volt_std,
        )
        if debug:
            print("fit started with")
            print(fit_initial)
            print("result:")
            print(fit_result[0])
            if np.isinf(fit_result[1]).sum() < 25:
                print("covariance")
                print(fit_result[1])

        if debug_plot:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                fig, ax = plt.subplots()
                for i, psp in enumerate(volts):
                    ax.plot(times_trimmed, psp, alpha=0.1)

                ax.plot(times_trimmed, volts_mean, label="mean", color="black")

                volts_initial = fn(times_for_fit, *fit_initial)
                volts_fitted = fn(times_for_fit, *fit_result[0])
                ax.plot(times_for_fit, volts_initial, label="initial for fit", color="C0")
                ax.plot(times_for_fit, volts_fitted, label="fitted", color="C1")

                ax.axvline(times_for_fit[0], ymax=0.2, color='black')
                ax.axvline(times_for_fit[-1], ymax=0.2, color='black')

                ax.set_title(f"{self.state} neuron {neuron}, time constant {fit_result[0][0]:.3f}μs")

                ax.legend()

                fig.savefig(f"{self.folder}/debug_calib_distr_fit_neuron_{neuron}_{self.state}.png")
                plt.close(fig)
        return fit_result

    def help_gettrace(self, net, neuron, target_weight, presyns, checkspikes=None, record_target="membrane"):
        # set weights
        weights = torch.zeros(256, 256).double()
        if params['relativeinputs']:
            weights[(presyns + neuron) % NUMBER_OF_NEURONS, neuron] = target_weight
        else:
            weights[(presyns) % NUMBER_OF_NEURONS, neuron] = target_weight

        net.layers[0].weights.data = weights
        net.write_weights_to_hicannx()

        # set recording
        net.hx_record_neuron = neuron
        net.hx_record_target = record_target

        # create batch
        inputs = torch.ones((params['batchsize'], 256)) * float('inf')
        if params['relativeinputs']:
            inputs[:, (presyns + neuron) % NUMBER_OF_NEURONS] = 7.
        else:
            inputs[:, (presyns) % NUMBER_OF_NEURONS] = 7.

        # # emulate for leak
        # label_times, [hidden_times] = net(
        #     torch.ones_like(inputs) * float('inf'))
        # # handle traces
        # leak_noinput = np.median(net.trace)
        # leakdata = net.trace
        # np.save(f"debug_calib_distr_inh_n{neuron}l.npy", leakdata)
        # self.all_neurons[neuron]['leak_noinput'] = leak_noinput
        # print(f"neuron {neuron} has leak_noinput of {leak_noinput}")

        # emulate for PSPs
        net.hx_settings['single_simtime'] = 20
        spiketimes, _ = net(inputs)
        # np.save(f"debug_calib_distr_inh_n{neuron}.npy", net.trace)

        trace = net.trace
        times = trace[:, 0] * 1e6
        volts_data = trace[:, 1]

        if debug_plot:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                fig, ax = plt.subplots(1, 1)
                ax.plot(times, volts_data, label="volts")
                # ax.plot(times[::diff_stepsize][:-1], np.diff(volts_data[::diff_stepsize]) * 20, label="diff")
                # ax.plot(times[::diff_stepsize][:-1], signal_finder(volts_data) * 20, label="signal")
                ax.legend()
                fig.savefig(f"{self.folder}/debug_calib_distr_trace_neuron_{neuron}_{self.state}.png")
                plt.close(fig)

        return (times, volts_data), "" if checkspikes is None else checkspikes(spiketimes)

    def manipulate_all(self):
        print("\n$$$$$$$$ manipulate_all")
        v_thresh_alpha = 40
        v_leak_alpha = 40
        tau_syn_alpha = -5  # -5e6
        tau_m_alpha = -4  # -4e6
        syn_weight_alpha = -100
        v_leak_target = self.hx_settings['neuron_params']['leak']
        v_thresh_target = self.hx_settings['neuron_params']['threshold'] - self.hx_settings['neuron_params']['leak']
        tau_syn_target = self.hx_settings['taus'] if 'taus' in self.hx_settings else self.hx_settings['taum']
        tau_m_target = self.hx_settings['taum']
        syn_weight_target = -len(params['inputs_syn']) / 2 * params['target_weight_syn'] / 1000.

        if osp.isfile(f"{self.folder}/postpostcalib.json"):
            with open(f"{self.folder}/postpostcalib.json", 'r') as f:
                history = json.load(f)
        else:
            history = {}
        for key in ['threshold_distance', 'leak', 'tau_syn_exc', 'tau_syn_inh', 'tau_m', 'synapse_bias']:
            if key not in history:
                history[key] = []

        if 'calibration_custom' not in self.net.hx_settings:
            self.net.hx_settings['calibration_custom'] = osp.splitext(
                self.net.hx_settings['calibration'])[0] + "_custom.json"
            calibration_custom = {}
            calib = np.load(self.net.hx_settings['calibration'], allow_pickle=True)['neuron'].item()

            for neuron_coord in halco.iter_all(halco.AtomicNeuronOnDLS):
                neuron_id = int(neuron_coord.toEnum())
                neuron_config = calib.neurons[neuron_coord]
                tmp = {
                    # 'i_syn_exc_gm': neuron_config.excitatory_input.i_bias_gm,
                    # 'i_syn_inh_gm': neuron_config.inhibitory_input.i_bias_gm,
                    'tau_m': neuron_config.leak.i_bias,
                    'tau_syn_exc': neuron_config.excitatory_input.i_bias_tau,
                    'tau_syn_inh': neuron_config.inhibitory_input.i_bias_tau,
                    'v_leak': neuron_config.leak.v_leak,
                    'v_reset': neuron_config.reset.v_reset,
                    'v_thresh': neuron_config.threshold.v_threshold,
                }
                for k, v in tmp.items():
                    if k not in calibration_custom:
                        calibration_custom[k] = np.zeros(512, dtype=int)
                    calibration_custom[k][neuron_id] = v.value()

            # calibration_custom['synapse_bias'] = np.ones(4, dtype=int) * 800

            # print(calibration_custom['v_reset'])
            # calibration_custom['v_reset'] = np.clip(calibration_custom['v_reset'] - 200, 0, 1022)

            with open(self.net.hx_settings['calibration_custom'], 'w') as f:
                json.dump(calibration_custom, f, cls=CustomJSONEncoder)

            print("******************************************************************************************")
            print("The relevant values have been extracted from the calib and are saved in "
                  f"{self.net.hx_settings['calibration_custom']} This needs to be put into the hx_settings.yaml")
            print("******************************************************************************************")
            sys.exit()

        # tests
        self.test_mem()
        self.test_memreset()
        self.test_synexc()
        self.test_syninh()

        # tau_syn_exc
        # restructure for specific neurons
        values = np.full(512, tau_syn_target)
        for i in self.all_neurons:
            real_idx = int(i) * 2
            values[real_idx] = self.all_neurons[i]['tau_synexc']
        capmem_change = np.clip((tau_syn_alpha * (tau_syn_target - values)).astype(np.int),
                                -10, 10)
        previous = self.net.hx_backend.calib_values['tau_syn_exc']
        final_value = np.clip(self.net.hx_backend.calib_values['tau_syn_exc'] + capmem_change, 0, 1022)
        self.net.hx_backend.calib_values['tau_syn_exc'] = final_value
        update = final_value - previous
        difference = values - tau_syn_target
        history['tau_syn_exc'].append({
            'difference': difference,
            'update': update,
        })

        # tau_syn_inh
        # restructure for specific neurons
        values = np.full(512, tau_syn_target)
        for i in self.all_neurons:
            real_idx = int(i) * 2
            values[real_idx] = self.all_neurons[i]['tau_syninh']
        capmem_change = np.clip((tau_syn_alpha * (tau_syn_target - values)).astype(np.int),
                                -10, 10)
        previous = self.net.hx_backend.calib_values['tau_syn_inh']
        final_value = np.clip(self.net.hx_backend.calib_values['tau_syn_inh'] + capmem_change, 0, 1022)
        self.net.hx_backend.calib_values['tau_syn_inh'] = final_value
        update = final_value - previous
        difference = values - tau_syn_target
        history['tau_syn_inh'].append({
            'difference': difference,
            'update': update,
        })

        # tau_m
        # restructure for specific neurons
        values = np.full(512, tau_m_target)
        for i in self.all_neurons:
            real_idx = int(i) * 2
            values[real_idx] = self.all_neurons[i]['_tau_m']
        capmem_change = np.clip((tau_m_alpha * (tau_m_target - values)).astype(np.int),
                                -10, 10)
        previous = self.net.hx_backend.calib_values['tau_m']
        final_value = np.clip(self.net.hx_backend.calib_values['tau_m'] + capmem_change, 0, 1022)
        self.net.hx_backend.calib_values['tau_m'] = final_value
        update = final_value - previous
        difference = values - tau_m_target
        history['tau_m'].append({
            'difference': difference,
            'update': update,
        })

        # leak
        # self.test_mem()
        # restructure for specific neurons
        values = np.full(512, v_leak_target)
        for i in self.all_neurons:
            real_idx = int(i) * 2
            values[real_idx] = self.all_neurons[i]['leak']
        capmem_change = np.clip((v_leak_alpha * (v_leak_target - values)).astype(np.int),
                                -10, 10)
        previous = self.net.hx_backend.calib_values['v_leak']
        final_value = np.clip(self.net.hx_backend.calib_values['v_leak'] + capmem_change, 0, 1022)
        self.net.hx_backend.calib_values['v_leak'] = final_value
        update = final_value - previous
        difference = values - v_leak_target
        history['leak'].append({
            'difference': difference,
            'update': update,
        })

        # threshold
        # self.test_mem()
        # restructure for specific neurons
        values = np.full(512, v_thresh_target)
        for i in self.all_neurons:
            real_idx = int(i) * 2
            values[real_idx] = self.all_neurons[i]['threshold'] - self.all_neurons[i]['leak']
        capmem_change = np.clip((v_thresh_alpha * (v_thresh_target - values)).astype(np.int),
                                -10, 10)
        previous = self.net.hx_backend.calib_values['v_thresh']
        final_value = np.clip(self.net.hx_backend.calib_values['v_thresh'] + capmem_change, 0, 1022)
        self.net.hx_backend.calib_values['v_thresh'] = final_value
        update = final_value - previous
        difference = values - v_thresh_target
        history['threshold_distance'].append({
            'difference': difference,
            'update': update,
        })

        # synapse bias
        values = np.full(512, syn_weight_target)
        for i in self.all_neurons:
            real_idx = int(i) * 2
            values[real_idx] = self.all_neurons[i]['synexc_weight']
            values[real_idx + 1] = self.all_neurons[i]['syninh_weight']  # dirty hack to accomodate inhibition
        capmem_change = np.clip(
            (syn_weight_alpha * (syn_weight_target - values)).reshape((4, -1)).mean(axis=1).astype(np.int),
            -10, 10)
        previous = self.net.hx_backend.calib_values['synapse_bias']
        final_value = np.clip(self.net.hx_backend.calib_values['synapse_bias'] + capmem_change, 0, 1022)
        self.net.hx_backend.calib_values['synapse_bias'] = final_value
        update = final_value - previous
        difference = values - syn_weight_target
        history['synapse_bias'].append({
            'difference': difference,
            'update': update,
        })

        print("## saving updated calib values")
        with open(self.net.hx_settings['calibration_custom'], 'w') as f:
            json.dump(self.net.hx_backend.calib_values, f, cls=CustomJSONEncoder)

        print("## saving postcalib history")
        with open(f"{self.folder}/postpostcalib.json", 'w') as f:
            json.dump(history, f, cls=CustomJSONEncoder)

        fig, axes = plt.subplots(2, len(history), figsize=(10, 8))
        percentiles = [0, 10, 50, 90, 100]
        for i, (key, dat) in enumerate(history.items()):
            color = f"C{i}"
            # axes[0, i].plot([j['difference'] for j in dat], color="C0", ls='solid')
            # axes[1, i].plot([j['update'] for j in dat], color="C0", ls='solid')
            tmp = np.percentile([j['difference'] for j in dat], percentiles, axis=1).transpose()
            xs = range(len(tmp))
            axes[0, i].plot(xs, tmp[:, 2], color=color, ls='solid')
            axes[0, i].fill_between(xs, tmp[:, 1], tmp[:, 3], alpha=0.5, color=color)
            axes[0, i].fill_between(xs, tmp[:, 0], tmp[:, 4], alpha=0.2, color=color)
            tmp = np.percentile([j['update'] for j in dat], percentiles, axis=1).transpose()
            xs = range(len(tmp))
            axes[1, i].plot(xs, tmp[:, 2], color=color, ls='solid')
            axes[1, i].fill_between(xs, tmp[:, 1], tmp[:, 3], alpha=0.5, color=color)
            axes[1, i].fill_between(xs, tmp[:, 0], tmp[:, 4], alpha=0.2, color=color)

            axes[0, i].set_ylabel(key)
        axes[0, int(len(history) / 2)].set_title("difference")
        axes[1, int(len(history) / 2)].set_title("update")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        axes[0, int(len(history) / 2)].set_title(f"difference\n{percentiles}%")
        fig.savefig(f"{self.folder}/postpostcalib.png")
        plt.close(fig)

    def manipulate_thresh(self):
        raise NotImplementedError
        print("\n$$$$$$$$ manipulate_thresh")
        v_thresh_alpha = 0.4
        v_thresh_target = self.hx_settings['neuron_params']['threshold'] - self.hx_settings['neuron_params']['leak']

        if osp.isfile(f"{self.folder}/postpostcalib.json"):
            with open(f"{self.folder}/postpostcalib.json", 'r') as f:
                history = json.load(f)
        else:
            history = {'threshold_distance': []}

        # meassure
        self.test_mem()
        # restructure for specific neurons
        interval_thre_leak = np.full(512, v_thresh_target)
        for i in self.all_neurons:
            real_idx = int(i) * 2
            interval_thre_leak[real_idx] = self.all_neurons[i]['threshold'] - self.all_neurons[i]['leak']
        capmem_change = np.clip((v_thresh_alpha * (v_thresh_target - interval_thre_leak)).astype(np.int),
                                -10, 10)
        print(f"new change has max {capmem_change.max()} and min {capmem_change.min()}, "
              f"unequal zero: {(capmem_change != 0).sum()}")
        fn = self.hx_settings['calibration'] + "/calibration_v_thresh_change.npy"
        if osp.isfile(fn):
            previous = np.load(fn)
            final_change = np.array([*previous, previous[-1] + capmem_change])
        else:
            final_change = np.array([capmem_change])
        final_change[-1] = np.clip(final_change[-1],
                                   0 - self.net.blackbox.calib_data['v_thresh'],
                                   1022 - self.net.blackbox.calib_data['v_thresh'])
        np.save(fn, final_change)

        update = final_change[-1] if len(final_change) == 1 else final_change[-1] - final_change[-2]
        difference = interval_thre_leak - v_thresh_target
        history['threshold_distance'].append({
            'l1': np.abs(difference).sum() / len(params['neurons']),
            'l2': np.sqrt((difference ** 2).sum()) / len(params['neurons']),
            'update_l1': np.abs(update).sum() / len(params['neurons']),
            'update_l2': np.sqrt((update ** 2).sum()) / len(params['neurons']),
        })

        with open(f"{self.folder}/postpostcalib.json", 'w') as f:
            json.dump(history, f, cls=CustomJSONEncoder)

        fig, ax = plt.subplots(1, 1)
        lgds = []
        for i, (key, dat) in enumerate(history.items()):
            tmp = ax.plot([j['l1'] for j in dat], color=f"C{i}", label=f"{key} difference l1", ls='solid')
            lgds.append(tmp)
            tmp1 = ax.plot([j['l2'] for j in dat], color=f"C{i}", label=f"{key} difference l2", ls='dotted')
            tmp2 = ax.plot([j['update_l1'] for j in dat], color=f"C{i}", label=f"{key} update l1", ls='dashed')
            tmp3 = ax.plot([j['update_l2'] for j in dat], color=f"C{i}", label=f"{key} update l2", ls='dashdot')
            if i == 0:
                lgds.extend([tmp1, tmp2, tmp3])
        ax.set_yscale('log')
        ax.legend(lgds)
        fig.savefig(f"{self.folder}/postpostcalib.png")
        plt.close(fig)

        # print(interval_thre_leak)
        # print(capmem_change)
        np.save(fn, final_change)
        print(f"saved as{fn}")

    def old_test_mem(self):
        # ######## leak, threshold, taum
        print("\n$$$$$$$$ test_mem")
        self.state = "mem"
        self.net.hx_backend.set_spiking(True)
        for i, neuron in enumerate(params['neurons']):
            if debug:
                print(f"neuron {neuron} starting")
            print(f"\r{i+1}/{len(params['neurons'])} "
                  f"({min(params['neurons'])}...{neuron}...{max(params['neurons'])})", end='')

            # # ######## syn ex
            # os.environ['HXCALIB_SPECIAL_READOUT'] = "exc_synin"
            # # tracedata, spikecheck = self.help_gettrace(
            # #     self.net, neuron, target_weight, params['inputs'],
            # #     lambda x: "" if torch.all(torch.isinf(x)) else f"spikes happened {x}")
            if debug:
                print(f"neuron {neuron} getting trace")
            repetitions = params['repetitions']
            done = False
            while not done:
                try:
                    (times, volts_data), spikecheck = self.help_gettrace(
                        self.net, neuron,
                        params['target_weight_mem'] / self.net.hx_settings['scale_weights'],
                        params['inputs_mem'],
                        # (lambda x: "" if (torch.isinf(x[:, neuron])).float().mean() < 0.3
                        #  else f"too few spikes happened {x[:, neuron]}")
                    )
                    if spikecheck:
                        print(f"neuron {neuron} gives checkspikes {spikecheck}")
                        self.all_neurons[neuron]['threshold'] = None
                        fit_result = [[None] * 3]
                    else:
                        # get threshold
                        self.all_neurons[neuron]['threshold'] = np.max(volts_data)

                        volts_mean = volts_data.mean()
                        if debug:
                            print(f"neuron {neuron} dissecting")
                        volts, times_trimmed = self.help_dissecttrace(
                            neuron, volts_data, times, lambda x: np.logical_and(
                                np.diff(x[::diff_stepsize]) > params['mindiff_signal_mem'],
                                x[::diff_stepsize][:-1] < volts_mean),
                        )

                        fit_result = self.help_fitfunction(neuron, volts, times_trimmed,
                                                           exponential_decay, [4, 0.400, -0.200])
                    done = True
                except Exception:
                    repetitions -= 1
                    print(f"Exception, retrying neuron {neuron} for {repetitions} more times")
                    if repetitions == 0:
                        raise

            self.all_neurons[neuron]['tau_m'] = fit_result[0][0]
            self.all_neurons[neuron]['leak'] = fit_result[0][1]
            self.all_neurons[neuron]['mem_difftoreset'] = fit_result[0][2]

    def test_mem(self):
        # ######## leak, threshold
        print("\n$$$$$$$$ test_mem")
        self.state = "mem"
        self.net.hx_backend.set_spiking(True)

        all_th = []
        all_leak = []
        for i, neuron in enumerate(params['neurons']):
            if debug:
                print(f"neuron {neuron} starting")
            print(f"\r{i+1}/{len(params['neurons'])} "
                  f"({min(params['neurons'])}...{neuron}...{max(params['neurons'])})", end='')

            if debug:
                print(f"neuron {neuron} getting trace")
            repetitions = params['repetitions']
            done = False
            while not done:
                try:
                    # get threshold
                    self.state = "mem_thre"
                    (times, volts_data), spikecheck = self.help_gettrace(
                        self.net, neuron, params['target_weight_mem'] / self.net.hx_settings['scale_weights'],
                        params['inputs_mem'],
                    )
                    self.all_neurons[neuron]['threshold'] = np.max(volts_data)
                    all_th.append([np.median(volts_data), np.mean(volts_data), np.std(volts_data)])
                    # get leak
                    self.state = "mem_leak"
                    (times, volts_data), spikecheck = self.help_gettrace(
                        self.net, neuron, 0, torch.tensor(np.empty((0), dtype=int))
                    )
                    self.all_neurons[neuron]['leak'] = np.median(volts_data)
                    self.all_neurons[neuron]['leak_std'] = np.std(volts_data)
                    all_leak.append([np.median(volts_data), np.mean(volts_data), np.std(volts_data)])
                    done = True
                except Exception:
                    repetitions -= 1
                    print(f"Exception, retrying neuron {neuron} for {repetitions} more times")
                    if repetitions == 0:
                        raise

            self.all_neurons[neuron]['tau_m'] = None
            self.all_neurons[neuron]['mem_difftoreset'] = None

        if (np.array([v['leak_std'] for v in self.all_neurons.values()]) > 0.1).sum() > 0:
            print("## There are neurons with unusually high std in the leak, check them out to ensure there is "
                  "no leak-over-threshold scenario\nneurons:")
            print([(k, v['leak_std']) for k, v in self.all_neurons.items() if v['leak_std'] > 0.1])
        np.save("all_th.npy", np.array(all_th))
        np.save("all_leak.npy", np.array(all_leak))
        # print(np.mean(all_th, axis=0), np.min(all_th, axis=0), np.max(all_th, axis=0))
        # print(np.mean(all_leak, axis=0), np.min(all_leak, axis=0), np.max(all_leak, axis=0))

    def test_synexc(self):
        # ######## leak, threshold, taum
        print("\n$$$$$$$$ test_synexc")
        self.state = "synexc"
        self.net.hx_backend.set_spiking(False)
        for i, neuron in enumerate(params['neurons']):
            if debug:
                print(f"neuron {neuron} starting")
            print(f"\r{i+1}/{len(params['neurons'])} "
                  f"({min(params['neurons'])}...{neuron}...{max(params['neurons'])})", end='')

            if debug:
                print(f"neuron {neuron} getting trace")
            (times, volts_data), spikecheck = self.help_gettrace(
                self.net, neuron,
                params['target_weight_syn'] / self.net.hx_settings['scale_weights'],
                params['inputs_syn'],
                lambda x: "", record_target="exc_synin")
            # if not torch.all(torch.isinf(x[:, neuron])) else f"spikes happened {x[:, neuron]}")
            if spikecheck:
                print(f"neuron {neuron} gives checkspikes {spikecheck}")
                fit_result = [[None] * 3]
            else:
                # volts_mean = volts_data.mean()
                if debug:
                    print(f"neuron {neuron} dissecting")
                volts, times_trimmed = self.help_dissecttrace(
                    neuron, volts_data, times,
                    lambda x: np.diff(x[::diff_stepsize]) < params['mindiff_signal_syn'],
                )

                fit_result = self.help_fitfunction(neuron, volts, times_trimmed,
                                                   exponential_decay, [4, 0.490, -0.050])

            self.all_neurons[neuron]['tau_synexc'] = fit_result[0][0]
            self.all_neurons[neuron]['synexc_rest'] = fit_result[0][1]
            self.all_neurons[neuron]['synexc_weight'] = fit_result[0][2]

    def test_syninh(self):
        # ######## leak, threshold, taum
        print("\n$$$$$$$$ test_syninh")
        self.state = "syninh"
        self.net.hx_backend.set_spiking(False)
        for i, neuron in enumerate(params['neurons']):
            if debug:
                print(f"neuron {neuron} starting")
            print(f"\r{i+1}/{len(params['neurons'])} "
                  f"({min(params['neurons'])}...{neuron}...{max(params['neurons'])})", end='')

            if debug:
                print(f"neuron {neuron} getting trace")
            (times, volts_data), spikecheck = self.help_gettrace(
                self.net, neuron,
                -params['target_weight_syn'] / self.net.hx_settings['scale_weights'],
                params['inputs_syn'],
                lambda x: "", record_target="inh_synin")
            # if torch.all(torch.isinf(x[:, neuron])) else f"spikes happened {x[:, neuron]}")
            if spikecheck:
                print(f"neuron {neuron} gives checkspikes {spikecheck}")
                fit_result = [[None] * 3]
            else:
                # volts_mean = volts_data.mean()
                if debug:
                    print(f"neuron {neuron} dissecting")
                volts, times_trimmed = self.help_dissecttrace(
                    neuron, volts_data, times,
                    lambda x: np.diff(x[::diff_stepsize]) < params['mindiff_signal_syn'],
                )

                fit_result = self.help_fitfunction(neuron, volts, times_trimmed,
                                                   exponential_decay, [4, 0.500, -0.050])

            self.all_neurons[neuron]['tau_syninh'] = fit_result[0][0]
            self.all_neurons[neuron]['syninh_rest'] = fit_result[0][1]
            self.all_neurons[neuron]['syninh_weight'] = fit_result[0][2]

    def test_memreset(self):
        # ######## leak, threshold, taum
        print("\n$$$$$$$$ test_memreset")
        self.state = "memreset"

        # print("*******************************SUBTRACTING 100 from reset to fit taum**********")
        # self.net.hx_backend.config_postcalib({"v_reset": np.zeros(512, dtype=int)})

        if params['taumem_disable_syn_input']:
            raise NotImplementedError
            parameters = {
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: np.zeros(512),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: np.zeros(512),
            }
            self.net.blackbox.set_neuron_cells(parameters)

        for i, neuron in enumerate(params['neurons']):
            if (neuron % self.net.hx_backend._neuron_size) != 0:
                if (neuron - neuron % self.net.hx_backend._neuron_size not in self.all_neurons or
                    '_tau_m' not in self.all_neurons[
                        neuron - neuron % self.net.hx_backend._neuron_size]):
                    raise IOError("taum is the same for all of a non-atomic neuron. "
                                  "thus the leftmost must also be measured")
                for key in ['_tau_m', '_leak', '_mem_difftoreset']:
                    self.all_neurons[neuron][key] = self.all_neurons[
                        neuron - neuron % self.net.hx_backend._neuron_size][key]
                continue
            if debug:
                print(f"neuron {neuron} starting")
            print(f"\r{i+1}/{len(params['neurons'])} "
                  f"({min(params['neurons'])}...{neuron}...{max(params['neurons'])})", end='')

            repetitions = params['repetitions']
            done = False
            while not done:
                try:
                    self.net.hx_backend.set_readout_alt(neuron, "membrane")
                    coord = halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(neuron))
                    builder = stadls.PlaybackProgramBuilder()
                    for i in range(params['batchsize']):
                        builder.block_until(
                            halco.TimerOnDLS(),
                            (i + 1) * 20 * int(self.hx_settings['taum']) * fisch.fpga_clock_cycles_per_us)
                        builder.write(coord.toNeuronResetOnDLS(), haldls.NeuronReset())
                    _, trace = self.net.hx_backend.run([np.empty((0, 2))], duration=2000e-6, experiment_builder=builder,
                                                       record_madc=True)

                    times = trace[:, 0] * 1e6
                    volts_data = trace[:, 1]

                    volts_mean = volts_data.mean()
                    if debug:
                        print(f"neuron {neuron} dissecting")
                    volts, times_trimmed = self.help_dissecttrace(
                        neuron, volts_data, times, lambda x: np.logical_and(
                            np.diff(x[::diff_stepsize]) > params['mindiff_signal_memreset'],
                            x[::diff_stepsize][:-1] < volts_mean),
                    )

                    fit_result = self.help_fitfunction(neuron, volts, times_trimmed,
                                                       exponential_decay, [4, 0.400, -0.200])
                    done = True
                except Exception:
                    repetitions -= 1
                    print(f"retrying neuron {neuron} for {repetitions} more times")
                    if repetitions == 0:
                        raise

            self.all_neurons[neuron]['_tau_m'] = fit_result[0][0]
            self.all_neurons[neuron]['_leak'] = fit_result[0][1]
            self.all_neurons[neuron]['_mem_difftoreset'] = fit_result[0][2]

        if params['taumem_disable_syn_input']:
            raise NotImplementedError
            parameters = {
                halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: np.random.randint(
                    1000, 1020, size=halco.NeuronConfigOnDLS.size),
                halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: np.random.randint(
                    1000, 1020, size=halco.NeuronConfigOnDLS.size),
            }
            self.net.blackbox.set_neuron_cells(parameters)


if __name__ == '__main__':
    # allows plotting of already produced results
    if len(sys.argv) > 2:
        lst = []
        for fn in sys.argv[1:]:
            if not osp.isfile(fn) or not fn[-5:] == ".json":
                raise IOError(f"all arguments ({sys.argv[1:]}) must exist and be json, but {fn} doesnt")
            with open(fn) as f:
                lst.append(json.load(f))
        plotmultiple(lst)
    elif len(sys.argv) > 1 and osp.isfile(sys.argv[1]) and sys.argv[1][-5:] == ".json":
        filename = sys.argv[1]
        with open(filename) as f:
            data = json.load(f)
        plotit(data, [int(i) for i in data["all_neurons"]], filename[:-5])
    else:
        unittest.main()
