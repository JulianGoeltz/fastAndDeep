import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pynn_brainscales.brainscales2 as pynn
import pyhalco_hicann_dls_vx_v3 as halco

from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pathlib import Path
from colorama import Fore, Style
from scipy.optimize import curve_fit
from scipy.stats import chi2


def single_run(nrn_indices, input_times, duration, weight, calib_path):
    calib = pynn.helper.chip_from_file(Path(calib_path))
    pynn.setup(initial_config=calib, neuronPermutation=nrn_indices)

    # for s in halco.iter_all(halco.SynapseDriverBlockOnDLS):
    #     pynn.simulator.state.grenade_chip_config.synapse_driver_blocks[s].padi_bus.dacen_pulse_extension.fill(10)
                                            
    in_channel = pynn.Population(1, pynn.cells.SpikeSourceArray(spike_times=input_times))
    parrots = []
    for i, nrn in enumerate(nrn_indices):
        neuron = pynn.Population(1, pynn.cells.HXNeuron())
        neuron.record(["spikes"])
        parrots.append(neuron)

        if isinstance(weight, (list, np.ndarray)):
            synapse = pynn.synapses.StaticSynapse(weight=weight[i])
        else:
            synapse = pynn.synapses.StaticSynapse(weight=weight)
        pynn.Projection(in_channel, neuron, pynn.AllToAllConnector(), synapse_type=synapse, receptor_type="excitatory")

    pynn.run(duration)
    out_spikes = []
    for neuron in parrots:
        spikes = neuron.get_data("spikes").segments[0].spiketrains
        out_spikes.append(spikes)
    pynn.end()
    return out_spikes


def calc_produced_delays(input_times, output_times, w, nrn):
    # Set a delay of nan if the neuron did not spike in response
    # handle case of no outputs
    if len(output_times[0]) == 0:
        delays = [np.nan for _ in input_times]
        return delays
    # assume that if we have a missing spike for one input, then for another input there will be no double spike
    if len(output_times[0]) > len(input_times):
        print(Fore.RED + "WARNING")
        print(f"got {len(output_times[0])} output spikes for {len(input_times)} "\
              f"inputs for nrn {nrn} with weight {w}, output spikes\n {output_times}")
        print(Style.RESET_ALL)
    delays = []
    for i, t in enumerate(input_times):
        diffs = output_times[0] - t
        # smallest positive value
        # Similarly replace non positive values with inf then use argmin to find the smallest positive:
        idx = np.where(diffs > 0, diffs, np.inf).argmin()
        # check if reasonable delay by comparing to ISI of input
        # ignore spikes that occur long past realistically reachable delays
        if i < len(input_times) - 1:
            isi = input_times[i+1] - t
        else:
            isi = t - input_times[i-1]
        if diffs[idx] < isi/2. and diffs[idx] > 0:
            # calculate delays in us
            delay = diffs[idx] * 1e3
        else:
            delay = np.nan
        delays.append(delay)
    delays = np.array(delays)
    return delays


def record_delays(nrn_indices, weights, n_runs, data_path, calib_name):
    input_spikes = np.arange(0.2, 5.2, 0.1)
    duration = 5.4
    delays = np.empty((n_runs, len(nrn_indices), len(weights), len(input_spikes)))
    for run in range(n_runs):
        print(f"################# RUN {run} ####################")
        for i, w in enumerate(weights):
            print(f"Running weight {w}")
            output_spikes = single_run(nrn_indices, input_spikes, duration, w, calib_name)
            for j, nrn in enumerate(nrn_indices):
                d = calc_produced_delays(input_spikes, np.array(output_spikes[j]), w, nrn)
                delays[run, j, i, :] = d
    np.save(data_path + f"recorded_delays.npy", delays)


def collect_data_for_fit(nrn_idx, weights, data_path):
    loaded_delays = np.load(data_path + f"recorded_delays.npy")
    all_delays = [[] for w in weights]
    for j, w in enumerate(weights):
        delays = loaded_delays[:, nrn_idx, j, :].flatten()
        all_delays[j].extend(delays)
    ws = []
    means = []
    stds = []
    for j, w in enumerate(weights):
        # only include runs with enough spikes in calculations (80% of all spikes)
        delays = all_delays[j]
        n_missing_spikes = np.isnan(delays).sum()
        if (n_missing_spikes / len(delays)) < 0.2:
            d_mean = np.nanmean(delays)
            d_std = np.nanstd(delays)
            if not np.isnan(d_mean):
                ws.append(w)
                means.append(d_mean)
                stds.append(d_std)
    return ws, means, stds


def check_fit_fail(nrn_nr, data_path, ws, means, stds, fit_res):
    exp_func = lambda x, a, b, c, d: a + b * np.exp(c * (x - d))
    chisqr = sum((np.array(means) - exp_func(np.array(ws), fit_res[0], fit_res[1], fit_res[2], fit_res[3]))**2 / np.array(stds)**2)
    dof = len(means) - len(fit_res)
    red_chisqr = chisqr / dof
    print("curve parameters", fit_res)
    print("chisquare / num_degrees_of_freedom", red_chisqr)
    if red_chisqr > 5.:
        print("Fit failed")
        if Path(data_path + "failed_delay_fits.txt").is_file():
            failed_nrns = np.loadtxt(data_path + "failed_delay_fits.txt")
            if not (nrn_nr in failed_nrns):
                failed_nrns = np.hstack((failed_nrns, [nrn_nr]))
        else:
            failed_nrns = [nrn_nr]
        np.savetxt(data_path + "failed_delay_fits.txt", failed_nrns, fmt='%i')
        return True
    else:
        return False


weight_scaling = 0.1


def exp_fit(weights, means, stds, nrn_nr, data_path):
    exp_func = lambda x, a, b, c, d: a + b * np.exp(c * (x - d))
    guess = [3., 5., -0.10 * weight_scaling, 20 / weight_scaling]
    print("#### exp-fit")
    fit_res, _ = curve_fit(exp_func, np.array(weights), means, sigma=stds, p0=guess)
    failed = check_fit_fail(nrn_nr, data_path, weights, means, stds, fit_res)
    to_save = list(fit_res)
    to_save.append(min(weights))
    to_save.append(max(weights))
    np.save(data_path + f"delay_fit_nrn_{nrn_nr}.npy", to_save)
    return fit_res, exp_func


def theory_fit(weights, means, stds, nrn_nr, data_path):
    from scipy.odr import ODR, Model, Data, RealData
    w_func = lambda beta, d: (beta[2] * (beta[0] - beta[1])) / \
            (beta[1] * (np.exp(-d / beta[0]) - np.exp(-d / beta[1])))
    guess = [1. / weight_scaling * 15., 10., 1.]
    data = RealData(np.array(means), np.array(weights), sx=np.array(stds))
    model = Model(w_func)
    odr = ODR(data, model, np.array(guess))
    odr.set_job(fit_type=0)
    output = odr.run()
    print("#### Theory-fit")
    fit_res = list(output.beta)
    print("curve parameters", fit_res)
    to_save = list(fit_res)
    to_save.append(min(weights))
    to_save.append(max(weights))
    np.save(data_path + f"delay_theory_fit_nrn_{nrn_nr}.npy", to_save)
    return fit_res, w_func


def plot_fit_results(nrn_nr, nrn_idx, n_runs, ws, means, stds, data_path, exp_res, exp_func, w_res, w_func):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=True)
    ax[2].set_xlabel("weight")
    ax[0].set_ylabel("delay [us]")        
    ax[0].set_title(f"Nrn {nrn_nr}")
    ax[2].plot(ws, means, color=f"C0", alpha=0.7, label='all runs')
    ax[2].errorbar(ws, means, yerr=stds, marker='x', color="C0")
    ax[2].fill_between(ws, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=f"C0", alpha=0.2)
    ws_more_detailed = np.linspace(np.min(ws), np.max(ws), 100)
    ax[2].plot(ws_more_detailed, exp_func(np.array(ws_more_detailed), exp_res[0], exp_res[1], exp_res[2], exp_res[3]), color=f"C1", alpha=0.7, label='exp fit')
    ax[2].plot(w_func(w_res, np.array(means)), means, color=f"C2", alpha=0.7, label='theory fit')
    loaded_delays = np.load(data_path + f"recorded_delays.npy")
    for i in range(n_runs):
        ws = []
        means = []
        stds = []
        for j, w in enumerate(weights):
            delays = loaded_delays[i, nrn_idx, j, :]
            d_mean = np.nanmean(delays)
            d_std = np.nanstd(delays)
            ax[0].scatter([w for _ in delays], delays, marker='.', color=f"C{i}", alpha=0.3)
            ws.append(w)
            means.append(d_mean)
            stds.append(d_std)
        ax[1].errorbar(ws, means, yerr=stds, marker='x', color=f"C{i}")
        ax[1].plot(weights, means, color=f"C{i}", alpha=0.7, label=f"run {i}")
        ax[1].fill_between(ws, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=f"C{i}", alpha=0.2)
    ax[1].legend()
    ax[2].legend()
    plt.savefig(data_path + f"delay_over_w_nrn_{nrn_nr}.png")
    plt.close(fig)


def create_delay_fit(nrn_nr, nrn_idx, n_runs, weights, data_path):
    print(f"############### nrn {nrn_nr}")
    ws, means, stds = collect_data_for_fit(nrn_idx, weights, data_path)
    to_save = [ws, means, stds]
    np.save(data_path + f"fit_data_ws_mean_std_nrn_{nrn_nr}.npy", to_save)
    fit_res_exp, fit_func = exp_fit(ws, means, stds, nrn_nr, data_path)
    fit_res_w, w_func = theory_fit(ws, means, stds, nrn_nr, data_path)
    plot_fit_results(nrn_nr, nrn_idx, n_runs, ws, means, stds, data_path, fit_res_exp, fit_func, fit_res_w, w_func)


if __name__ == '__main__':
    import sys
    pynn.logger.default_config(level=pynn.logger.LogLevel.WARN)
    hxcommlogger = pynn.logger.get("hxcomm")

    mode = sys.argv[1]

    calib_name = sys.argv[2]
    assert Path(calib_name).is_file(), f"Calibration {calib_name} does not exist"

    if mode == 'generate':
        assert os.path.splitext(calib_name)[1] == '.pbin'
        name = osp.basename(osp.splitext(calib_name)[0])
        data_name = "data/" + name + "/"
        data_path = Path(data_name)
        data_path.mkdir(parents=True, exist_ok=True)

        n_runs = 10
        np.save(data_name + "n_runs.npy", n_runs)
        neurons = range(256, 512, 1)
        np.save(data_name + "nrn_indices.npy", neurons)
        weights = range(10, 63, 1)
        np.save(data_name + "weights.npy", weights)

        record_delays(neurons, weights, n_runs, data_name, calib_name)
        for i, nrn in enumerate(neurons):
            create_delay_fit(nrn, i, n_runs, weights, data_name) 

    elif mode == 'reevaluate':
        name = osp.basename(osp.splitext(calib_name)[0])
        data_name = "data/" + name + "/"
        n_runs = np.load(data_name + "n_runs.npy")
        neurons = np.load(data_name + "nrn_indices.npy")
        weights = np.load(data_name + "weights.npy")
        for i, nrn in enumerate(neurons):
            create_delay_fit(nrn, i, n_runs, weights, data_name) 

    elif mode == 'test':
        raise NotImplementedError(f"TODO")
#        name = calib_name[7:-5]
#        print(f"################# Rerunning for range of targets ####################")
#        targets = np.arange(1.5, 8.5, 0.2)
#        tgt_vs_real_delay(neurons, targets, data_name, calib_name)

    else:
        raise NotImplementedError(f"Ran with mode = {mode}, but must run in either 'generate' or 'test' mode")
