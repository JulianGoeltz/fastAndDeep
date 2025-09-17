#!/usr/bin/env python3
# in the paper we settled on the naming reset and adaptation compartment, throughout the project we
# used many names, often delay for the former, and trigger or spiker for the latter
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import os
import os.path as osp
from pathlib import Path
from pprint import pprint
import quantities as pq
import sys

import calix.common
import calix.spiking.neuron
import pyhalco_hicann_dls_vx_v3 as halco
import pyhxcomm_vx as hxcomm
from dlens_vx_v3.sta import PlaybackProgramBuilderDumper, to_portablebinary
import pynn_brainscales.brainscales2 as pynn
import pystadls_vx_v3 as stadls


global_calib_path = (
    f"calibrations/{os.environ.get('SLURM_HARDWARE_LICENSES')}__adaptationDelays-adapted.pbin"
)


def load_calib(filename="spiking_cocolist.pbin"):
    path_to_calib = Path(filename)
    chip = pynn.helper.chip_from_file(path_to_calib)
    return chip


def calc_delays(input_times, delayed_times):
    delays = []
    for t, td in zip(input_times, delayed_times):
        if len(td) == 0:
            delays.append(np.full_like(t, np.inf))
            continue
        tmp = td.reshape((-1, 1)) - t.reshape((1, -1))
        tmp[np.logical_or(tmp < 0, tmp > 0.100)] = np.inf
        delays.append(tmp.min(axis=0))
    delays = np.array(delays) * 1000
    delays[np.isinf(delays)] = np.nan
    return delays


def run_calix(targets, filename):
    with hxcomm.ManagedConnection() as connection:
        stateful_connection = calix.common.base.StatefulConnection(connection)
        # calibrate CADCs
        cadc_result = calix.common.cadc.calibrate(stateful_connection)
        # calibrate neurons
        neuron_target = calix.spiking.neuron.NeuronCalibTarget(**targets)
        target = calix.spiking.SpikingCalibTarget(neuron_target=neuron_target)

        neuron_result = calix.spiking.neuron.calibrate(
            stateful_connection,
            target=neuron_target)

        target_file_npz = Path(filename + ".npz")
        target_file_pbin = Path(filename + ".pbin")

        np.savez(target_file_npz, cadc=cadc_result, neuron=neuron_result, targets=targets)
        builder = PlaybackProgramBuilderDumper()
        neuron_result.apply(builder)

        with target_file_pbin.open(mode="wb") as target:
            target.write(to_portablebinary(builder.done()))


def run_initial_calibration():
    if not osp.isdir('calibrations'):
        os.mkdir('calibrations')
    taus = 10
    taum = 15
    dac = 1022
    isyngm = 50
    vth = 75

    targets_delays = {
            "leak": 50,
            "reset": 50,
            "threshold": vth,
            "tau_mem": taum * 1e-6,
            "tau_syn": taus * 1e-6,
            "i_synin_gm": isyngm,
            "membrane_capacitance": 63,
            "refractory_time": 20e-6 * pq.s,
            "synapse_dac_bias": dac,        # calix can't share this use delay value for both and adapt i_syn_gm for ttfs nrns
            }

    targets_pre_calib = targets_delays.copy()
    targets_pre_calib["tau_mem"] *= pq.s
    targets_pre_calib["tau_syn"] *= pq.s
    target_file_pre_calib = f"calibrations/{os.environ.get('SLURM_HARDWARE_LICENSES')}__precalib"

    print("Starting single-target pre-calib")
    run_calix(targets_pre_calib, target_file_pre_calib)
    print("Finished pre-calib")

    targets_ttfs = {
            "leak": 80,
            "reset": 80,
            "threshold": 150,
            "tau_mem": 6e-6,
            "tau_syn": 6e-6,
            "i_synin_gm": int(700 * 800 / dac),
            }

    targets_combined = {
            "membrane_capacitance": 63,
            "refractory_time": 20e-6 * pq.s,
            "synapse_dac_bias": dac,        # calix can't share this use delay value for both and adapt i_syn_gm for ttfs nrns
            }

    # in case of neuron specific target for i_bias_gm there are crosstalk problems if too many have the same values
    # (if you give a global target, calix varies the targets itself, but not for neuron-specific targets, which we
    #  need here because upper/lower half are set differently)
    # fix: load calibrated values for i_bias_gm as targets for neuron-specific setup
    cal_single_gm_syn = np.load(target_file_pre_calib + ".npz", allow_pickle=True)
    gm_values = []
    for nrn_idx in range(256):
        gm_values.append(list(cal_single_gm_syn['neuron'].item().neurons.values())[256 + nrn_idx].excitatory_input.i_bias_gm.value())

    nrns_per_half = 256
    targets_combined["leak"] = [targets_ttfs["leak"] for _ in range(nrns_per_half)]
    targets_combined["leak"].extend([targets_delays["leak"] for _ in range(nrns_per_half)])

    targets_combined["reset"] = [targets_ttfs["reset"] for _ in range(nrns_per_half)]
    targets_combined["reset"].extend([targets_delays["reset"] for _ in range(nrns_per_half)])

    targets_combined["threshold"] = [targets_ttfs["threshold"] for _ in range(nrns_per_half)]
    targets_combined["threshold"].extend([targets_delays["threshold"] for _ in range(nrns_per_half)])

    targets_combined["tau_mem"] = [targets_ttfs["tau_mem"] for _ in range(nrns_per_half)]
    targets_combined["tau_mem"].extend([targets_delays["tau_mem"] for _ in range(nrns_per_half)])
    targets_combined["tau_mem"] *= pq.s

    targets_combined["tau_syn"] = [targets_ttfs["tau_syn"] for _ in range(nrns_per_half)]
    targets_combined["tau_syn"].extend([targets_delays["tau_syn"] for _ in range(nrns_per_half)])
    targets_combined["tau_syn"] *= pq.s

    targets_combined["i_synin_gm"] = [targets_ttfs["i_synin_gm"] + np.random.randint(low=-20, high=20) for _ in range(nrns_per_half)]
    targets_combined["i_synin_gm"].extend([val for val in gm_values])

    target_file = f"calibrations/{os.environ.get('SLURM_HARDWARE_LICENSES')}__adaptationDelays"

    print("Starting actual two-halves calib")
    run_calix(targets_combined, target_file)
    print("Finished two-halves calib")


class Network():
    """This class is a wrapper around the pynn calls"""
    def __del__(self):
        if hasattr(self, 'pynn'):
            try:
                print("Ending pynn in a delay_utils.Network")
                self.pynn.end()
            except:
                pass

    def setup(self, n_input, n_hidden, n_out, calib_path, use_delays):
        calib = load_calib(calib_path)
        self.pynn = pynn
        self.pynn_use_delays = use_delays
        self.v_mem = None
        self.neurontoberecorded = None

        # enforce network neurons to be on first half of chip and delay nrn on other
        nrn_indices_network = list(range(n_hidden + n_out))
        nrn_indices_delay_delays = np.arange(256, 512, 2)[:n_input + n_hidden]
        nrn_indices_delay_triggers = nrn_indices_delay_delays + 1
        # depends on whether both layers are delayed
        assert len(nrn_indices_delay_delays) >= n_hidden + n_out, \
            f"After blocklist, too few ({len(nrn_indices_delay_delays)}) delay neurons remain"

        print(f"using neurons on chip: network {nrn_indices_network} "
              f"delays {nrn_indices_delay_delays}"
              f"trigger {nrn_indices_delay_triggers}")
        pynn.setup(
            initial_config=calib,
            neuronPermutation=(
                nrn_indices_network +
                list(nrn_indices_delay_delays) +
                list(nrn_indices_delay_triggers)
            ),
        )

        self.pynn_pop_inp = pynn.Population(n_input, pynn.cells.SpikeSourceArray())

        # first define all network neurons
        self.pynn_pop_hid = pynn.Population(n_hidden, pynn.cells.HXNeuron())
        self.pynn_pop_hid.record(["spikes"])
        self.pynn_pop_out = pynn.Population(n_out, pynn.cells.HXNeuron())
        self.pynn_pop_out.record(["spikes"])

        # then define all delay parrots
        # all HX parameters (but the multicompartment) are set by the adapted calib

        self.pynn_popD_inpD = pynn.Population(n_input, pynn.cells.HXNeuron(
            multicompartment_connect_right=True,
        ))
        self.pynn_popD_hidD = pynn.Population(n_hidden, pynn.cells.HXNeuron(
            multicompartment_connect_right=True,
        ))
        self.pynn_popD_inpT = pynn.Population(n_input, pynn.cells.HXNeuron(
        ))
        self.pynn_popD_hidT = pynn.Population(n_hidden, pynn.cells.HXNeuron(
        ))
        self.pynn_popD_inpT.record(["spikes"])
        self.pynn_popD_hidT.record(["spikes"])

        synapse_type = pynn.synapses.StaticSynapse()

        # to shorten the code for both delayed and undelayed settings, use
        # a reference to the last population that carries information

        last_pop = self.pynn_pop_inp

        if use_delays[0]:
            self.pynn_proj_IId = pynn.Projection(
                last_pop,
                self.pynn_popD_inpD,
                pynn.OneToOneConnector(),
                synapse_type=pynn.synapses.StaticSynapse(weight=800),
                receptor_type="excitatory",
            )
            last_pop = self.pynn_popD_inpT

        self.pynn_proj_IdH_e = pynn.Projection(
            last_pop, 
            self.pynn_pop_hid,
            pynn.AllToAllConnector(),
            synapse_type=synapse_type,
            receptor_type="excitatory",
        )
        self.pynn_proj_IdH_i = pynn.Projection(
            last_pop, 
            self.pynn_pop_hid,
            pynn.AllToAllConnector(),
            synapse_type=synapse_type,
            receptor_type="inhibitory",
        )
        last_pop = self.pynn_pop_hid

        if use_delays[1]:
            self.pynn_proj_HHd = pynn.Projection(
                last_pop,
                self.pynn_popD_hidD,
                pynn.OneToOneConnector(),
                synapse_type=pynn.synapses.StaticSynapse(weight=800),
                receptor_type="excitatory",
            )
            last_pop = self.pynn_popD_hidT

        self.pynn_proj_HdO_e = pynn.Projection(
            last_pop,
            self.pynn_pop_out,
            pynn.AllToAllConnector(),
            synapse_type=synapse_type,
            receptor_type="excitatory",
        )
        self.pynn_proj_HdO_i = pynn.Projection(
            last_pop,
            self.pynn_pop_out,
            pynn.AllToAllConnector(),
            synapse_type=synapse_type,
            receptor_type="inhibitory",
        )
        # To have fully built connection pynn needs a short test run
        self.pynn.run(0.01)
        self.pynn.reset()

    def write_params(self, weights, delays=[None, None]):
        # setting delays
        if self.pynn_use_delays[0]:
            trefs = np.clip(
                self.translate_delay_to_tref(delays[0]),
                50, 250)
            self.pynn_popD_inpD.set(refractory_period_refractory_time=trefs.flatten())
        if self.pynn_use_delays[1]:
            trefs = np.clip(
                self.translate_delay_to_tref(delays[1]),
                50, 250)
            self.pynn_popD_hidD.set(refractory_period_refractory_time=trefs.flatten())

        # setting weights
        self.pynn_proj_IdH_e.set(weight=weights[0].clip(min=0))
        self.pynn_proj_IdH_i.set(weight=weights[0].clip(max=0))
        self.pynn_proj_HdO_e.set(weight=weights[1].clip(min=0))
        self.pynn_proj_HdO_i.set(weight=weights[1].clip(max=0))

    def stimulate(self, input_times, duration, record_neuron=None):
        if record_neuron is not None:
            if self.neurontoberecorded is not None:
                self.neurontoberecorded.record(None)
                self.pynn_pop_hid.record(["spikes"])
                self.pynn_pop_out.record(["spikes"])
            self.neurontoberecorded = record_neuron
            record_target = 'v'  # 'exc_synin'
            self.neurontoberecorded.record([record_target])

        self.pynn_pop_inp.set(spike_times=list(input_times))

        pynn.run(duration)

        if record_neuron is not None:
            self.v_mem = self.neurontoberecorded.get_data(
                record_target,
            ).segments[-1].irregularlysampledsignals[0]

        inp_delay_spikes = self._get_spiketimes(self.pynn_popD_inpT)
        hidden_spikes = self._get_spiketimes(self.pynn_pop_hid)
        hidden_delay_spikes = self._get_spiketimes(self.pynn_popD_hidT)
        out_spikes = self._get_spiketimes(self.pynn_pop_out)

        pynn.reset()

        return inp_delay_spikes, hidden_spikes, hidden_delay_spikes, out_spikes

    def translate_delay_to_tref(self, tgt_delays):
        return (tgt_delays - 2.) * 8

    def translate_tref_to_delay(self, refractory_times):
        return refractory_times / 8.0 + 2

    @staticmethod
    def _get_spiketimes(pop):
        # small hack to prevent slowness due to doc/log generation in pynnBrainscales
        sids = sorted(pop.recorder.filter_recorded('spikes', None))
        data = pop.recorder._get_spiketimes(sids, 0, clear=True)
        return [spikes for _, spikes in data.items()]


def plot_layered_results(ax, input_times, inp_delay_spikes, hidden_spikes, hidden_delay_spikes, out_spikes,
         v_mem=None, v_mem_id=None, delays=[None, None]):
    artists = []
    tmp = ax.eventplot(
        [np.array(v) * 1000 for v in input_times],
        color='C0',
        lineoffsets=np.arange(len(input_times)) + 0,
        label="inputs",
    )
    artists.append(tmp[0])
    if delays[0] is not None:
        tmp = ax.eventplot(
            [np.array(v) * 1000 + delays[0][i] for i, v in enumerate(input_times)],
            color='black', ls=(0, (1, 1)), alpha=0.3,
            lineoffsets=np.arange(len(input_times)) + 0,
            label="inp+delay",
        )
        artists.append(tmp[0])
    tmp = ax.eventplot(
        [np.array(v) * 1000 for v in inp_delay_spikes],
        color='C0', ls=':',
        lineoffsets=np.arange(len(inp_delay_spikes)),
        label="delayed inputs",
    )
    artists.append(tmp[0])
    tmp = ax.eventplot(
        [np.array(v) * 1000 for v in hidden_spikes],
        color='C1',
        lineoffsets=np.arange(len(hidden_spikes)) + len(input_times),
        label="hidden",
    )
    artists.append(tmp[0])
    if delays[1] is not None:
        tmp = ax.eventplot(
            [np.array(v) * 1000 + delays[1][i] for i, v in enumerate(hidden_spikes)],
            color='black', ls=(0, (1, 1)), alpha=0.3,
            lineoffsets=np.arange(len(hidden_spikes)) + len(input_times),
            label="hid+del",
        )
        artists.append(tmp[0])
    tmp = ax.eventplot(
        [np.array(v) * 1000 for v in hidden_delay_spikes],
        color='C1', ls=':',
        lineoffsets=np.arange(len(hidden_delay_spikes)) + len(input_times),
        label="delayed hiddens",
    )
    artists.append(tmp[0])
    tmp = ax.eventplot(
        [np.array(v) * 1000 for v in out_spikes],
        color='C2',
        lineoffsets=np.arange(len(out_spikes)) + len(input_times) + len(hidden_spikes),
        label="label",
    )
    artists.append(tmp[0])

    if v_mem is not None:
        ax.plot(v_mem.times * 1000,
                (v_mem-500) * 50. / 1023 + v_mem_id,
                color='green', lw=3)

    ax.set_xlabel("time [μs]")
    ax.set_ylabel("neuron indices")
    # ax.set_xlim(0, 120)
    ax.set_ylim(-0.5, len(input_times) + len(hidden_spikes) + len(out_spikes))
    ax.legend(loc='upper right', handles=artists)


def test_layered_net():
    n_input = 20
    n_hidden = 30
    n_out = 3
    duration = 0.10
    n_inputSpikes = 1

    np.random.seed(696969)

    # input_times = np.random.rand(n_input, n_inputSpikes) * duration
    # input_times = np.random.rand(4, n_inputSpikes).repeat(5, axis=0) * duration * 0.3
    input_times = np.random.rand(1, n_inputSpikes).repeat(20, axis=0) * duration * 0.3

    print('input times')
    print(input_times)

    weights = [
        np.random.randint(low=62, high=63, size=(n_input, n_hidden)),
        np.random.randint(low=62, high=63, size=(n_hidden, n_out)),
    ]

    delays = [
        np.random.random(size=n_input) * 0. + 15.,
        np.random.random(size=n_hidden)* 0. + 15.,
    ]
    print('Target delays')
    print(delays)

    fig, axes = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
    network = Network()

    network.setup(n_input, n_hidden, n_out, global_calib_path, [True, True])
    

    v_mem_id = 0
    record_neuron = None # network.pynn_popD_inp[v_mem_id:v_mem_id+1]

    network.write_params(weights, delays)
    for i, ax in enumerate(axes.flatten()):
        inp_delay_spikes, hidden_spikes, hidden_delay_spikes, out_spikes = \
            network.stimulate(input_times, duration,
                              record_neuron=record_neuron)
        v_mem = network.v_mem

        network.write_params(weights, delays)

        plot_layered_results(ax, input_times, inp_delay_spikes, hidden_spikes, hidden_delay_spikes, out_spikes,
             v_mem, v_mem_id=v_mem_id, delays=delays)

    fig.tight_layout()
    fig.savefig(f'dbg_pynn_raster_{os.environ.get("SLURM_HARDWARE_LICENSES")}.png')


def test_delays():
    calib = pynn.helper.chip_from_file(global_calib_path)

    n_neurons = 256 // 2
    neurons = 256 +  np.arange(n_neurons) * 2
    pynn.setup(
        initial_config=calib,
        neuronPermutation=list(neurons) + list(neurons + 1)
    )

    stimulus = pynn.Population(1, pynn.cells.SpikeSourceArray())
    for n in range(1):
        stimulus[n:n+1].set(spike_times=[0.1 + n*1e-5])


    delay_population = pynn.Population(n_neurons, pynn.cells.HXNeuron(
        multicompartment_connect_right=True,
    ))
    trigger_population = pynn.Population(n_neurons, pynn.cells.HXNeuron(
        refractory_period_refractory_time=50,
    ))


    synapse = pynn.standardmodels.synapses.StaticSynapse(weight=800)
    projection = pynn.Projection(stimulus, delay_population, pynn.AllToAllConnector(),
                       synapse_type=synapse, receptor_type="excitatory")


    refractory_times = np.arange(50, 251, 50)


    delays = []
    plot = True


    offsets = np.arange(128)
    results = np.empty((len(offsets), len(refractory_times), 4))

    for i_offset, offset in enumerate(offsets):
        delay = delay_population[i_offset:i_offset+1]
        spiker = trigger_population[i_offset:i_offset+1]
        
        fig = plt.figure(figsize=(8, 8))
        grid = gs.GridSpec(refractory_times.size, 1)
        for i, refractory_time in enumerate(refractory_times):
            delay.set(refractory_period_refractory_time=refractory_time)
            target = "adaptation"
            spiker.record(None)
            spiker.record([target, "spikes"])

            # run the experiment
            pynn.run(0.3)
            
            pop = spiker
            mem_v = pop.get_data(target).segments[-1].irregularlysampledsignals[0]
            spikes = pop.get_data("spikes").segments[-1].spiketrains[0]

            pynn.reset()

            if fig is not None:
                ax = fig.add_subplot(grid[i, 0])

                pop = spiker

                ax.plot(mem_v.times, mem_v, c="k")
                ax.axhline(mem_v[:100].mean(), c='k', alpha=0.1)

                for spike in spikes:
                    ax.axvline(spike, c="C1")

            results[i_offset, i, :] = np.array([
                mem_v.max(), mem_v.min(), mem_v[:100].mean(), (
                    float(spikes[0]) - 0.1 if len(spikes) > 0 else np.inf
                )
            ])
        plt.close(fig)


        fig.savefig(f'test/dbgdbg_offset{offset}.png')
    np.save(f'dbg_results_{os.environ.get("SLURM_HARDWARE_LICENSES")}.npy', results)

    results[np.isinf(results)] = 0.0
    fig, ax = plt.subplots(1, 1)
    for i in range(len(offsets)):
        print(refractory_times)
        print(results[i, :, 3])
        ax.plot(refractory_times, results[i, :, 3], ".-", c="k", alpha=0.1)

    ax.plot(refractory_times, (refractory_times / 8.0 + 2) * 1e-3, c="C1")
    fig.savefig(f'dbg_schar_{os.environ.get("SLURM_HARDWARE_LICENSES")}.png')
    plt.close(fig)

    print('saved')


def calibrate_adaptation_v_ref(neurons, iterations, target, lower=200, upper=1000):
    delay, spiker = neurons

    spiker.record(None)
    spiker.record(["adaptation"])

    # fig = plt.figure(figsize=(8, 8))
    # grid = gs.GridSpec(iterations, 1)
    fig = None
    
    print(f"  calibrating adaptation baseline (target = {target:3.1f})")
    
    for i in range(iterations):
        pivot = (lower + upper) // 2
        spiker.set(adaptation_v_ref=pivot)

        # run the experiment
        pynn.run(0.3)
        
        trace = spiker.get_data("adaptation").segments[-1].irregularlysampledsignals[0]
        base = float(trace.mean())

        print(f"    {pivot: 3d} → μ = {base:3.1f}")

        if fig is not None:
            ax = fig.add_subplot(grid[i, 0])

            ax.plot(trace.times, trace, c="k")
            ax.axhline(base, c='k', alpha=0.1)
            ax.set_title(f"μ = {base:.1f}")

        if base > target:
            upper = pivot
        else:
            lower = pivot

        pynn.reset()

    return pivot, base


def calibrate_adaptation_v_leak(neurons, iterations, target, lower=200, upper=1000):
    delay, spiker = neurons

    spiker.record(None)
    spiker.record(["v"])

    print(f"  calibrating membrane resting state (target = {target:3.1f})")
    
    spiker.set(adaptation_enable=True)

    # fig = plt.figure(figsize=(8, 8))
    # grid = gs.GridSpec(n_iterations, 1)
    fig = None
    
    for i in range(iterations):
        pivot = (lower + upper) // 2
        spiker.set(adaptation_v_leak=pivot)

        # run the experiment
        pynn.run(0.3)
        
        trace = spiker.get_data("v").segments[-1].irregularlysampledsignals[0]
        base = float(trace.mean())

        print(f"    {pivot: 3d} → μ = {base:3.1f}")

        if fig is not None:
            ax = fig.add_subplot(grid[i, 0])

            ax.plot(trace.times, trace, c="k")
            ax.axhline(base, c='k', alpha=0.1)
            ax.set_title(f"μ = {base:.1f}")

        if base > target:
            upper = pivot
        else:
            lower = pivot

        pynn.reset()

    return pivot, base


def calibrate_leak_v_leak(neurons, iterations, target, lower=200, upper=1000):
    delay, spiker = neurons

    spiker.record(None)
    spiker.record(["v"])

    print(f"  calibrating membrane resting state (target = {target:3.1f})")
    
    # fig = plt.figure(figsize=(8, 8))
    # grid = gs.GridSpec(n_iterations, 1)
    fig = None
    
    for i in range(iterations):
        pivot = (lower + upper) // 2
        spiker.set(leak_v_leak=pivot)

        # run the experiment
        pynn.run(0.3)
        
        trace = spiker.get_data("v").segments[-1].irregularlysampledsignals[0]
        base = float(trace.mean())

        print(f"    {pivot: 3d} → μ = {base:3.1f}")

        if fig is not None:
            ax = fig.add_subplot(grid[i, 0])

            ax.plot(trace.times, trace, c="k")
            ax.axhline(base, c='k', alpha=0.1)
            ax.set_title(f"μ = {base:.1f}")

        if base > target:
            upper = pivot
        else:
            lower = pivot

        pynn.reset()

    return pivot, base


def calibrate_threshold_v_threshold(neurons, iterations, target, lower=200, upper=1000):
    delay, spiker = neurons

    spiker.record(None)
    spiker.record(["v", "spikes"])

    print(f"  calibrating threshold (target = {target:3.1f})")
    
    fig = plt.figure(figsize=(8, 8))
    grid = gs.GridSpec(iterations, 1)
    # fig = None
    
    for i in range(iterations):
        pivot = (lower + upper) // 2
        spiker.set(threshold_v_threshold=pivot)

        # run the experiment
        pynn.run(0.3)
        
        trace = spiker.get_data("v").segments[-1].irregularlysampledsignals[0]
        base = float(trace[:100].mean())
        maximum = float(trace.max())
        spikes = np.array(spiker.get_data("spikes").segments[-1].spiketrains)

        print(f"    {pivot: 3d} → {spikes.size:2d} spikes, μ = {maximum:3.1f}")

        if fig is not None:
            ax = fig.add_subplot(grid[i, 0])

            ax.plot(trace.times, trace, c="k")
            ax.axhline(base, c='k', alpha=0.1)
            ax.set_title(f"μ = {base:.1f}, max = {maximum:.1f} ({target:.1f})")

        if spikes.size < 1 or maximum > target:
            upper = pivot
        else:
            lower = pivot

        pynn.reset()

    fig.savefig(f"test/threshold_calib_{delay.first_id-1:03d}.png")

    
def calibrate_delays():
    n_neurons = 256 // 2
    neurons = 256 + np.arange(n_neurons) * 2
    calib = pynn.helper.chip_from_file(
        f"calibrations/{os.environ.get('SLURM_HARDWARE_LICENSES')}__adaptationDelays.pbin"
    )
    # need neuronPermutation so we can directly instantiate the delay and trigger pops
    pynn.setup(
        initial_config=calib,
        neuronPermutation=list(neurons) + list(neurons + 1) + list(np.arange(256))
    )

    if not osp.isdir('test'):
        os.mkdir('test')
    stimulus = pynn.Population(1, pynn.cells.SpikeSourceArray())
    for n in range(1):
        stimulus[n:n+1].set(spike_times=[0.1 + n*1e-3])

    delay_population = pynn.Population(n_neurons, pynn.cells.HXNeuron(
        leak_i_bias=0,
        leak_enable_multiplication=False,
        leak_enable_division=True,
        reset_v_reset=250,
        threshold_enable=False,
        refractory_period_refractory_time=50,
        multicompartment_connect_right=True,
        excitatory_input_enable=False,
        inhibitory_input_enable=False,
        excitatory_input_enable_small_capacitance=True,
        excitatory_input_enable_high_resistance=True,
        excitatory_input_i_bias_gm=0,
        excitatory_input_i_bias_tau=0,
        inhibitory_input_i_bias_tau=0,
        refractory_period_input_clock=0,
        event_routing_enable_bypass_excitatory=True,
        
        membrane_capacitance_capacitance=10,
    ))
    trigger_population = pynn.Population(n_neurons, pynn.cells.HXNeuron(
        membrane_capacitance_capacitance=0,
        threshold_enable=False,
        threshold_v_threshold=432,
        leak_v_leak=500,
        leak_i_bias=500,
        leak_enable_multiplication=False,
        leak_enable_division=False,
        excitatory_input_enable=False,
        inhibitory_input_enable=False,
        adaptation_v_ref=900,
        adaptation_v_leak=700,
        adaptation_i_bias_tau=1000 + np.random.randint(-5, 5, n_neurons),
        adaptation_i_bias_a=1000,
        adaptation_i_bias_b=1000,
        adaptation_invert_b=False,
        adaptation_enable_pulse=True,
        adaptation_enable=True,
        reset_v_reset=600,
        refractory_period_input_clock=0,
        refractory_period_refractory_time=25,
    ))

    synapse = pynn.standardmodels.synapses.StaticSynapse(weight=800)
    projection = pynn.Projection(stimulus, delay_population, pynn.AllToAllConnector(),
                       synapse_type=synapse, receptor_type="excitatory")


    for i in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
        # clock_scale_fast is set by calix
        pynn.simulator.state.grenade_chip_config.neuron_block.backends[i].clock_scale_slow = 4
        pynn.simulator.state.grenade_chip_config.neuron_block.backends[i].clock_scale_adaptation_pulse = 8

    # offsets = range(337, 338, 3)
    offsets = range(n_neurons)
    # offsets = [6]
    results = np.empty((len(offsets), 4))
    for i_offset, offset in enumerate(offsets):
        print(f"calibrating neuron {offset}")
        

        delay = delay_population[i_offset:i_offset+1]
        spiker = trigger_population[i_offset:i_offset+1]
        
        # there are two options of determining a calibrate target:
        # 1. record the original (adaptation-less) resting potential and use it as a target
        # 2. define an arbitrary target
        # 
        # The latter option might be the safest choice, as it is independent of the quality of the original calibration.
        # Code for option 1. is, however, present in a comment.
        # record resting state with adaptation disabled as target

        # # determine adaptation-free resting state to use as target
        # spiker.set(adaptation_enable=False)
        # pynn.run(0.3)
        # 
        # trace = spiker.get_data("v").segments[-1].irregularlysampledsignals[0]
        # target = float(trace.mean())
        # pynn.reset()
        
        spiker.set(threshold_enable=False)

        calibrate_adaptation_v_ref((delay, spiker), target=500, iterations=8)
        calibrate_adaptation_v_leak((delay, spiker), target=600, iterations=8)
        calibrate_adaptation_v_ref((delay, spiker), target=500, iterations=8)
        calibrate_leak_v_leak((delay, spiker), target=600, iterations=8)
        calibrate_adaptation_v_ref((delay, spiker), target=500, iterations=8)
        calibrate_leak_v_leak((delay, spiker), target=600, iterations=8)

        stimulus.set(spike_times=[0.1])
        
        spiker.record(None)
        spiker.record(["v"])
        pynn.run(0.3)
        trace = spiker.get_data("v").segments[-1].irregularlysampledsignals[0]

        # fig = plt.figure()
        # ax = fig.gca()
        # ax.plot(trace.times, trace)
        # fig.savefig("test/asdasd.png")

        rest = trace[:100].mean()
        maximum = trace.max()

        print(rest, maximum)

        target = (rest + maximum) / 2

        pynn.reset()


        spiker.set(threshold_enable=True)
        calibrate_threshold_v_threshold((delay, spiker), target=target, iterations=8)
        
        
        # check calibration of adaptation baseline
        spiker.record(None)
        spiker.record(["adaptation"])
        pynn.run(0.3)
        trace = spiker.get_data("adaptation").segments[-1].irregularlysampledsignals[0]
        results[i_offset, 1] = float(trace.mean())
        pynn.reset()
       
        
        # check calibration of membrane resting state
        spiker.record(None)
        spiker.record(["v"])
        pynn.run(0.3)
        trace = spiker.get_data("v").segments[-1].irregularlysampledsignals[0]
        results[i_offset, 0] = float(trace[:100].mean())
        pynn.reset()

        # neuron_config = pynn.simulator.state.grenade_chip_config.neuron_block.atomic_neurons[halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(offset*2 + 1))]

        # # neuron_config = calib.neuron_block.atomic_neurons[halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(256 + offset*2 + 1))]
        # neuron_config.leak.v_leak = int(spiker.get("leak_v_leak"))
        # neuron_config.adaptation.v_ref = int(spiker.get("adaptation_v_ref"))
        # neuron_config.adaptation.v_leak = int(spiker.get("adaptation_v_leak"))

        print(f"  calibration result: v = {results[i_offset, 0]:3.1f}, w = {results[i_offset, 1]:3.1f}")

    np.save(f'dbg_calibration_results_{os.environ.get("SLURM_HARDWARE_LICENSES")}.npy', results)


    fig, ax = plt.subplots(2, 1)
    for i in range(len(results)):
        ax[0].hist(results[:, 0])
        ax[1].hist(results[:, 1])
    fig.savefig(f'dbg_calibration_results_{os.environ.get("SLURM_HARDWARE_LICENSES")}.png')
    plt.close(fig)

    builder = stadls.PlaybackProgramBuilderDumper()
    builder.write(halco.ChipOnDLS(), pynn.simulator.state.grenade_chip_config)

    with open(global_calib_path, mode="wb") as target:
        target.write(stadls.to_portablebinary(builder.done()))


if __name__ == "__main__":
    calls = ["doAll", "calix", "calibDelays", "tests"]
    if len(sys.argv) <= 1 or sys.argv[1] not in calls:
        sys.exit(f"need to be called with one of the following {calls}")

    call = sys.argv[1]
    if call in ["doAll", "calix"]:
        print("************ Starting initial calibration with calix")
        run_initial_calibration()
    if call in ["doAll", "calibDelays"]:
        print("************ calibrating adaptations for delay mechanism")
        calibrate_delays()
    if call in ["doAll", "tests"]:
        print("************ startin some tests to record basic functionality of delays")
        test_delays()
        test_layered_net()
