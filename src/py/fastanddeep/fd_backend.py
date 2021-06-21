#!python3
"""Class for BSS-2 API that inherits from strobe

Strobe is an interface for the BrainScaleS-2 chip that is developed by 
Sebastian Billaudelle and Benjamin Cramer, and we want to take this opportunity
again to thank them for their time, energy and support.
The FandDBackend class here inherits from their backend, and the code is often
copied from their backend and only slightly varied.
"""
import json
import numpy as np
from pprint import pprint
import os
import os.path as osp
import sys
import time

# TODO: MOVE THIS TO SRC
import calix.common
import calix.constants
import pyhxcomm_vx as hxcomm
import pyhaldls_vx_v2 as haldls
import pystadls_vx_v2 as stadls
import pyfisch_vx as fisch
import pylola_vx_v2 as lola
import pyhalco_hicann_dls_vx_v2 as halco
import gonzales

import utils

import strobe.backend
from strobe.backend import PPUSignal, enable_recurrency_builder, disable_recurrency_builder


class FandDBackend(strobe.backend.StrobeBackend):
    def __init__(self, *args, **kwargs):
        # call configure of base class
        super().__init__(*args, **kwargs)

        self.calib_translate_dict = {
            "synapse_bias":
                0,
            "tau_m":
                halco.CapMemRowOnCapMemBlock.i_bias_leak,
            "tau_syn_exc":
                halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau,
            "tau_syn_inh":
                halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau,
            "v_leak":
                halco.CapMemRowOnCapMemBlock.v_leak,
            "v_reset":
                halco.CapMemRowOnCapMemBlock.v_reset,
            "v_thresh":
                halco.CapMemRowOnCapMemBlock.v_threshold,
        }

        self._timing_offset = 100e-6  # TODO: was 100e-6
        self._timing_separation = 500e-0  # TODO: think about this

    def configure(self, reduce_power=False, initialize=True):
        # call configure of base class
        super().configure(reduce_power=reduce_power, initialize=initialize)

        # configure MADC
        builder = stadls.PlaybackProgramBuilder()
        builder.write(halco.CapMemCellOnDLS.readout_ac_mux_i_bias, haldls.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_madc_in_500na, haldls.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_sc_amp_i_bias, haldls.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_pseudo_diff_v_ref, haldls.CapMemCell(400))
        builder.write(halco.CapMemCellOnDLS.readout_sc_amp_v_ref, haldls.CapMemCell(400))

        for block in halco.iter_all(halco.CapMemBlockOnDLS):
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.neuron_i_bias_readout_amp, block),
                          haldls.CapMemCell(70))
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.neuron_i_bias_leak_source_follower,
                                                block),
                          haldls.CapMemCell(100))
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.neuron_i_bias_spike_comparator, block),
                          haldls.CapMemCell(100))
            builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.neuron_v_bias_casc_n, block),
                          haldls.CapMemCell(270))

        # INA219 config for longer averaging
        config = haldls.INA219Config()
        config.bus_adc_mode = config.ADCMode.bits12_samples8
        config.shunt_adc_mode = config.ADCMode.bits12_samples8
        for i in halco.iter_all(halco.INA219ConfigOnBoard):
            builder.write(i, config)

        # static MADC config
        config = haldls.MADCConfig()
        builder.write(halco.MADCConfigOnDLS(), config)

        # PPU stuff
        ppu_control_reg_reset = haldls.PPUControlRegister()
        ppu_control_reg_reset.inhibit_reset = False
        ppu_control_reg_reset.force_clock_off = True

        for i in range(10):
            for ppu in range(2):
                # ensure PPU is in reset state
                builder.write(halco.PPUControlRegisterOnDLS(ppu), ppu_control_reg_reset)

        stadls.run(self._connection, builder.done())
        self.load_ppu_program("/wang/users/jgoeltz/cluster_home/strobeStuff/bin/strobe.bin")

    def config_postcalib(self, postcalib):
        """load and apply postcalib

        postcalib either path or dict"""
        if isinstance(postcalib, str):
            if osp.isfile(postcalib):
                with open(postcalib, 'r') as f:
                    tmp = json.load(f)
                    self.calib_values = {k: np.array(v) for k, v in tmp.items()}
            else:
                raise IOError("needs valid path")
        else:
            self.calib_values = postcalib

        capmem_dict = {self.calib_translate_dict[k]: self.calib_values[k] for k in self.calib_values}
        # print(capmem_dict)
        builder = stadls.PlaybackProgramBuilder()
        if 'synapse_bias' in self.calib_values:
            capmem_dict.pop(0)
            for block in halco.iter_all(halco.CapMemBlockOnDLS):
                builder.write(halco.CapMemCellOnDLS(halco.CapMemCellOnCapMemBlock.syn_i_bias_dac, block),
                              haldls.CapMemCell(int(self.calib_values['synapse_bias'][block.value()])))
        calix.common.helpers.capmem_set_neuron_cells(builder, capmem_dict)
        builder = calix.common.helpers.wait_for_us(builder, calix.constants.capmem_level_off_time)
        # print(builder)
        calix.common.base.run(self._connection, builder)
        print("#### Loaded and applied custom calibration")

    def set_readout_alt(self, record_neuron, record_target):
        builder = stadls.PlaybackProgramBuilder()

        madc_config = haldls.MADCConfig()
        madc_config.number_of_samples = 10000
        builder.write(halco.MADCConfigOnDLS(), madc_config)

        readout_params = {halco.CapMemCellOnDLS.readout_ac_mux_i_bias: 500,
                          halco.CapMemCellOnDLS.readout_madc_in_500na: 500,
                          halco.CapMemCellOnDLS.readout_sc_amp_i_bias: 500,
                          halco.CapMemCellOnDLS.readout_sc_amp_v_ref: 400,
                          halco.CapMemCellOnDLS.readout_pseudo_diff_v_ref: 400}

        for k, v in readout_params.items():
            builder.write(k, haldls.CapMemCell(v))

        for c in halco.iter_all(halco.AtomicNeuronOnDLS):
            config = self._neuron_calib.neurons[c].asNeuronConfig()
            config.enable_readout = (int(c.toEnum()) == record_neuron)
            config.readout_source = getattr(config.ReadoutSource, record_target)
            builder.write(c.toNeuronConfigOnDLS(), config)

        hemisphere = record_neuron // halco.NeuronColumnOnDLS.size
        is_odd = (record_neuron % 2) == 1
        is_even = (record_neuron % 2) == 0

        config = haldls.ReadoutSourceSelection()
        sm = config.SourceMultiplexer()
        sm.neuron_odd[halco.HemisphereOnDLS(hemisphere)] = is_odd
        sm.neuron_even[halco.HemisphereOnDLS(hemisphere)] = is_even
        config.set_buffer(halco.SourceMultiplexerOnReadoutSourceSelection(0), sm)
        config.enable_buffer_to_pad[halco.SourceMultiplexerOnReadoutSourceSelection(0)] = True
        builder.write(halco.ReadoutSourceSelectionOnDLS(), config)

        stadls.run(self._connection, builder.done())

    def set_readout(self, neuron_index: int, target="membrane"):
        neuron_coord = halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(neuron_index * self._neuron_size))

        builder = stadls.PlaybackProgramBuilder()
        mux_config = haldls.ReadoutSourceSelection.SourceMultiplexer()
        if neuron_coord.toNeuronColumnOnDLS() % 2:
            mux_config.neuron_odd[
                neuron_coord.toNeuronRowOnDLS().toHemisphereOnDLS()] = True
        else:
            mux_config.neuron_even[
                neuron_coord.toNeuronRowOnDLS().toHemisphereOnDLS()] = True

        config = haldls.ReadoutSourceSelection()
        config.set_buffer(
            halco.SourceMultiplexerOnReadoutSourceSelection(0),
            mux_config)
        builder.write(halco.ReadoutSourceSelectionOnDLS(), config)

        for c in halco.iter_all(halco.AtomicNeuronOnDLS):
            config = self._neuron_calib.neurons[c]
            config.readout.enable_buffered_access = (c == neuron_coord)
            config.readout.source = getattr(lola.AtomicNeuron.Readout.Source, target)

            builder.write(c.toNeuronConfigOnDLS(), config.asNeuronConfig())

        stadls.run(self._connection, builder.done())

    def set_spiking(self, spiking=True):
        self.structure = [strobe.backend.LayerSize(size=layer, spiking=spiking)
                          for layer in self.structure]
        self.configure()
        if hasattr(self, "calib_values"):
            self.config_postcalib(self.calib_values)

    def run(self, input_spikes, n_samples=None, duration=None, measure_power=False,
            record_madc=False, experiment_builder=None, fast_eval=False, record_timings=False):
        # input_spikes = [np.empty((0, 2))]
        if record_timings:
            # input_spikes = [input_spikes[0][0:5]]
            timer = utils.TIMER("====")
            timer_very = utils.TIMER("....")
        # TODO: do this more properly. first 256 neurons are the on chip neurons -> can give bias for all layers now)
        input_spikes[0][:, 1] += 256

        builder = stadls.PlaybackProgramBuilder()

        # # PPU stuff
        # ppu_control_reg_reset = haldls.PPUControlRegister()
        # ppu_control_reg_reset.force_clock_off = True
        # for ppu in range(2):
        #     builder.write(halco.PPUControlRegisterOnDLS(ppu), ppu_control_reg_reset)

        # enable recurrent connections
        # TODO: for continuous experiments we need to have recurrency
        if record_timings:
            print("due to timing measurement don't enable recurrency: no multiple experiments possible")
        else:
            builder.copy_back(enable_recurrency_builder)

        # extend recording a bit
        if record_madc:
            duration += 150e-6
        # arm MADC
        if record_madc:
            madc_control = haldls.MADCControl()
            madc_control.enable_power_down_after_sampling = False
            madc_control.start_recording = False
            madc_control.wake_up = True
            madc_control.enable_pre_amplifier = True
            madc_control.enable_continuous_sampling = True

            builder.write(halco.MADCControlOnDLS(), madc_control)

        # sync time
        builder.write(halco.SystimeSyncOnFPGA(), haldls.SystimeSync(True))
        builder.write(halco.TimerOnDLS(), haldls.Timer())
        builder.block_until(halco.TimerOnDLS(), 100)

        if record_timings:
            time_ticket_start = builder.read(halco.EventRecordingConfigOnFPGA())

        if record_madc:
            # trigger MADC sampling
            madc_control.start_recording = True
            builder.write(halco.MADCControlOnDLS(), madc_control)

        # c = haldls.CrossbarOutputConfig()
        # c.enable_event_counter[halco.CrossbarOutputOnDLS(0)] = True
        # c.enable_event_counter[halco.CrossbarOutputOnDLS(1)] = True
        # c.enable_event_counter[halco.CrossbarOutputOnDLS(2)] = True
        # c.enable_event_counter[halco.CrossbarOutputOnDLS(3)] = True
        # builder.write(halco.CrossbarOutputConfigOnDLS(), c)

        event_config = haldls.EventRecordingConfig()
        event_config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        hw_batch_size = len(input_spikes)
        if hw_batch_size > self.max_hw_batch_size:
            raise IndexError("The hardware batch size (inferred from the input spike trains) is \
                    larger than the maximum ({self.max_hw_batch_size}).")

        times = input_spikes[0][:, 0] + self._timing_offset
        labels = input_spikes[0][:, 1].astype(np.int)

        # shift inputs in case the first layer is recurrent
        labels += self._input_shift

        if experiment_builder is not None:
            builder.merge_back(experiment_builder)
        # for i in range(10):
        #     for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        #         builder.write(coord.toNeuronResetOnDLS(), haldls.NeuronReset())
        if record_timings:
            timer.time("stuff")

        # print(f"number spikes {len(times)}, number unique times {len(np.unique(times))} "
        #       f"difference {len(times) - len(np.unique(times))}")

        if record_timings:
            time_ticket_spikes_start = builder.read(halco.EventRecordingConfigOnFPGA())
        tmp = self._routing.generate_spike_train(times, labels)
        if record_timings:
            timer.time("gonzales generating spike train")
        builder.merge_back(tmp)
        if record_timings:
            time_ticket_spikes_end = builder.read(halco.EventRecordingConfigOnFPGA())

        builder.block_until(
            halco.TimerOnDLS(),
            int(duration * 1e6 * fisch.fpga_clock_cycles_per_us))

        if record_madc:
            # stop MADC
            madc_control.start_recording = False
            madc_control.stop_recording = True
            builder.write(halco.MADCControlOnDLS(), madc_control)

        # measure power consumption
        if measure_power:
            tickets = {}
            for ina in halco.iter_all(halco.INA219StatusOnBoard):
                tickets[ina] = builder.read(ina)

        builder.block_until(halco.BarrierOnFPGA(), haldls.Barrier.omnibus)

        event_config = haldls.EventRecordingConfig()
        event_config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(), event_config)

        # disable recurrent connections
        builder.copy_back(disable_recurrency_builder)

        builder.write(halco.TimerOnDLS(), haldls.Timer())
        builder.block_until(halco.TimerOnDLS(), 10000)

        if record_timings:
            # print(f" builder has size {builder.size_to_fpga()}")
            timer.time("stuff")

            time_ticket_end = builder.read(halco.EventRecordingConfigOnFPGA())
            builder.block_until(halco.BarrierOnFPGA(), haldls.Barrier.omnibus)

        # # PPU stuff
        # ppu_control_reg_reset = haldls.PPUControlRegister()
        # ppu_control_reg_reset.force_clock_off = False
        # for ppu in range(2):
        #     builder.write(halco.PPUControlRegisterOnDLS(ppu), ppu_control_reg_reset)

        event_counter_tickets = [builder.read(halco.CrossbarOutputOnDLS(i)) for i in range(4)]
        builder.block_until(halco.BarrierOnFPGA(), haldls.Barrier.omnibus)

        program = builder.done()
        if record_timings:
            timer.time("builder.done")

        # print(program)

        if record_timings:
            timer_very.time("from start")
        response_of_stadls = stadls.run(self._connection, program)
        # for h in program.highspeed_link_notifications:
        #     print(f"++++++++++++++++++++++++{h}")
        if record_timings:
            timer.time("stadls.run")
            timer_very.time("stadls.run")
            print(response_of_stadls)

            # print(f"start {time_ticket_start.fpga_time.value()} "
            #       f"end {time_ticket_end.fpga_time.value()}"
            #       f"difference {time_ticket_end.fpga_time.value() - time_ticket_start.fpga_time.value()}"
            #       f"time {(time_ticket_end.fpga_time.value() - time_ticket_start.fpga_time.value()) * 8e-9}")
            print(f"....specific time around spikes {(time_ticket_spikes_end.fpga_time.value() - time_ticket_start.fpga_time.value()) * 8e-9}s")

        # tmp = [i.get().value for i in event_counter_tickets]
        # print(f"send in spikes {len(input_spikes[0])}")
        # print(tmp)
        # print(f"spike number {np.sum(tmp)}")

        if measure_power:
            total_power = 0.0
            for k, v in tickets.items():
                total_power += v.get().toUncalibratedPower().calculate()
            pprint({k: f"{v.get().toUncalibratedPower().calculate() * 1000:.1f}" for k, v in tickets.items()})
            print(f".... POWER {total_power * 1000:.0f}")
            # # lst = list(np.load("powers_idle.npy"))
            # lst = list(np.load("powers_dynamic.npy"))
            # lst.append([v.get().toUncalibratedPower().calculate() * 1000 for k, v in tickets.items()])
            # np.save("powers_idle.npy", lst)
            # # np.save("powers_dynamic.npy", lst)

        if record_timings:
            timer.time("stuff")
        spike_times, spike_labels = self._routing.transform_events_from_chip(program.spikes.to_numpy())
        if record_timings:
            timer.time("transform events")

        raw_spikes = np.stack([spike_times, spike_labels]).T
        raw_spikes[:, 0] -= self._timing_offset
        if record_timings:
            timer.time("rawing")

        # group spikes according to layers
        spikes = [[] for l in range(len(self.structure) - 1)]
        for b in range(hw_batch_size):
            if record_timings:
                timer_in = utils.TIMER("======")
            b_begin = b * self._timing_separation
            b_end = (b + 1) * self._timing_separation
            mask = (raw_spikes[:, 0] > b_begin) & (raw_spikes[:, 0] < b_end)

            if record_timings:
                timer_in.time("stuff")
            dissected_spikes = raw_spikes[mask, :]
            if record_timings:
                timer_in.time("masking")

            boundaries = np.hstack([np.zeros(1, dtype=int), np.array(self.structure[1:]).cumsum()])
            for l in range(len(self.structure) - 1):
                if fast_eval and l != len(self.structure) - 2:
                    spikes[l].append(np.empty((0, 2)))
                    continue
                layer_mask = (dissected_spikes[:, 1] >= boundaries[l]) & (dissected_spikes[:, 1] < boundaries[l + 1])
                s = dissected_spikes[layer_mask, :]

                # subtract timing offset and population indices
                s[:, 0] -= b_begin
                s[:, 1] -= boundaries[l]
                spikes[l].append(s)

            if (raw_spikes[:, 1] >= boundaries[-1]).any():
                print("Received spikes from unused neurons!")
            if record_timings:
                timer_in.time("loop")

        if record_madc:
            samples = program.madc_samples.to_numpy()
            # TODO: there seems to be an offset (100e-6 is good)
            times = samples["chip_time"][10:] / 125 * 1e-6 - self._timing_offset
            trace = samples["value"][10:].astype(np.float) * 2e-3
            # fig, ax = plt.subplots(1, 1)
            # ax.plot(times, trace)
            # fig.savefig("tmp.png")
            # plt.close(fig)

            self._madc_samples = np.stack([times, trace]).T

        if record_timings:
            timer.time("end")
            timer_very.time("till end")
        return spikes, self._madc_samples if record_madc else None

    def load_ppu_program(self, program_path):
        # load PPU program
        elf_file = lola.PPUElfFile(program_path)
        elf_symbols = elf_file.read_symbols()

        self._ppu_n_ppus = elf_symbols["n_ppus"].coordinate
        self._ppu_ppu_id = elf_symbols["ppu_id"].coordinate
        self._ppu_duration_coordinate = elf_symbols["duration"].coordinate
        self._ppu_signal_coordinate = elf_symbols["command"].coordinate
        self._ppu_n_samples_coordinate = elf_symbols["n_samples"].coordinate

        # load and prepare ppu program
        builder = stadls.PlaybackProgramBuilder()

        ppu_control_reg_run = haldls.PPUControlRegister()
        ppu_control_reg_run.inhibit_reset = True

        ppu_control_reg_reset = haldls.PPUControlRegister()
        ppu_control_reg_reset.inhibit_reset = False

        program = elf_file.read_program()
        program_on_ppu = halco.PPUMemoryBlockOnPPU(
            halco.PPUMemoryWordOnPPU(0),
            halco.PPUMemoryWordOnPPU(program.size() - 1)
        )

        for ppu in range(2):
            program_on_dls = halco.PPUMemoryBlockOnDLS(program_on_ppu,
                                                       halco.PPUOnDLS(ppu))

            # ensure PPU is in reset state
            builder.write(halco.PPUControlRegisterOnDLS(ppu), ppu_control_reg_reset)

            # manually initialize memory where symbols will lie, issue #3477
            for _name, symbol in elf_symbols.items():
                value = haldls.PPUMemoryBlock(symbol.coordinate.toPPUMemoryBlockSize())
                symbol_on_dls = halco.PPUMemoryBlockOnDLS(symbol.coordinate,
                                                          halco.PPUOnDLS(ppu))
                builder.write(symbol_on_dls, value)

            builder.write(program_on_dls, program)
            builder.write(halco.PPUControlRegisterOnDLS(ppu), ppu_control_reg_run)

            builder.write(
                halco.PPUMemoryWordOnDLS(self._ppu_n_ppus[0], halco.PPUOnDLS(ppu)),
                haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(self._n_vectors)))

            builder.write(
                halco.PPUMemoryWordOnDLS(self._ppu_ppu_id[0], halco.PPUOnDLS(ppu)),
                haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(ppu)))

        # stop second PPU if it is not used
        if True:
            builder.write(program_on_dls, program)
            builder.write(halco.PPUControlRegisterOnDLS(1), ppu_control_reg_run)
            command = haldls.PPUMemoryWord(haldls.PPUMemoryWord.Value(PPUSignal.HALT.value))
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_signal_coordinate[0], halco.PPUOnDLS(0)), command)
            builder.write(halco.PPUMemoryWordOnDLS(self._ppu_signal_coordinate[0], halco.PPUOnDLS(1)), command)

        builder.block_until(halco.BarrierOnFPGA(), haldls.Barrier.omnibus)

        stadls.run(self._connection, builder.done())
