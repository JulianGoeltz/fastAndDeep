import argparse
import numpy as np
import os
import os.path as osp
import quantities as pq

import pyhalco_hicann_dls_vx_v3 as halco
import pystadls_vx_v3 as stadls
import pyhxcomm_vx as hxcomm
import calix.common
import calix.spiking.neuron
from dlens_vx_v3.sta import PlaybackProgramBuilderDumper, to_portablebinary

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str)

args = parser.parse_args()

fold = osp.dirname(args.output)
if not osp.isdir(fold):
    os.mkdir(fold)

targets = {
        "leak": 80,
        "reset": 80,
        "threshold": 150,
        "tau_mem": 6e-6,
        "tau_syn": 6e-6,
    # CAUTION: this is the setting for smaller networks, as also used with pynn
    # for larger networks with neurons that are less sensitive, use
        # "i_synin_gm": 700,
    # as well as in py/hx_settings.yaml
        # threshold: 1.52
        "i_synin_gm": 550,
        "membrane_capacitance": 63,
        "synapse_dac_bias": 800,
        "refractory_time": 16e-6
        }

targets_in_weird_units = targets.copy()
targets_in_weird_units["tau_mem"] *= pq.s
targets_in_weird_units["tau_syn"] *= pq.s
targets_in_weird_units["refractory_time"] *= pq.s

with hxcomm.ManagedConnection() as connection:
    stateful_connection = calix.common.base.StatefulConnection(connection)

    # calibrate CADCs
    cadc_result = calix.common.cadc.calibrate(stateful_connection)


    # calibrate neurons
    neuron_target = calix.spiking.neuron.NeuronCalibTarget(**targets_in_weird_units)
    target = calix.spiking.SpikingCalibTarget(neuron_target=neuron_target)

    neuron_result = calix.spiking.neuron.calibrate(
        stateful_connection,
        target=neuron_target)

    np.savez(args.output, cadc=cadc_result, neuron=neuron_result, targets=targets)

    target_file_pbin = osp.splitext(args.output)[0] + ".pbin"
    builder = PlaybackProgramBuilderDumper()
    neuron_result.apply(builder)
    with open(target_file_pbin, mode="wb") as target:
        target.write(to_portablebinary(builder.done()))
