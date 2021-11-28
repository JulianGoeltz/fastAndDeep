#!python
"""Testing the forward pass of the network

Using the fwd pass from utils.py with the analytical formula for a comparison
with a numerical integrator.
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import unittest

import datasets
import training
import utils


num_input = 120
num_output = 30
batch_size = 50
input_binsize = 0.01  # doubles as initial resolution
sim_time = 3.
debug = False

resols = input_binsize / 2**np.arange(0, 7)

seed = np.random.randint(0, 100000)

weights_mean = 0.06
weights_std = weights_mean * 4.
weights_normal = True

sim_params_eventbased = {
    'leak': 0.,
    'g_leak': 1.,
    'threshold': 1.,
    'tau_syn': 1.,
    # or e.g. the following, but adapt weights (weights_mean = 0.10)
    # 'leak': 0.7,
    # 'g_leak': 4.,
    # 'threshold': 1.08,
    # 'tau_syn': 1.,
}


class TestIntegratorVsTimebased(unittest.TestCase):
    def test_nograd(self):
        print(f"####### Using seed {seed} ########")
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = utils.get_default_device()

        sim_params_integrator = copy.deepcopy(sim_params_eventbased)
        sim_params_integrator.update({
            'use_forward_integrator': True,
            'resolution': input_binsize,
            'sim_time': sim_time,
        })

        print("### generating data and weights")
        times_input = torch.empty((batch_size, num_input), device=device)
        # torch.nn.init.normal_(times_input, mean=0.2)
        torch.nn.init.uniform_(times_input, a=0.1, b=1.5)
        times_input = torch.abs(times_input)
        print("#### binning inputs")
        times_input = (times_input / input_binsize).round() * input_binsize
        if debug:
            print(f"using inputs {times_input}")

        weights = torch.empty((num_input, num_output), device=device)
        if weights_normal:
            torch.nn.init.normal_(weights, mean=weights_mean, std=weights_std)
        else:
            torch.nn.init.uniform_(weights, a=weights_mean - 2 * weights_std, b=weights_mean + 2 * weights_std)

        print("### generating layers")
        layer_eventbased = utils.EqualtimeLayer(
            num_input, num_output, sim_params_eventbased, weights,
            device, 0)
        layer_integrator = utils.EqualtimeLayer(
            num_input, num_output, sim_params_integrator, weights,
            device, 0)

        def cutitdown(to_argmax, times_input, weights,
                      outputs_eventbased, outputs_integrator):
            """for plotting a membrane trace out of many

            looks at argmax of to_argmax"""
            argmaxed = np.argmax(to_argmax)
            critical = np.unravel_index(argmaxed, shape=to_argmax.shape)
            # critical = (33, 34)  # for 0.01
            # critical = (49, 21)   # for 0.001
            # critical = (5, 25)  # 0.1
            print(f"at point {critical} where argument is largest int has {outputs_integrator[critical]}, "
                  f"event has {outputs_eventbased[critical]}")
            times_input = times_input[critical[0]].reshape((1, -1))
            weights = weights[:, critical[1]].reshape((-1, 1))
            dbg_layer_integrator = utils.EqualtimeLayer(
                num_input, 1, sim_params_integrator, weights,
                device, 0)
            dbg_outputs_integrator = dbg_layer_integrator(
                times_input,
                output_times=outputs_eventbased[critical].reshape(1, -1))
            print(dbg_outputs_integrator)
            sys.exit()

        # eventbased will not change
        print("### one time eventbased forward pass")
        with torch.no_grad():
            outputs_eventbased = layer_eventbased(times_input)
            outputs_eventbased_inf = torch.isinf(outputs_eventbased)

        print("### looping integrator passed with different resolutions")
        differences_l1, differences_l2 = [], []
        int_infs = []
        for resol in resols:
            layer_integrator.sim_params['resolution'] = resol
            layer_integrator.sim_params['steps'] = int(np.ceil(sim_params_integrator['sim_time'] / resol))
            layer_integrator.sim_params['decay_syn'] = float(np.exp(-resol / sim_params_integrator['tau_syn']))
            layer_integrator.sim_params['decay_mem'] = float(np.exp(-resol / sim_params_integrator['tau_syn']))

            assert input_binsize >= layer_integrator.sim_params['resolution'], \
                "inputs are binned too weakly compared to resolution"

            print(f"#### forward pass for resol {resol}")
            with torch.no_grad():
                outputs_integrator = layer_integrator(times_input, outputs_eventbased)

            # handle infs
            if debug:
                print(f"##### event has {outputs_eventbased_inf.sum()} infs, "
                      f"integrator has {torch.isinf(outputs_integrator).sum()} infs")
            int_infs.append(torch.isinf(outputs_integrator).sum())

            # mean_difference = torch.mean(torch.abs(outputs_eventbased - outputs_integrator))
            # max_difference = torch.max(torch.abs(outputs_eventbased - outputs_integrator))
            difference_nonnan = outputs_eventbased - outputs_integrator
            difference_nonnan[torch.isnan(difference_nonnan)] = 0
            difference_l1 = torch.sum(torch.abs(difference_nonnan))
            differences_l1.append(difference_l1)
            difference_l2 = torch.sqrt(torch.sum((difference_nonnan)**2))
            differences_l2.append(difference_l2)
            if debug:
                print(f"#####               difference_l1 {difference_l1}")
                print(f"#####               difference_l2 {difference_l2}")

        print("### plotting")
        differences_l1 = np.array(differences_l1) / (batch_size * num_output)
        differences_l2 = np.array(differences_l2) / (batch_size * num_output)
        fig, ax = plt.subplots(1, 1)
        ax.plot(resols, differences_l1, marker='x', label="l1")
        ax.plot(resols, differences_l2, marker='x', label="l2")
        # ax.axvline(input_binsize, label="input_binsize")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("resolution of integrator")
        ax.set_title(f"bin size of input {input_binsize}, seed {seed}, {num_input} inputs\nweights"
                     f" {weights_mean}+-{weights_std}" if weights_normal else
                     f"in [{weights_mean - 2 * weights_std}, {weights_mean + 2 * weights_std}")
        ax.set_ylabel("normed difference per output time\n(/ batch_size / num_output)")
        ax2 = ax.twinx()
        int_infs = np.array(int_infs) / (batch_size * num_output)
        ax2.bar(resols, int_infs, alpha=0.4, width=0.5 * resols,
                label="number of infs in integrator", color='C3')
        ax2.axhline(float(outputs_eventbased_inf.sum()) / (batch_size * num_output),
                    label="number of infs in evba", color='C3')
        ax2.set_ylim(min(int_infs) - 0.1, max(int_infs) + 0.1)
        ax2.set_ylabel('fraction infs')

        ax.legend()
        fig.tight_layout()
        fig.savefig('debug_integrator.png')

        differences_l1[np.isinf(differences_l1)] = 1000.
        differences_l2[np.isinf(differences_l2)] = 1000.
        self.assertTrue(
            np.all(np.diff(differences_l1) <= 0),
            "The l1 norms are not decreasing with increasing integrator resolution, see plot.")
        self.assertTrue(
            np.all(np.diff(differences_l2) <= 0),
            "The l2 norms are not decreasing with increasing integrator resolution, see plot.")


if __name__ == '__main__':
    unittest.main()
