#!python3
import numba
import numpy as np
import os
import os.path as osp
import sys
import time
import torch
import torch.nn
import torch.autograd
import yaml

torch.set_default_dtype(torch.float64)

import utils_spiketime_et as utils_spiketime


class EqualtimeFunctionEventbased(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights,
                neuron_params, device, output_times=None):
        """Class that calculates the EventBased spikes, and provides backward function

        Arguments:
            ctx: from torch, used for saving for backward pass
            input_spikes, input_weights: input that is used for calculations
            neuron_params: constants used in calculation
            device: torch specifics, for GPU use
            output_times: used only in combination with HicannX, that inherits the backward
                pass from this class, see below for details
        """
        # create causal set
        sort_indices = input_spikes.argsort(1)
        sorted_spikes = input_spikes.gather(1, sort_indices)
        sorted_weights = input_weights[sort_indices]
        # (in each batch) set weights=0 and spiketime=0 for neurons with inf spike time to prevent nans
        mask_of_inf_spikes = torch.isinf(sorted_spikes)
        sorted_spikes_masked = sorted_spikes.clone().detach()
        sorted_spikes_masked[mask_of_inf_spikes] = 0.
        sorted_weights[mask_of_inf_spikes] = 0.

        output_spikes = to_device(torch.ones(len(sorted_spikes), input_weights.size()[1]) * np.inf, device)

        tmp_output = utils_spiketime.get_spiketime(
            sorted_spikes_masked,
            sorted_weights,
            neuron_params, device)

        not_after_last_input = tmp_output < sorted_spikes.unsqueeze(-1)
        not_earlier_than_next = tmp_output > sorted_spikes.unsqueeze(-1).roll(-1, dims=1)
        not_earlier_than_next[:, -1, :] = 0.  # last has no subsequent spike

        tmp_output[not_after_last_input] = float('inf')
        tmp_output[not_earlier_than_next] = float('inf')

        output_spikes, causal_set_lengths = torch.min(tmp_output, dim=1)

        ctx.sim_params = neuron_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, output_spikes)
        if torch.isnan(output_spikes).sum() > 0:
            raise ArithmeticError("There are NaNs in the output times, this means a serious error occured")
        return output_spikes

    @staticmethod
    def backward(ctx, propagated_error):
        # recover saved values
        input_spikes, input_weights, output_spikes = ctx.saved_tensors
        sort_indices = input_spikes.argsort(1)
        sorted_spikes = input_spikes.gather(1, sort_indices)
        sorted_weights = input_weights[sort_indices]
        # with missing label spikes the propagated error can be nan
        propagated_error[torch.isnan(propagated_error)] = 0

        batch_size = len(input_spikes)
        number_inputs = input_weights.size()[0]
        number_outputs = input_weights.size()[1]

        dw_ordered, dt_ordered = utils_spiketime.get_spiketime_derivative(
            sorted_spikes, sorted_weights, ctx.sim_params, ctx.device, output_spikes)

        # retransform it in the correct way
        dw = to_device(torch.zeros(dw_ordered.size()), ctx.device)
        dt = to_device(torch.zeros(dt_ordered.size()), ctx.device)
        mask_from_spikeordering = torch.argsort(sort_indices)[:, :, np.newaxis].repeat((1, 1, number_outputs))

        dw = torch.gather(dw_ordered, 1, mask_from_spikeordering)
        dt = torch.gather(dt_ordered, 1, mask_from_spikeordering)

        error_to_work_with = propagated_error.view(
            propagated_error.size()[0], 1, propagated_error.size()[1])

        weight_gradient = dw * error_to_work_with

        if ctx.sim_params['max_dw_norm'] is not None:
            """ to prevent large weight changes, we identify output spikes with a very large dw
            this happens when neuron barely spikes, aka small changes determine whether it spikes or not
            technically, the membrane maximum comes close to the threshold,
            and the derivative at the threshold will vanish.
            as the derivative here are wrt the times, kinda a switch of axes (see LambertW),
            the derivatives will diverge in those cases."""
            weight_gradient_norms, _ = weight_gradient.abs().max(dim=1)
            gradient_jumps = weight_gradient_norms > ctx.sim_params['max_dw_norm']
            if gradient_jumps.sum() > 0:
                print(f"gradients too large (input size {number_inputs}), chopped the following:"
                      f"{weight_gradient_norms[gradient_jumps]}")
            weight_gradient = weight_gradient.permute([0, 2, 1])
            weight_gradient[gradient_jumps] = 0.
            weight_gradient = weight_gradient.permute([0, 2, 1])

        # averaging over batches to get final update
        weight_gradient = weight_gradient.sum(0)

        new_propagated_error = torch.bmm(
            dt,
            error_to_work_with.permute(0, 2, 1)
        ).view(batch_size, number_inputs)

        if torch.any(torch.isinf(weight_gradient)) or \
           torch.any(torch.isinf(new_propagated_error)) or \
           torch.any(torch.isnan(weight_gradient)) or \
           torch.any(torch.isnan(new_propagated_error)):
            print(f" wg nan {torch.isnan(weight_gradient).sum()}, inf {torch.isinf(weight_gradient).sum()}")
            print(f" new_propagated_error nan {torch.isnan(new_propagated_error).sum()}, "
                  f"inf {torch.isinf(new_propagated_error).sum()}")
            print('found nan or inf in propagated_error or weight_gradient, something is wrong oO')
            sys.exit()

        return new_propagated_error, weight_gradient, None, None, None


class EqualtimeFunctionIntegrator(EqualtimeFunctionEventbased):
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights,
                sim_params, device, output_times=None):
        """use a simple euler integration, then compare with a threshold to determine spikes"""
        batch_size, input_features = input_spikes.shape
        _, output_features = input_weights.shape

        input_transf = torch.zeros(tuple(input_spikes.shape) + (sim_params['steps'], ),
                                   device=device, requires_grad=False)
        input_times_step = (input_spikes / sim_params['resolution']).long()
        input_times_step[input_times_step > sim_params['steps'] - 1] = sim_params['steps'] - 1
        input_times_step[torch.isinf(input_spikes)] = sim_params['steps'] - 1

        # one-hot code input times for easier multiplication
        input_transf = torch.eye(sim_params['steps'], device=device)[
            input_times_step].reshape((batch_size, input_features, sim_params['steps']))

        charge = torch.einsum("abc,bd->adc", (input_transf, input_weights))

        # init is no synaptic current and mem at leak
        syn = torch.zeros((batch_size, output_features), device=device)
        mem = torch.ones((batch_size, output_features), device=device) * sim_params['leak']

        plotting = False
        all_mem = []
        # want to save spikes
        output_spikes = torch.ones((batch_size, output_features), device=device) * float('inf')
        for step in range(sim_params['steps']):
            # print(step)
            mem = sim_params['decay_mem'] * (mem - sim_params['leak']) \
                + 1. / sim_params['g_leak'] * syn * sim_params['resolution'] + sim_params['leak']
            syn = sim_params['decay_syn'] * syn + charge[:, :, step]

            # mask is a logical_and implemented by multiplication
            output_spikes[
                (torch.isinf(output_spikes) *
                 mem > sim_params['threshold'])] = step * sim_params['resolution']
            # reset voltage after spike for plotting
            # mem[torch.logical_not(torch.isinf(output_spikes))] = 0.
            if plotting:
                all_mem.append(mem.numpy())

        if plotting:
            import matplotlib.pyplot as plt
            import warnings
            if batch_size >= 9:
                fig, axes = plt.subplots(3, 3, figsize=(16, 10))
            else:
                fig, ax = plt.subplots(1, 1)
                axes = np.array([ax])
            all_mem = np.array(all_mem)
            np.save("membrane_trace.npy", all_mem)
            np.save("membrane_spike.npy", output_spikes)
            batch_to_plot = 0
            for batch_to_plot, ax in enumerate(axes.flatten()):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    ax.plot(np.arange(sim_params['steps']) * sim_params['resolution'],
                            all_mem[:, batch_to_plot, :])

                    ax.axhline(sim_params['leak'], color='black', lw=0.4)
                    ax.axhline(sim_params['threshold'], color='black', lw=1)
                    for i, sp in enumerate(output_spikes[batch_to_plot]):
                        ax.axvline(sp, color=f"C{i}", ls="-.", ymax=0.5)
                    if output_times is not None:
                        for i, ti in enumerate(output_times[batch_to_plot]):
                            ax.axvline(ti, color=f"C{i}", ls=":", ymin=0.5)

                    ax.set_ylim(
                        (sim_params['threshold'] - sim_params['leak']) * np.array((-1, 1.1)) + sim_params['leak'])

                    ax.set_ylabel(f"C{batch_to_plot}", fontweight='bold')
                    ax.yaxis.label.set_color(f"C{batch_to_plot}")
            fig.tight_layout()
            fig.savefig('debug_int.png')
            plt.close(fig)

        if torch.isnan(output_spikes).sum() > 0:
            raise ArithmeticError("There are NaNs in the output times, this means a serious error occured")

        ctx.sim_params = sim_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, output_spikes)

        return output_spikes


class EqualtimeFunctionHicannx(EqualtimeFunctionEventbased):
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights,
                sim_params, device, output_times):
        """output spikes are determined by the hardware"""
        ctx.sim_params = sim_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, output_times)

        return output_times.clone()


class EqualtimeLayer(torch.nn.Module):
    def __init__(self, input_features, output_features, sim_params, weights_init,
                 device, bias=0):
        """Setup up a layer of neurons

        Arguments:
            input_features, output_features: number of inputs/outputs
            sim_params: parameters used for simulation
            weights_init: if tuple it is understood as two lists of mean and std, otherwise an array of weights
            device: torch, gpu stuff
            bias: number of bias inputs
        """
        super(EqualtimeLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.sim_params = sim_params
        self.bias = bias
        self.device = device
        self.use_forward_integrator = sim_params.get('use_forward_integrator', False)
        if self.use_forward_integrator:
            assert 'resolution' in sim_params and 'sim_time' in sim_params
            self.sim_params['steps'] = int(np.ceil(sim_params['sim_time'] / sim_params['resolution']))
            self.sim_params['decay_syn'] = float(np.exp(-sim_params['resolution'] / sim_params['tau_syn']))
            self.sim_params['decay_mem'] = float(np.exp(-sim_params['resolution'] / sim_params['tau_syn']))

        self.weights = torch.nn.Parameter(torch.Tensor(input_features + bias, output_features))

        if isinstance(weights_init, tuple):
            self.weights.data.normal_(weights_init[0], weights_init[1])
        else:
            assert weights_init.shape == (input_features + bias, output_features)
            self.weights.data = weights_init

        self.use_hicannx = sim_params.get('use_hicannx', False)

    def forward(self, input_times, output_times=None):
        # depending on configuration use either eventbased, integrator or the hardware
        if not self.use_hicannx:
            assert output_times is None
            if self.use_forward_integrator:
                return EqualtimeFunctionIntegrator.apply(input_times, self.weights,
                                                         self.sim_params,
                                                         self.device)
            else:
                return EqualtimeFunctionEventbased.apply(input_times, self.weights,
                                                         self.sim_params,
                                                         self.device)
        else:
            return EqualtimeFunctionHicannx.apply(input_times, self.weights,
                                                  self.sim_params,
                                                  self.device,
                                                  output_times)


def bias_inputs(number_biases, t_bias=[0.05]):
    assert len(t_bias) == number_biases
    times = [t_bias[i] for i in range(number_biases)]
    return torch.tensor(times)


@numba.jit(nopython=True, parallel=True, cache=True)
def hx_spiketrain_purgeinf(spiketrain):
    return spiketrain[spiketrain[:, 0] < np.inf]


@numba.jit(nopython=True, parallel=True, cache=True)
def hx_spiketrain_create(batch, single_simtime, scale_times, offsetter, per_pattern_sort):
    """thread minibatch by offsetting patterns. return ordered spiketrain"""
    num_batch, num_neur = batch.shape
    # first dimension is batch_id, along this offset is implemented
    long_input_times = batch + offsetter * single_simtime

    # create spiketrain out of input times (aka matrix of (neuron id, spike times)
    spiketrain = np.zeros((num_batch * num_neur, 2))

    # sorting, done in parallel: each pattern can be sorted individually
    for i in numba.prange(batch.shape[0]):
        sorting = np.argsort(long_input_times[i, :])
        spiketrain[i * num_neur:(i + 1) * num_neur, 1] = sorting  # neuron ids
        spiketrain[
            i * num_neur:(i + 1) * num_neur, 0
        ] = long_input_times[i][sorting] * scale_times

    return spiketrain, num_batch * single_simtime * scale_times


@numba.jit(nopython=True, parallel=True, cache=True)
def hx_spiketrain_disect(spikes_t, spikes_id, single_simtime, num_batch, num_neurons, first_spikes, scale_times):
    """turns spiketrain into batched tensor

    i.e. slicing with single_simtime, using only first spikes.
    spikes times (t) and ids are given separatley for jitting"""
    def get_spike_slices(spikes_t, spikes_id, num_slices, time_per_slice):
        """
        Spikes array is expected to be sorted by spiketime!
        Returns:
            Iterator over (neuron_id, spiketime) slices where neuron_id[i] fired at
            spiketime[i].
        """
        idx_batch = np.arange(num_slices)
        idx_start = np.searchsorted(
            spikes_t, idx_batch * time_per_slice, side="left"
        )
        idx_stop = np.searchsorted(
            spikes_t, (idx_batch + 1) * time_per_slice, side="left"
        )
        lst = []
        for i_start, i_stop in zip(idx_start, idx_stop):
            lst.append((
                spikes_id[i_start:i_stop],
                spikes_t[i_start:i_stop]))
        return lst

    # spiketrain = spiketrain[np.argsort(spiketrain[:, 0])]
    hw_simtime = single_simtime * scale_times
    # iterating over the slices
    for i_batch, (neuron_ids, spike_times) in enumerate(
        get_spike_slices(spikes_t, spikes_id, num_batch, hw_simtime)
    ):
        spike_times -= i_batch * hw_simtime
        # go through all spike (id, time) pairs in that slice
        for n_id, t in zip(neuron_ids, spike_times):
            # if id is sensible, and given time is earlier then registered, use it
            if n_id >= 0 and n_id < num_neurons:
                first_spikes[i_batch, n_id] = min(first_spikes[i_batch, n_id], t)
    return first_spikes / scale_times


def network_load(path, basename, device):
    net = to_device(torch.load(osp.join(path, basename + "_network.pt"),
                               map_location=device),
                    device)
    if net.use_hicannx:
        error = ""
        if 'SLURM_HARDWARE_LICENSES' not in os.environ:
            error = "### some evaluations need hw access, run on the specific hw"
        # verify we are on the correct hicann
        if not osp.isfile(osp.join(path, basename + '_hw_licences.txt')):
            error = "### It seems you want to continue software training on hardware"
        else:
            with open(osp.join(path, basename + '_hw_licences.txt'), 'r') as f:
                hw_licences = f.read()
            if hw_licences != os.environ.get('SLURM_HARDWARE_LICENSES'):
                error = ("You have to continue training on the same chip, instead got: "
                         f"licence read from file '{hw_licences}', from env "
                         f"'{os.environ.get('SLURM_HARDWARE_LICENSES')}'")

        if error == "":
            net.hx_lastsetweights = [torch.full(l.weights.data.shape, -64) for l in net.layers]
            net.init_hicannx(device)
        else:
            print(f"#### untrained network loaded -> not initialising, also got error:\n{error}")

    net.device = get_default_device()
    return net


class LossFunction(torch.nn.Module):
    def __init__(self, number_labels, tau_syn, xi, alpha, beta, device):
        super().__init__()
        self.number_labels = number_labels
        self.tau_syn = tau_syn
        self.xi = xi
        self.alpha = alpha
        self.beta = beta
        self.device = device
        return

    def forward(self, label_times, true_label):
        label_idx = to_device(true_label.clone().type(torch.long).view(-1, 1), self.device)
        true_label_times = label_times.gather(1, label_idx).flatten()
        loss = torch.log(torch.exp(-1 * label_times / (self.xi * self.tau_syn)).sum(1)) \
            + true_label_times / (self.xi * self.tau_syn)
        regulariser = self.alpha * torch.exp(true_label_times / (self.beta * self.tau_syn))
        total = loss + regulariser
        total[true_label_times == np.inf] = 100.
        return total.mean()

    def select_classes(self, outputs):
        firsts = outputs.argmin(1)
        firsts_reshaped = firsts.view(-1, 1)
        # count how many firsts had inf or nan as value
        nan_mask = torch.isnan(torch.gather(outputs, 1, firsts_reshaped)).flatten()
        inf_mask = torch.isinf(torch.gather(outputs, 1, firsts_reshaped)).flatten()
        # set firsts to -1 so that they cannot be counted as correct
        firsts[nan_mask] = -1
        firsts[inf_mask] = -1
        return firsts


class LossFunctionMSE(torch.nn.Module):
    def __init__(self, number_labels, tau_syn, correct, wrong, device):
        super().__init__()
        self.number_labels = number_labels
        self.tau_syn = tau_syn
        self.device = device

        self.t_correct = self.tau_syn * correct
        self.t_wrong = self.tau_syn * wrong
        return

    def forward(self, label_times, true_label):
        label_idx = to_device(true_label.clone().type(torch.long).view(-1, 1), self.device)
        true_label_times = label_times.gather(1, label_idx).flatten()

        target = torch.eye(self.number_labels)[true_label] * (self.t_correct - self.t_wrong) + self.t_wrong
        loss = 1. / 2. * (label_times - target)**2
        loss[true_label_times == np.inf] = 100.
        return loss.mean()

    def select_classes(self, outputs):
        closest_to_target = torch.abs(outputs - self.t_correct).argmin(1)
        ctt_reshaped = closest_to_target.view(-1, 1)
        # count how many firsts had inf or nan as value
        nan_mask = torch.isnan(torch.gather(outputs, 1, ctt_reshaped)).flatten()
        inf_mask = torch.isinf(torch.gather(outputs, 1, ctt_reshaped)).flatten()
        # set firsts to -1 so that they cannot be counted as correct
        closest_to_target[nan_mask] = -1
        closest_to_target[inf_mask] = -1
        return closest_to_target


def get_default_device():
    # Pick GPU if avialable, else CPU
    if torch.cuda.is_available():
        print("Using GPU, Yay!")
        return torch.device('cuda')
    else:
        print("Using CPU, Meh!")
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    # Wrap a dataloader to move data to device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


class TIMER:
    def __init__(self, pre=""):
        self.timestmp = time.perf_counter()
        self.pre = pre

    def time(self, label=""):
        if self.timestmp > 0:
            print(f"{self.pre}{label} {(time.perf_counter() - self.timestmp) * 1e3:.0f}ms")
        self.timestmp = time.perf_counter()
