#!python3
import copy
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

import utils_spiketime


class NeuronFunctionEventbased(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights,
                neuron_params, training_params, device, output_times=None):
        """Class that calculates the EventBased spikes, and provides backward function

        Arguments:
            ctx: from torch, used for saving for backward pass
            input_spikes, input_weights: input that is used for calculations
            neuron_params: layerwise constants used in calculation
            training_params: more global constants (used in integrator)
            device: torch specifics, for GPU use
            output_times: used only in combination with HicannX, that inherits the backward
                pass from this class, see below for details
        """
        n_batch, n_presyn, n_postsyn = input_spikes.shape
        n_presyn, n_postsyn2 = input_weights.shape
        assert n_postsyn == n_postsyn2

        # create causal set
        sort_indices = input_spikes.argsort(1)
        sorted_spikes = input_spikes.gather(1, sort_indices)
        broadcasted_weights = input_weights.unsqueeze(0).repeat([n_batch, 1, 1])
        sorted_weights = broadcasted_weights.gather(1, sort_indices)
        # (in each batch) set weights=0 and spiketime=0 for neurons with inf spike time to prevent nans
        mask_of_inf_spikes = torch.isinf(sorted_spikes)
        sorted_spikes_masked = sorted_spikes.clone().detach()
        sorted_spikes_masked[mask_of_inf_spikes] = 0.
        sorted_weights[mask_of_inf_spikes] = 0.

        tmp_output = utils_spiketime.get_spiketime(
            sorted_spikes_masked,
            sorted_weights,
            neuron_params, device)

        not_after_last_input = tmp_output < sorted_spikes
        not_earlier_than_next = tmp_output > sorted_spikes.roll(-1, dims=1)
        not_earlier_than_next[:, -1, :] = 0.  # last has no subsequent spike

        tmp_output[not_after_last_input] = float('inf')
        tmp_output[not_earlier_than_next] = float('inf')

        output_spikes, causal_set_lengths = torch.min(tmp_output, dim=1)

        # add noise
        if neuron_params.get('jitter', 0.) != 0:
            noise = torch.randn(
                output_spikes.size(), device=output_spikes.device, generator=utils_spiketime.gen_hwAware
            ) * neuron_params.get('jitter', 0.)
            output_spikes += noise

        ctx.neuron_params = neuron_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, output_spikes)

        if torch.isnan(output_spikes).sum() > 0:
            raise ArithmeticError(
                "There are NaNs in the output times, this means a serious error occured"
                f" (in layer with presyn {n_presyn} and postsyn {n_postsyn}"
            )
        return output_spikes

    @staticmethod
    def backward(ctx, propagated_error):
        # recover saved values
        input_spikes, input_weights, output_spikes = ctx.saved_tensors

        n_batch, n_presyn, n_postsyn = input_spikes.shape
        n_presyn, n_postsyn = input_weights.shape

        sort_indices = input_spikes.argsort(1)
        sorted_spikes = input_spikes.gather(1, sort_indices)
        broadcasted_weights = input_weights.unsqueeze(0).repeat([n_batch, 1, 1])
        sorted_weights = broadcasted_weights.gather(1, sort_indices)
        # with missing label spikes the propagated error can be nan
        propagated_error[torch.isnan(propagated_error)] = 0

        batch_size = len(input_spikes)
        number_inputs = input_weights.size()[0]
        number_outputs = input_weights.size()[1]

        dw_ordered, dt_ordered = utils_spiketime.get_spiketime_derivative(
            sorted_spikes, sorted_weights, ctx.neuron_params, ctx.device, output_spikes)

        # retransform it in the correct way
        dw = to_device(torch.zeros(dw_ordered.size()), ctx.device)
        dt = to_device(torch.zeros(dt_ordered.size()), ctx.device)
        mask_from_spikeordering = torch.argsort(sort_indices, dim=1)

        dw = torch.gather(dw_ordered, 1, mask_from_spikeordering)
        dt = torch.gather(dt_ordered, 1, mask_from_spikeordering)

        error_to_work_with = propagated_error.view(
            propagated_error.size()[0], 1, propagated_error.size()[1])

        weight_gradient = dw * error_to_work_with

        if ctx.neuron_params['max_dw_norm'] is not None:
            """ to prevent large weight changes, we identify output spikes with a very large dw
            this happens when neuron barely spikes, aka small changes determine whether it spikes or not
            technically, the membrane maximum comes close to the threshold,
            and the derivative at the threshold will vanish.
            as the derivative here are wrt the times, kinda a switch of axes (see LambertW),
            the derivatives will diverge in those cases."""
            weight_gradient_norms, _ = weight_gradient.abs().max(dim=1)
            gradient_jumps = weight_gradient_norms > ctx.neuron_params['max_dw_norm']
            if gradient_jumps.sum() > 0:
                print(f"gradients too large (input size {number_inputs}), chopped the following:"
                      f"{weight_gradient_norms[gradient_jumps]}")
            weight_gradient = weight_gradient.permute([0, 2, 1])
            weight_gradient[gradient_jumps] = 0.
            weight_gradient = weight_gradient.permute([0, 2, 1])

        # averaging over the batch to get final update
        weight_gradient = weight_gradient.sum(0)

        new_propagated_error = dt * error_to_work_with

        if torch.any(torch.isinf(weight_gradient)) or \
           torch.any(torch.isinf(new_propagated_error)) or \
           torch.any(torch.isnan(weight_gradient)) or \
           torch.any(torch.isnan(new_propagated_error)):
            print(f" wg nan {torch.isnan(weight_gradient).sum()}, inf {torch.isinf(weight_gradient).sum()}")
            print(f" new_propagated_error nan {torch.isnan(new_propagated_error).sum()}, "
                  f"inf {torch.isinf(new_propagated_error).sum()}")
            print('found nan or inf in propagated_error or weight_gradient, something is wrong oO')
            print("There are safeguards in place to keep this from happening, "
                  "check utils_spiketime.py for details and an explanation.");
            sys.exit()

        return new_propagated_error, weight_gradient, None, None, None, None


class NeuronFunctionIntegrator(NeuronFunctionEventbased):
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights,
                neuron_params, training_params, device, output_times=None):
        """use a simple euler integration, then compare with a threshold to determine spikes"""
        n_batch, n_presyn, n_postsyn = input_spikes.shape
        n_presyn, n_postsyn2 = input_weights.shape

        input_transf = torch.zeros(tuple(input_spikes.shape) + (training_params['steps'], ),
                                   device=device, requires_grad=False)
        input_times_step = (input_spikes / training_params['resolution']).long()
        input_times_step[input_times_step > training_params['steps'] - 1] = training_params['steps'] - 1
        input_times_step[torch.isinf(input_spikes)] = training_params['steps'] - 1

        # one-hot code input times for easier multiplication
        input_transf = torch.eye(training_params['steps'], device=device)[input_times_step]

        charge = torch.einsum("abcd,bc->acd", (input_transf, input_weights))

        # init is no synaptic current and mem at leak
        syn = torch.zeros((n_batch, n_postsyn), device=device)
        mem = torch.ones((n_batch, n_postsyn), device=device) * neuron_params['leak']

        plotting = False
        all_mem = []
        # want to save spikes
        output_spikes = torch.ones((n_batch, n_postsyn), device=device) * float('inf')
        for step in range(training_params['steps']):
            # print(step)
            # 1 / C_mem = 1 would appear multiplying syn
            mem = (
                neuron_params['decay_mem'] * (mem - neuron_params['leak']) \
                + neuron_params['leak']
                + syn * training_params['resolution']
            )
            """The above is the Euler approximation for solving the ODE.
            The fully correct one is written below (first for tau_m != tau_s,
            then for tau_m == tau_s).
            For generality, simplicity and speed, we only use the
            approximated version,
            """
            # tau_m != tau_s
            # mem = (
            #     neuron_params['decay_mem'] * (mem - neuron_params['leak'])
            #     + neuron_params['leak']
            #     + (
            #         syn / neuron_params['g_leak'] * neuron_params['tau_syn']
            #         / (neuron_params['tau_syn'] - neuron_params['tau_mem'])
            #         * (neuron_params['decay_syn'] - neuron_params['decay_mem'])
            #     )
            # )
            # tau_m == tau_s
            # mem = (
            #     neuron_params['decay_mem'] * (mem - neuron_params['leak'])
            #     + neuron_params['leak']
            #     + (
            #         syn
            #         * training_params['resolution']
            #         * neuron_params['decay_syn']
            #     )
            # )
            syn = neuron_params['decay_syn'] * syn + charge[:, :, step]

            # mask is a logical_and implemented by multiplication
            output_spikes[
                (torch.isinf(output_spikes) *
                 mem > neuron_params['threshold'])] = step * training_params['resolution']
            # reset voltage after spike for plotting
            # mem[torch.logical_not(torch.isinf(output_spikes))] = 0.
            if plotting:
                all_mem.append(mem.cpu().numpy())

        if plotting:
            import matplotlib.pyplot as plt
            import warnings
            tmp_output_spikes = output_spikes.cpu()

            spiketimes_eb = NeuronFunctionEventbased.apply(
                input_spikes, input_weights,
                neuron_params, training_params, device
            ).cpu().detach()
            diff = spiketimes_eb - tmp_output_spikes
            diff[torch.isinf(spiketimes_eb)] = 0
            deviating = diff.abs() > 15 * training_params['resolution']
            problematic_samples = torch.arange(len(input_spikes))[deviating.sum(axis=1) > 0]

            if n_batch >= 9:
                fig, axes = plt.subplots(3, 3, figsize=(16, 10))
            else:
                fig, ax = plt.subplots(1, 1)
                axes = np.array([ax])
            all_mem = np.array(all_mem)
            np.save(f"membrane_trace_{n_presyn}.npy", all_mem)
            np.save(f"membrane_spike_{n_presyn}.npy", tmp_output_spikes)
            batch_to_plot = 0
            for i, ax in enumerate(axes.flatten()):
                batch_to_plot = problematic_samples[i] if i < len(problematic_samples) else i
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    ax.plot(np.arange(training_params['steps']) * training_params['resolution'],
                            all_mem[:, batch_to_plot, :])

                    ax.axhline(neuron_params['leak'], color='black', lw=0.4)
                    ax.axhline(neuron_params['threshold'], color='black', lw=1)
                    for i, sp in enumerate(tmp_output_spikes[batch_to_plot]):
                        ax.axvline(sp, color=f"C{i}", ls="-.", ymax=0.5, ymin=0.2)
                        ax.axvline(spiketimes_eb[batch_to_plot][i], color=f"C{i}", ls=":", ymax=0.2)
                    if output_times is not None:
                        for i, ti in enumerate(output_times[batch_to_plot]):
                            ax.axvline(ti, color=f"C{i}", ls=":", ymin=0.5)

                    ax.set_ylim(
                        (neuron_params['threshold'] - neuron_params['leak']) * np.array((-1, 1.1)) + neuron_params['leak'])

                    ax.set_ylabel(f"sample {batch_to_plot}\n"
                                  f"{np.round(diff[batch_to_plot].numpy() / training_params['resolution']).astype(int)}*"
                                  f"{training_params['resolution']}", fontweight='bold')
                    ax.yaxis.label.set_color(f"C{batch_to_plot}")
            fig.tight_layout()
            fig.savefig(f'debug_int{n_postsyn}.png')
            plt.close(fig)

        if torch.isnan(output_spikes).sum() > 0:
            raise ArithmeticError("There are NaNs in the output times, this means a serious error occured")

        ctx.neuron_params = neuron_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, output_spikes)

        return output_spikes


class NeuronFunctionHicannx(NeuronFunctionEventbased):
    @staticmethod
    def forward(ctx,
                input_spikes, input_weights,
                neuron_params, training_params, device, output_times):
        """output spikes are determined by the hardware"""
        ctx.neuron_params = neuron_params
        ctx.device = device
        ctx.save_for_backward(
            input_spikes, input_weights, output_times)

        return output_times.clone()


class NeuronLayer(torch.nn.Module):
    def __init__(self, layer_def, default_neuron_params, training_params, device):
        """Setup up a layer of neurons"""
        super(NeuronLayer, self).__init__()
        self.input_features = layer_def['feat_in'][0]
        self.output_features = layer_def['size']
        self.training_params = training_params
        self.device = device

        # Add the default neuron params to the layer specific ones if missing
        self.neuron_params = copy.copy(default_neuron_params)
        self.neuron_params.update(layer_def.get('neuron_params', {}))

        # Get and set the trial-to-trial (t2t) and the fixed pattern (fp) noise on the threshold
        self.neuron_params.setdefault('threshold_noise_s2s_std', 0.0)
        self.neuron_params.setdefault('threshold_noise_b2b_std', 0.0)
        self.neuron_params.setdefault('threshold_noise_fp_std', 0.0)
        self.neuron_params.setdefault('threshold_noise_fp_mean', 0.0)
        self.neuron_params['threshold_noise_fp'] = self.neuron_params['threshold_noise_fp_mean'] + torch.randn(
            self.output_features,
            generator=utils_spiketime.gen_hwAware,
            device=self.device,
        ) * self.neuron_params['threshold_noise_fp_std']

        # Sanity check of the neuron parameters values
        tau_syn, tau_mem, g_leak = self.neuron_params['tau_syn'], self.neuron_params['tau_mem'], self.neuron_params['g_leak']

        self.substrate = training_params.get('substrate', 'sim')

        if self.substrate == 'sim':
            assert tau_syn / tau_mem == g_leak, \
                f"did you set g_leak according to tau ratio (probably {tau_syn / tau_mem}, " \
                f"currently {g_leak})"
        if 'model_tau_ratio' not in self.neuron_params:
            # Only infer the model_tau_ratio if the ratio is exactly 1 or 2
            # otherwise we could be in a noisy simulation
            if tau_mem / tau_syn in {1, 2}:
                self.neuron_params['model_tau_ratio'] = tau_mem / tau_syn
            else:
                raise IOError("Need to specify model_tau_ratio")

        self.lr = layer_def.get('learning_rate', None)
        self.use_forward_integrator = training_params.get('use_forward_integrator', False)
        if self.use_forward_integrator:
            assert 'resolution' in training_params and 'sim_time' in training_params
            self.training_params['steps'] = int(np.ceil(training_params['sim_time'] / training_params['resolution']))
            self.neuron_params['decay_syn'] = float(np.exp(-training_params['resolution'] / self.neuron_params['tau_syn']))
            self.neuron_params['decay_mem'] = float(np.exp(-training_params['resolution'] / self.neuron_params['tau_mem']))

        self.weights = to_device(torch.Tensor(self.input_features,
                                              self.output_features),
                                 device)
        self.trainable = layer_def.get('trainable', True)
        if self.trainable:
            self.weights = torch.nn.Parameter(self.weights)
        self.monitor = layer_def.get('monitor', False)
        assert not self.monitor or self.trainable, "Only trainable layers can be monitored"
        assert self.trainable or layer_def['max_num_missing_spikes'] == 1, \
            "If weights are untrainable, max_num_missing_spikes has to be set 1 to prevent bumping"

        self.weights.data.normal_(layer_def['weights_mean'],
                                  layer_def['weights_std'])

    def forward(self, input_times, output_times=None):
        # depending on configuration use either eventbased, integrator or the hardware
        if self.substrate == 'sim':
            assert output_times is None
            if self.use_forward_integrator:
                return NeuronFunctionIntegrator.apply(input_times, self.weights,
                                                      self.neuron_params, self.training_params,
                                                      self.device)
            else:
                return NeuronFunctionEventbased.apply(input_times, self.weights,
                                                      self.neuron_params, self.training_params,
                                                      self.device)
        else:
            return NeuronFunctionHicannx.apply(input_times, self.weights,
                                               self.neuron_params, self.training_params,
                                               self.device,
                                               output_times)


class BroadcastLayer(torch.nn.Module):
    def __init__(self, n_presyn, n_postsyn):
        super(BroadcastLayer, self).__init__()
        self._n_postsyn = n_postsyn
        self.trainable = False
        self.monitor = False

        self.lr = None

    def forward(self, input_spikes):
        """
        Input dimension : B x pre_synaptic
        Output dimension : B x pre_synaptic x post_synaptic
        """
        return input_spikes.unsqueeze(2).repeat([1, 1, self._n_postsyn])


class BiasesLayer(torch.nn.Module):
    def __init__(self, layer_def, device):
        super(BiasesLayer, self).__init__()
        self._times = to_device(torch.tensor(layer_def['times']), device)
        self.monitor = False
        self.lr = None

    def forward(self, input_spikes):
        if len(input_spikes.shape) == 2:
            input_times_including_bias = torch.cat(
                (input_spikes,
                 self._times.unsqueeze(0).repeat([len(input_spikes), 1])),
                1)
            return input_times_including_bias
        else:
            input_times_including_bias = torch.cat(
                (input_spikes,
                 self._times.unsqueeze(0).unsqueeze(2).repeat(
                     [input_spikes.shape[0], 1, input_spikes.shape[2]])),
                1)
            return input_times_including_bias


class DelayLayer(torch.nn.Module):
    def __init__(self, layer_def, n_presyn, n_postsyn, device):
        super(DelayLayer, self).__init__()

        self.lr = layer_def.get('learning_rate', None)
        if 'positivity' not in layer_def:
            raise NotImplementedError("You have to specify a way to make the delays positive")
        positivity_name = layer_def['positivity']['name']
        if positivity_name == 'sigmoid':
            self._fn_on_delay = self.scaled_shift_sigmoid
            self.sigmoid_shift = layer_def['positivity'].get('shift', 0.0)
            self.sigmoid_scale = layer_def['positivity'].get('scale', 1.0)
            self._fn_on_rectify = None
        elif positivity_name == 'clip':
            self._fn_on_delay = None
            self._fn_on_rectify = clip_negatives
        elif positivity_name == 'shift':
            self._fn_on_delay = None
            self._fn_on_rectify = shift_to_positive
        else:
            raise NotImplementedError(f"delay positivity {layer_def['positivity']} not implemented")

        # determine the number of trainable delay parameters after into account delay sharing
        self.presyn_sharing_factor = layer_def.get('presyn_sharing_factor', 1)
        if n_presyn % self.presyn_sharing_factor != 0:
            raise ValueError("The number of presynaptic neurons must be divisble by the presynaptic sharing factor")
        n_trainable_presyn = n_presyn // self.presyn_sharing_factor

        self.postsyn_sharing_factor = layer_def.get('postsyn_sharing_factor', 1)
        if n_postsyn % self.postsyn_sharing_factor != 0:
            raise ValueError("The number of postsynaptic neurons must be divisble by the postsynaptic sharing factor")
        n_trainable_postsyn = n_postsyn // self.postsyn_sharing_factor

        if layer_def['name'] == 'DelaySynaptic':
            delays_shape = (n_trainable_presyn, n_trainable_postsyn)
        elif layer_def['name'] == 'DelayAxonal':
            delays_shape = (n_trainable_presyn, 1)
        elif layer_def['name'] == 'DelayDendritic':
            delays_shape = (1, n_trainable_postsyn)
        else:
            raise NotImplementedError(f"DelayLayer {layer_def['name']} is not implemented yet")

        self._delay_parameters = to_device(torch.normal(
            size=delays_shape, mean=layer_def['delay_mean'], std=layer_def['delay_std']
        ), device)
        self.trainable = layer_def.get("trainable", True)
        if self.trainable:
            self._delay_parameters = torch.nn.Parameter(self._delay_parameters)

        self.monitor = layer_def.get('monitor', False)
        assert not self.monitor or self.trainable, "Only trainable layers can be monitored"

        self._n_postsyn = n_postsyn
        self._output_features = n_presyn

        self.jitter = layer_def.get('jitter', 0.)

    def effective_delays(self):
        # sigmoid is not the final solution,
        # better would be regularisation on delays or encuraging early spikes
        if self._fn_on_delay is None:
            effective_delays = self._delay_parameters
        else:
            effective_delays = self._fn_on_delay(self._delay_parameters)

        # repeat delays according to the pre- and postsynaptic sharing factors
        effective_delays = effective_delays.repeat(self.presyn_sharing_factor, 1)
        effective_delays = effective_delays.repeat(1, self.postsyn_sharing_factor)

        return effective_delays

    def delays_rectify(self):
        if self._fn_on_rectify is not None:
            with torch.no_grad():
                self._delay_parameters.data = self._fn_on_rectify(self._delay_parameters)

    def forward(self, input_spikes, hardware_times=None):
        """output_times only used on HicannX when we have analog delays"""
        if len(input_spikes.shape) == 2:
            input_spikes = input_spikes.unsqueeze(2).repeat(
                    [1, 1, self._n_postsyn])
        output_spikes = input_spikes + self.effective_delays().unsqueeze(0)

        if self.jitter != 0:
            # add noise
            output_spikes += torch.randn(
                output_spikes.size(), device=output_spikes.device, generator=utils_spiketime.gen_hwAware
            ) * self.jitter

        if hardware_times is None:
            return output_spikes
        else:
            retval = output_spikes - (
                output_spikes - hardware_times.unsqueeze(2).repeat([1, 1, self._n_postsyn])
            ).detach()  # return hardware_times but with gradient of output_spikes

            # get rid of nans
            retval[torch.isinf(hardware_times.unsqueeze(2).repeat([1, 1, self._n_postsyn]))] = torch.inf

            return retval

    def scaled_shift_sigmoid(self, delays):
        return self.sigmoid_shift + self.sigmoid_scale * torch.sigmoid(delays)


class MultiplexLayer(torch.nn.Module):
    def __init__(self, layer_def):
        super(MultiplexLayer, self).__init__()
        self.trainable = False
        self.monitor = False
        self.factor = layer_def['factor']
        self.lr = None

    def forward(self, input_spikes):
        return input_spikes.repeat_interleave(self.factor, dim=1)


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


def network_load(path, basename, device, init_hx=True):
    net = to_device(torch.load(osp.join(path, basename + "_network.pt"),
                               map_location=device, weights_only=False, ),
                    device)
    if net.substrate == 'hx':
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
            net.hx_lastsetweights = [torch.full(l.weights.data.shape, -64)
                                     for l in net.layers if isinstance(l, NeuronLayer)]
            if init_hx:
                net.init_hicannx(device)
        else:
            print(f"############# untrained network loaded -> not initialising, also got error:\n{error}")
    elif net.substrate == 'hx_pynn':
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
            net.hx_lastsetweights = [torch.full(l.weights.data.shape, -64)
                                     for l in net.layers if isinstance(l, NeuronLayer)]
            if init_hx:
                net.init_hicannx(device)
        else:
            print(f"############# untrained network loaded -> not initialising, also got error:\n{error}")

    net.device = get_default_device()
    return net


class RegulariserEarlyOutput(torch.nn.Module):
    def __init__(self, reg_params, tau_syn):
        super().__init__()
        self.alpha = reg_params['alpha']
        self.beta = reg_params['beta']
        self.tau_syn = tau_syn
        return

    def forward(self, true_label_times, **kwargs):
        return self.alpha * torch.exp(true_label_times.flatten() / (self.beta * self.tau_syn))


class RegulariserL2Delay(torch.nn.Module):
    def __init__(self, reg_params, tau_syn):
        super().__init__()
        self.alpha = reg_params['alpha']
        self.tau_syn = tau_syn
        self.use_effective_delays = reg_params['use_effective_delays']
        return

    def forward(self, net, **kwargs):
        list_of_delays = [
            l.effective_delays() if self.use_effective_delays else l._delay_parameters
            for l in net.layers if isinstance(l, DelayLayer)
        ]
        delays = torch.cat([d.flatten() for d in list_of_delays])
        return self.alpha * torch.mean(delays**2)


class LossFunctionTTFS(torch.nn.Module):
    def __init__(self, number_labels, tau_syn, params, device):
        super().__init__()
        self.number_labels = number_labels
        self.tau_syn = tau_syn
        self.xi = params['xi']
        self.device = device
        return

    def forward(self, label_times, true_label_times, **kwargs):
        loss = torch.log(torch.exp(-1 * label_times / (self.xi * self.tau_syn)).sum(1)) \
            + true_label_times.flatten() / (self.xi * self.tau_syn)
        return loss

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
    def __init__(self, number_labels, tau_syn, params, device):
        super().__init__()
        self.number_labels = number_labels
        self.tau_syn = tau_syn
        self.device = device

        self.t_correct = self.tau_syn * params['t_correct']
        self.t_wrong = self.tau_syn * params['t_wrong']
        return

    def forward(self, label_times, true_label, **kwargs):
        target = torch.eye(self.number_labels, device=self.device)[true_label] * (self.t_correct - self.t_wrong) + self.t_wrong
        loss = 1. / 2. * ((label_times - target)**2).sum(1)
        return loss

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


class LossFunctionShiftInvariantMSE(torch.nn.Module):
    def __init__(self, number_labels, tau_syn, params, device):
        super().__init__()
        self.number_labels = number_labels
        self.tau_syn = tau_syn
        self.device = device

        self.delta_t = self.tau_syn * params['delta_t']
        return

    def forward(self, label_times, true_label, true_label_times):
        label_time_differences = label_times - true_label_times
        target = self.delta_t - torch.eye(self.number_labels, device=self.device)[true_label] * self.delta_t
        loss = 1. / 2. * ((label_time_differences - target)**2).sum(1)
        return loss

    def select_classes(self, outputs):
        if self.delta_t > 0:
            # the correct label neuron is trained to fire first
            closest_to_target = torch.abs(outputs).argmin(1)
        else:
            # the correct label neuron is trained to fire last
            closest_to_target = torch.abs(outputs).argmax(1)
        ctt_reshaped = closest_to_target.view(-1, 1)
        # count how many firsts had inf or nan as value
        nan_mask = torch.isnan(torch.gather(outputs, 1, ctt_reshaped)).flatten()
        inf_mask = torch.isinf(torch.gather(outputs, 1, ctt_reshaped)).flatten()
        # set firsts to -1 so that they cannot be counted as correct
        closest_to_target[nan_mask] = -1
        closest_to_target[inf_mask] = -1
        return closest_to_target


class GetLoss(torch.nn.Module):
    def __init__(self, training_params, number_labels, tau_syn, device):
        "Dynamically get the loss function depending on the params"
        super().__init__()
        self.device = device
        params = training_params['loss']
        if params['type'] == 'TTFS':
            self.loss = LossFunctionTTFS(number_labels, tau_syn, params, device)
        elif params['type'] == 'MSE':
            self.loss = LossFunctionMSE(number_labels, tau_syn, params, device)
        elif params['type'] == 'shift_invariant_MSE':
            self.loss = LossFunctionShiftInvariantMSE(number_labels, tau_syn, params, device)
        else:
            raise NotImplementedError(f"loss of type '{params['type']}' not implemented")

        reg_params_list = params.get('regulariser', [])
        self.reg_list = []
        for reg_params in reg_params_list:
            assert 'alpha' in reg_params, 'regulariser needs alpha, to scale the regularisation'
            reg_name = reg_params['type']

            if reg_name == 'early_output':
                reg = RegulariserEarlyOutput(reg_params, tau_syn)
            elif reg_name == 'l2_delay':
                reg = RegulariserL2Delay(reg_params, tau_syn)
            else:
                raise NotImplementedError(f"regulariser {reg_name} not implemented")

            self.reg_list.append(reg)

    def forward(self, label_times, true_label, net):
        label_idx = to_device(true_label.clone().type(torch.long).view(-1, 1), self.device)
        true_label_times = label_times.gather(1, label_idx)

        total_loss = self.loss(label_times = label_times,
                               true_label = true_label,
                               true_label_times = true_label_times)

        for reg in self.reg_list:
            total_loss += reg(
                label_times = label_times,
                true_label = true_label,
                true_label_times = true_label_times,
                net = net)
        total_loss[total_loss == np.inf] = 100.
        total_loss[true_label_times.flatten() == np.inf] = 100.
        return total_loss.mean()

    def select_classes(self, outputs):
        return self.loss.select_classes(outputs)


def get_default_device():
    # Pick GPU if avialable, else CPU
    if torch.cuda.is_available():
        print("Using GPU, Yay!")
        # if there are multiple GPUs, either use 'cuda:1,2' or
        # environment variable `export CUDA_VISIBLE_DEVICES=1,3`
        # see https://stackoverflow.com/a/64825728
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


def sigmoid_inverse(x):
    return -torch.log(-1 + 1 / x)


def clip_negatives(x):
    return torch.clip(x, min=0)


def shift_to_positive(x):
    return x - min(0, x.min())


def make_violin_color(violin, colour, zorder=None):
    for key in ['cbars','cmins','cmaxes','cmeans','cmedians']:
        if key in violin:
            violin[key].set_facecolor(colour)
            violin[key].set_edgecolor(colour)
            if zorder is not None:
                violin[key].set_zorder(zorder)
    for part in violin['bodies']: 
        part.set_facecolor(colour)
        part.set_edgecolor(colour)
        if zorder is not None:
            part.set_zorder(zorder)
