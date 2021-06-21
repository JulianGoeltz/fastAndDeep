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

import utils

torch.set_default_dtype(torch.float64)

if "USE_LAMBERTW_SCIPY" in os.environ or not torch.cuda.is_available():
    from scipy.special import lambertw as lambertw_scipy

    def lambertw(inpt, device):
        # return reals, and set those with nonvanishing imaginary part to -inf
        factorW = lambertw_scipy(inpt.cpu().detach().numpy())
        factorW[np.imag(factorW) != 0] = -np.inf
        factorW = utils.to_device(torch.tensor(np.real(factorW)), device)
        return factorW
else:
    # try to import the module
    try:
        from lambertw_cuda import lambertw0 as lambertw_cuda

        def lambertw(inpt, device):
            ret_val = lambertw_cuda(inpt)
            # cuda lambertw can't return inf and returns 697.789 instead
            ret_val[ret_val > 690.] = float('inf')
            return ret_val

    except ImportError:
        raise NotImplementedError(
            "If you have a GPU, "
            "please go into ./pytorch_cuda_lambertw and "
            "run 'python setup.py install --user'. "
            "Alternatively define USE_LAMBERTW_SCIPY in your env.")
    # test it
    try:
        test_cuda_tensor = torch.ones(1).to(torch.device('cuda'))
        lambertw(test_cuda_tensor, torch.device('cuda'))
    except Exception:
        print("when trying to evalutate the cuda lambertw there was a problem")
        raise


def get_spiketime(input_spikes, input_weights, neuron_params, device):
    """Calculating spike times, all at once.

    Called from EqualtimeFunction below, for each layer.
    Dimensions are crucial:
        input weights have dimension BATCHESxNxM with N pre and M postsynaptic neurons.
    """
    n_batch, n_presyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn = input_weights.shape
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2

    # split up weights for each causal set length
    weights_split = input_weights[:, :, None, :]
    weights_split = weights_split.repeat(1, 1, n_presyn, 1)
    tmp_mask = torch.tril_indices(n_presyn, n_presyn, offset=-1)  # want diagonal thus offset
    weights_split[:, tmp_mask[0], tmp_mask[1], :] = 0.

    # temporary reshape for torch reasons
    weights_split = weights_split.view(n_batch, n_presyn, n_presyn * n_postsyn)
    # new (empty) dimension needed for torch reasons
    input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    tau_syn = neuron_params['tau_syn']
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)

    # to prevent NaNs when first (sorted) weight(s) is 0, thus A and B, and ratio NaN add epsilon
    eps = 1e-6
    factor_a1 = torch.matmul(exponentiated_spike_times_syn, weights_split)
    factor_b = torch.matmul(input_spikes * exponentiated_spike_times_syn, weights_split) / tau_syn + eps
    factor_c = (neuron_params['threshold'] - neuron_params['leak']) * neuron_params['g_leak']
    zForLambertW = -factor_c / factor_a1 * torch.exp(factor_b / factor_a1)

    factor_W = lambertw(zForLambertW, device)

    ret_val = tau_syn * (factor_b / factor_a1 - factor_W)
    ret_val = ret_val.view(n_batch, n_presyn, n_postsyn)

    return ret_val


def get_spiketime_derivative(input_spikes, input_weights, neuron_params, device,
                             output_spikes):
    """Calculating the derivatives, see above.

    Weights have shape batch,presyn,postsyn, are ordered according to spike times
    """
    n_batch, n_presyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn = input_weights.shape
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2

    output_minus_input = -input_spikes.view(n_batch, n_presyn, 1) + output_spikes.view(n_batch, 1, n_postsyn)
    mask = (output_minus_input < 0) | torch.isinf(output_minus_input) | torch.isnan(output_minus_input)
    causal_weights = input_weights
    # set infinities to 0 preventing nans
    causal_weights[mask] = 0.
    input_spikes[torch.isinf(input_spikes)] = 0.
    output_spikes[torch.isinf(output_spikes)] = 0.

    input_spikes = input_spikes.view(n_batch, 1, n_presyn)

    tau_syn = neuron_params['tau_syn']
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)

    eps = 1e-6
    factor_a1 = torch.matmul(exponentiated_spike_times_syn, causal_weights)
    factor_b = torch.matmul(input_spikes * exponentiated_spike_times_syn, causal_weights) / tau_syn + eps
    factor_c = (neuron_params['threshold'] - neuron_params['leak']) * neuron_params['g_leak']
    zForLambertW = -factor_c / factor_a1 * torch.exp(factor_b / factor_a1)

    factor_W = lambertw(zForLambertW, device)

    exponentiated_spike_times_syn = exponentiated_spike_times_syn.squeeze().unsqueeze(-1)

    dw = -1. / factor_a1 / (1. + factor_W) * exponentiated_spike_times_syn * output_minus_input

    dt = -1. / factor_a1 / (1. + factor_W) * causal_weights * exponentiated_spike_times_syn * \
        (output_minus_input - tau_syn) / tau_syn

    # manually set the uncausal and inf output spike entries 0
    dw[mask] = 0.
    dt[mask] = 0.

    return dw, dt
