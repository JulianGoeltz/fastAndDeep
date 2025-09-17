#!python3
"""Calculating both the spiketimes and their derivatives"""

"""
There are safeguards here in case the model does not produce an output spike.
Passes:
* For the fwd pass:
    because we check all potential causal_sets, we will have a lot of sets
    where the input is not sufficient for an output, i.e. no output spike, i.e.
    we set the output spike time to infinity.
* For the bwd pass:
    in case the fwd pass differs from the exact model because we use hardware,
    or integrator, or jitter spike times, or something.
    In these cases, it can be that a spike is recorded where, according to the
    ideal model, there should not be a spike. In this case the membrane voltage
    is below the threshold at all times, so we don't have a meaningful gradient,
    and we set all gradients to zero.

Models:
* tau_ratio = 1:
    We check whether the lambertw returns a valid number. This is not the case
    if its input is below -1/e, mathematically, we get complex values with nonzero
    imaginary part. This is equivalent to input not sufficient to elicit a spike.
* tau_ratio = 2:
    The discriminant ($x$ in the paper) is negative. Taking the srqt
    of a negative discriminante will create a nan.

Actually, there is another version to this. This might have unexpected consequences
    because it could conceal further problems:
    * put all eps = 0
    * in fwd at the end
        ret_val[torch.isnan(ret_val)] = float('inf')
    * in bwd at the end
        dw[torch.isnan(dw)] = 0
        dt[torch.isnan(dt)] = 0
"""
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


# create a torch generator just for the hw aware stochasticity
gen_hwAware = torch.Generator(device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
gen_hwAware.manual_seed(torch.randint(0, 0xffff_ffff_ffff_fff, (1, )).item())


torch.set_default_dtype(torch.float64)

# try to import the custom lambertw_cuda module
imported_module_worked = True
try:
    from lambertw_cuda import lambertw0 as lambertw_cuda
except ImportError:
    imported_module_worked = False

if not torch.cuda.is_available() or not imported_module_worked:
    from scipy.special import lambertw as lambertw_scipy

    def lambertw(inpt, device):
        # return reals, and set those with nonvanishing imaginary part to Nan
        factorW = lambertw_scipy(inpt.cpu().detach().numpy())
        factorW[np.imag(factorW) != 0] = np.nan
        factorW = utils.to_device(torch.tensor(np.real(factorW)), device)
        return factorW
else:
    def lambertw(inpt, device):
        ret_val = lambertw_cuda(inpt.contiguous())
        # cuda lambertw can't return inf and returns 697.789 instead
        ret_val[ret_val > 690.] = float('inf')
        return ret_val

    # test it
    try:
        test_cuda_tensor = torch.ones(1).to(torch.device('cuda'))
        lambertw(test_cuda_tensor, torch.device('cuda'))
    except Exception:
        print("when trying to evalutate the cuda lambertw there was a problem")
        print("If you have a GPU and installed the module, it is still not executable, recheck your installation"
              "please go into ./pytorch_cuda_lambertw and "
              "run 'python setup.py install --user'. ")
        raise


def get_spiketime(input_spikes, input_weights, neuron_params, device):
    """Calculating spike times, all at once.

    Called from NeuronFunction below, for each layer.
    Dimensions are crucial:
        input weights have dimension BATCHESxNxM with N pre and M postsynaptic neurons.
    """
    n_batch, n_presyn, n_postsyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn2 = input_weights.shape
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2
    assert n_postsyn == n_postsyn2

    # split up weights for each causal set length
    weights_split = input_weights[:, :, None, :]
    weights_split = weights_split.repeat(1, 1, n_presyn, 1)
    tmp_mask = torch.tril_indices(n_presyn, n_presyn, offset=-1)  # want diagonal thus offset
    weights_split[:, tmp_mask[0], tmp_mask[1], :] = 0.

    model_tau_ratio = neuron_params['model_tau_ratio']
    tau_syn = neuron_params['tau_syn']
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)

    # to prevent NaNs when first (sorted) weight(s) is 0, thus A and B, and ratio NaN add epsilon
    eps = 1e-6
    factor_a1 = torch.einsum('ijkl,ijl->ikl',
                             weights_split, exponentiated_spike_times_syn) + eps

    # add random noise across postsynaptic neurons and batches
    threshold = torch.full((n_batch, 1, n_postsyn), neuron_params['threshold'], device=device)

    threshold_noise_s2s_std = neuron_params['threshold_noise_s2s_std']
    threshold_noise_b2b_std = neuron_params['threshold_noise_b2b_std']

    # add noise
    threshold_noise_fp = neuron_params['threshold_noise_fp'].view(1, 1, n_postsyn).expand(n_batch, 1, n_postsyn)
    threshold += threshold_noise_fp
    if threshold_noise_s2s_std != 0 or threshold_noise_b2b_std != 0:
        threshold_noise_s2s = torch.randn(
            (n_batch, 1, n_postsyn), generator=gen_hwAware, device=device
        ) * threshold_noise_s2s_std
        threshold_noise_b2b = torch.randn(
            (1, 1, n_postsyn), generator=gen_hwAware, device=device
        ).expand(
            n_batch, 1, n_postsyn
        ) * threshold_noise_b2b_std

        threshold += threshold_noise_s2s + threshold_noise_b2b

    factor_c = (threshold - neuron_params['leak']) * neuron_params['g_leak']

    if model_tau_ratio == 1:
        factor_b = torch.einsum('ijkl,ijl->ikl',
                                weights_split, input_spikes * exponentiated_spike_times_syn) / tau_syn + eps
        zForLambertW = -factor_c / factor_a1 * torch.exp(factor_b / factor_a1)
        factor_W = lambertw(zForLambertW, device)
        ret_val = tau_syn * (factor_b / factor_a1 - factor_W)

        # check the comment on safeguards above
        ret_val[torch.isnan(factor_W)] = float('inf')

    elif model_tau_ratio == 2:
        tau_mem = neuron_params['tau_mem']
        exponentiated_spike_times_mem = torch.exp(input_spikes / tau_mem)
        factor_a2 = torch.einsum('ijkl,ijl->ikl',
                                 weights_split, exponentiated_spike_times_mem) + eps
        factor_sqrt = torch.sqrt(factor_a2 ** 2 - 4 * factor_a1 * factor_c)
        factor_forLog = 2. * factor_a1 / (factor_a2 + factor_sqrt)
        ret_val = 2. * tau_syn * torch.log(factor_forLog)

        # check the comment on safeguards above
        ret_val[torch.isnan(factor_sqrt)] = float('inf')
        ret_val[factor_forLog <= 0] = float('inf')
    else:
        raise NotImplementedError("tau_ratio not implemented")

    return ret_val


def get_spiketime_derivative(input_spikes, input_weights, neuron_params, device,
                             output_spikes):
    """Calculating the derivatives, see above.

    Weights have shape batch,presyn,postsyn, are ordered according to spike times
    """
    n_batch, n_presyn, n_postsyn = input_spikes.shape
    n_batch2, n_presyn2, n_postsyn2 = input_weights.shape
    assert n_batch == n_batch2, "Deep problem with unequal batch sizes"
    assert n_presyn == n_presyn2
    assert n_postsyn == n_postsyn2

    output_minus_input = -input_spikes + output_spikes.view(n_batch, 1, n_postsyn)
    mask = (output_minus_input < 0) | torch.isinf(output_minus_input) | torch.isnan(output_minus_input)
    causal_weights = input_weights
    # set infinities to 0 preventing nans
    causal_weights[mask] = 0.
    input_spikes[torch.isinf(input_spikes)] = 0.
    output_spikes[torch.isinf(output_spikes)] = 0.

    model_tau_ratio = neuron_params['model_tau_ratio']
    tau_syn = neuron_params['tau_syn']
    exponentiated_spike_times_syn = torch.exp(input_spikes / tau_syn)

    # to prevent NaNs when first (sorted) weight(s) is 0, thus A and B, and ratio NaN add epsilon
    eps = 1e-6
    factor_a1 = torch.einsum(
        'ijl,ijl->il',
        causal_weights, exponentiated_spike_times_syn
    ).unsqueeze(1) + eps
    factor_c = (neuron_params['threshold'] - neuron_params['leak']) * neuron_params['g_leak']
    if model_tau_ratio == 1:
        factor_b = torch.einsum(
            'ijl,ijl->il',
            causal_weights, input_spikes * exponentiated_spike_times_syn
        ).unsqueeze(1) / tau_syn + eps
        zForLambertW = -factor_c / factor_a1 * torch.exp(factor_b / factor_a1)

        factor_W = lambertw(zForLambertW, device)

        dw = -1. / factor_a1 / (1. + factor_W) * exponentiated_spike_times_syn * output_minus_input

        dt = -1. / factor_a1 / (1. + factor_W) * causal_weights * exponentiated_spike_times_syn * \
                (output_minus_input - tau_syn) / tau_syn

        # check the comment on safeguards above
        dw[torch.isnan(factor_W).repeat([1, n_presyn, 1])] = 0.
        dt[torch.isnan(factor_W).repeat([1, n_presyn, 1])] = 0.
    elif model_tau_ratio == 2:
        tau_mem = neuron_params['tau_mem']
        exponentiated_spike_times_mem = torch.exp(input_spikes / tau_mem)
        exponentiated_spike_times_out_mem = torch.exp(output_spikes / tau_mem).unsqueeze(1)
        factor_a2 = torch.einsum(
            'ijl,ijl->il',
            causal_weights, exponentiated_spike_times_mem
        ).unsqueeze(1) + eps
        factor_sqrt = torch.sqrt(factor_a2 ** 2 - 4 * factor_a1 * factor_c)

        dw = tau_mem * (((1. + factor_c / factor_sqrt * exponentiated_spike_times_out_mem)
                         / factor_a1 * exponentiated_spike_times_syn)
                        - 1. / factor_sqrt * exponentiated_spike_times_mem)
        dt = causal_weights * (((1. + factor_c / factor_sqrt * exponentiated_spike_times_out_mem)
                                / factor_a1 * 2. * exponentiated_spike_times_syn)
                               - 1. / factor_sqrt * exponentiated_spike_times_mem)

        # check the comment on safeguards above
        dw[torch.isnan(factor_sqrt).repeat([1, n_presyn, 1])] = 0.
        dt[torch.isnan(factor_sqrt).repeat([1, n_presyn, 1])] = 0.
    else:
        raise NotImplementedError("tau_ratio not implemented")

    # manually set the uncausal and inf output spike entries 0
    dw[mask] = 0.
    dt[mask] = 0.

    return dw, dt
