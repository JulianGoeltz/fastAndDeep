#!/bin/python3
import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import os, os.path as osp
import sys
import torch

import datasets
import networks
import training
import utils


device = utils.get_default_device()


if True:
    # config_path = "../experiment_configs/yin_yang_hxpynn.yaml"
    path_network = "/wang/users/jgoeltz/cluster_home/Documents/utils/fastAndDeep_development_PyNN" \
        "/experiment_results/yin_yang_hxpynn_2024-10-08_08-32-06/epoch_150"
    config_path = path_network + "/config.yaml"
    dataset_params, default_neuron_params, network_layout, training_params = training.load_config(config_path)

    # # massaging dataset
    # multiply_input_layer = 1 if not training_params['substrate'].startswith('hx') else 5
    # dataset_params['multiply_input_layer'] = multiply_input_layer
    dataset_val = datasets.YinYangDataset({**dataset_params, 'size':1000, 'seed':41})
    loader_val = utils.DeviceDataLoader(torch.utils.data.DataLoader(
        dataset_val, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)
    for inputs, labels in loader_val:
        break

    # massaging settings
    training_params["sim_time"] = 10.0
    # network_layout['layers'][0]['name'] = "BroadcastLayer"
    # network_layout['layers'][2]['name'] = "BroadcastLayer"
    # network_layout['n_inputs'] = network_layout['n_inputs'] * multiply_input_layer

    training_params_integrator = copy.deepcopy(training_params)
    training_params_integrator.update(dict(substrate= 'sim', use_forward_integrator=True, ))
    training_params_eb = copy.deepcopy(training_params)
    training_params_eb.update(dict(substrate= 'sim', ))
    default_neuron_params_hw = copy.deepcopy(default_neuron_params)

    # getting special hw params
    assert os.environ.get('SLURM_HARDWARE_LICENSES') is not None
    hx_settings = training.get_hx_settings()
    default_neuron_params_hw.update(hx_settings['neuron_params'])
    print(f"Set up some network based on {config_path}, spiced up with hw "
          f"params for {os.environ.get('SLURM_HARDWARE_LICENSES')}:")
    print(default_neuron_params)
    print(network_layout)
    print(training_params)

    net_tmp = utils.network_load(
        path_network,
        "yin_yang", device, init_hx=False)
    tmp_layer_data = [[], [], [], []]
    if network_layout['layers'][0]['name'] != "BroadcastLayer":
        tmp_layer_data[0] = net_tmp.layers[0]._delay_parameters
    tmp_layer_data[1] = net_tmp.layers[1].weights.data
    if network_layout['layers'][2]['name'] != "BroadcastLayer":
        tmp_layer_data[2] = net_tmp.layers[2]._delay_parameters
    tmp_layer_data[3] = net_tmp.layers[3].weights.data
    # one dummy run, otherwise pynn will create problems
    del(net_tmp)

    torch.manual_seed(42069)
    pprint(default_neuron_params_hw)
    pprint(network_layout)
    pprint(training_params)
    net_hxpynn = networks.get_network(default_neuron_params_hw, network_layout,
        training_params, device)
    net_int = networks.get_network(default_neuron_params_hw, network_layout,
                                   training_params_integrator, device)
    net_eb = networks.get_network(default_neuron_params_hw, network_layout,
                                   training_params_eb, device)
    use_trained_weights = True
    if use_trained_weights:
        for i in [0, 2]:
            if network_layout['layers'][i]['name'] != "BroadcastLayer":
                net_hxpynn.layers[i]._delay_parameters.data = tmp_layer_data[i]
                net_int.layers[i]._delay_parameters.data = tmp_layer_data[i]
                net_eb.layers[i]._delay_parameters.data = tmp_layer_data[i]
        for i in [1, 3]:
            net_hxpynn.layers[i].weights.data = tmp_layer_data[i]
            net_int.layers[i].weights.data = tmp_layer_data[i]
            net_eb.layers[i].weights.data = tmp_layer_data[i]
    else:
        for i in [0, 2]:
            if network_layout['layers'][i]['name'] != "BroadcastLayer":
                net_hxpynn.layers[i]._delay_parameters.data = torch.full_like(tmp_layer_data[i], 3.0)
                net_int.layers[i]._delay_parameters.data = torch.full_like(tmp_layer_data[i], 3.0)
                net_eb.layers[i]._delay_parameters.data = torch.full_like(tmp_layer_data[i], 3.0)
        for i in [1, 3]:
            net_hxpynn.layers[i].weights.data = torch.full_like(tmp_layer_data[i], 1.0)
            net_int.layers[i].weights.data = torch.full_like(tmp_layer_data[i], 1.0)
            net_eb.layers[i].weights.data = torch.full_like(tmp_layer_data[i], 1.0)
    net_hxpynn.write_weights()
    print("Networks set up")

    outputs_int = net_int(inputs)
    print("Done int pass")
    outputs_eb = net_eb(inputs)
    print("Done eb pass")

    trace_hid = np.load("membrane_trace_20.npy")
    trace_lab = np.load("membrane_trace_30.npy")


def record_and_plot(
    record_idx_hidden=0,
    record_idx_label=0,
    plotted_patterns=[0, 1, 2, 3],
    n_repetitions=5,
):
    if record_idx_hidden is not None:
        # #################### hidden
        net_hxpynn.hx_record_neuron = net_hxpynn.network.pynn_pop_hid[record_idx_hidden:record_idx_hidden+1]
        fig, axes = plt.subplots(2, len(plotted_patterns), sharex=True)
        axes[0, 0].set_title(f"hidden neuron {record_idx_hidden}")
        for i_pat, plotted_pattern in enumerate(plotted_patterns):
            # inputs
            for t in inputs.detach().cpu().numpy()[plotted_pattern][:4]:
                axes[0, i_pat].axvline(t, ymin=0.0, ymax=0.1, c='black')
                axes[1, i_pat].axvline(t, ymin=0.0, ymax=0.1, c='black')

            # formatting
            if i_pat==0:
                axes[0, i_pat].set_ylabel("hx-pynn")
                axes[1, i_pat].set_ylabel("integ./ev.-b.")
            else:
                axes[0, i_pat].sharey(axes[0, 0])
                axes[1, i_pat].sharey(axes[1, 0])
                plt.setp(axes[0, i_pat].get_yticklabels(), visible=False)
                plt.setp(axes[1, i_pat].get_yticklabels(), visible=False)
            axes[0, i_pat].set_xlim(0.0, 3.5)
            axes[-1, i_pat].set_xlabel("time [taus]")

            # simulations
            axes[1, i_pat].plot(
                np.arange(0, training_params['sim_time'], training_params['resolution']),
                trace_hid[:, plotted_pattern, record_idx_hidden,]
            )
            axes[1, i_pat].axhline(default_neuron_params_hw['threshold'], color="black", ls=":", alpha=0.4)
            spike = outputs_int[1][1].detach().cpu().numpy()[plotted_pattern, record_idx_hidden]
            axes[1, i_pat].axvline(spike, ymin=0.5)
            spike = outputs_int[1][1].detach().cpu().numpy()[plotted_pattern, record_idx_hidden]
            axes[1, i_pat].axvline(spike, ymax=0.5, color="C1", ls="--")

        # hardware plots
        differences = np.zeros((n_repetitions, *outputs_eb[1][1].shape))
        all_spiketimes_hw = np.zeros((n_repetitions, *outputs_eb[1][1].shape))
        all_outputs_hw = []
        for i_rep in range(n_repetitions):
            outputs_hw = net_hxpynn(inputs)
            all_outputs_hw.append(outputs_hw)
            print(f"Done hw rep {i_rep}/{n_repetitions} for hidden {record_idx_hidden}")
            for i_pat, plotted_pattern in enumerate(plotted_patterns):
                axes[0, i_pat].plot(
                    np.array((net_hxpynn.trace.times / net_hxpynn.hx_settings['scale_times'])) - plotted_pattern * net_hxpynn.hx_settings['single_simtime'],
                    net_hxpynn.trace,
                    color=f"C{i_rep}", alpha=0.3,
                )
                spike = outputs_hw[1][1].detach().cpu().numpy()[plotted_pattern, record_idx_hidden]
                axes[0, i_pat].axvline(spike, color=f"C{i_rep}", alpha=0.3,)
                differences[i_rep] = (
                    outputs_eb[1][1].detach().cpu().numpy() -
                    outputs_hw[1][1].detach().cpu().numpy()
                )
                all_spiketimes_hw[i_rep] = outputs_hw[1][1].detach().cpu().numpy()

        # ############################## statistics of spike times
        np.save('dbg_all_spiketimes_hw.npy', all_spiketimes_hw)
        all_spiketimes_hw = np.load('dbg_all_spiketimes_hw.npy')
        # all_stds = np.std(all_spiketimes_hw[:, :, :], axis=0)
        # print(all_stds)
        list_neurons = np.arange(10)
        for i_neuron in list_neurons:
            fig, axes = plt.subplots(1, 1,
                                     # figsize=(16, 10)
                                     )
            for i_pattern in np.arange(5):
                axes.hist(
                    all_spiketimes_hw[:, i_pattern, i_neuron],
                    bins=np.arange(0.5, 4.5, 0.01),
                    color=f"C{i_pattern}", alpha=0.5, label=f"pattern {i_pattern}")
            axes.set_title(f"nrn {i_neuron}")
                # list_neurons = [np.nanargmax(all_stds[i_pattern, :]) ]
                # for i_neuron in list_neurons:
                #     axes[1].hist(all_spiketimes_hw[:, i_pattern, i_neuron],
                #                  bins=np.arange(3.0, 7.0, 0.01),
                #                  color=f"C{i_neuron}", alpha=0.5, label=f"pattern {i_pattern} nrn {i_neuron}")
            axes.set_xlabel('time [tau]')
            # axes[0].legend()
            axes.legend()
            fig.savefig(f"dbg_spiketime_distr_nrn{i_neuron}_{os.environ.get('SLURM_HARDWARE_LICENSES')}.png")
            plt.close(fig)
        sys.exit()
        # ############################## statistics on threshold for calculations
        if not True:
            n_iter = 100
            n_batch, n_neurons = outputs_eb[1][1].shape
            vals_threshold = np.linspace(2.0, 3.25, n_iter)
            all_differences = np.zeros((n_iter, n_repetitions, *outputs_eb[1][1].shape))

            for i_thresh, thresh in enumerate(vals_threshold):
                with torch.no_grad():
                    net_eb.layers[1].neuron_params['threshold'] = thresh
                    new_outputs_eb = net_eb(inputs)
                    for i_rep in range(n_repetitions):
                        all_differences[i_thresh, i_rep] = np.abs(
                            new_outputs_eb[1][1].detach().numpy() -
                            all_outputs_hw[i_rep][1][1].detach().numpy()
                        )
            print(f"Done with {n_iter} iterations, plotting and printing result")
            replace_inf_with = np.nan
            replace_nan_with = 100
            print(f"Replacing {np.isinf(all_differences).sum() / all_differences.size * 100:.0f}% infs with {replace_inf_with}")
            print(f"Replacing {np.isnan(all_differences).sum() / all_differences.size * 100:.0f}% nan with {replace_nan_with}")
            all_differences[np.isinf(all_differences)] = replace_inf_with
            all_differences[np.isnan(all_differences)] = replace_nan_with

            idcs_argmin = np.argmin(all_differences, axis=0)
            optimal_thresh = vals_threshold[idcs_argmin]
            print(f"Found {(idcs_argmin==0).sum()} idcs_argmin==0, those values replaced with nan")
            optimal_thresh[idcs_argmin==0] = np.nan

            optimal_thresh_perneuron_mean = np.nanmean(optimal_thresh, axis=(0, 1))
            optimal_thresh_perneuron_std = np.nanstd(optimal_thresh, axis=(0, 1))
            print(optimal_thresh_perneuron_mean)
            print(optimal_thresh_perneuron_std)

            print(f"perneuron_mean statistics: {np.nanmean(optimal_thresh_perneuron_mean):.2f}"
                  f"±{np.nanstd(optimal_thresh_perneuron_mean):.2f}")
            print(f"perneuron_std statistics: {np.nanmean(optimal_thresh_perneuron_std):.2f}"
                  f"±{np.nanstd(optimal_thresh_perneuron_std):.2f}")

            fig, axes = plt.subplots(4, 1, figsize=(16, 10))
            fig.suptitle(
                f"perneuron_mean statistics: {np.nanmean(optimal_thresh_perneuron_mean):.2f}±{np.nanstd(optimal_thresh_perneuron_mean):.2f}"
                "\n"
                f"perneuron_std statistics: {np.nanmean(optimal_thresh_perneuron_std):.2f}±{np.nanstd(optimal_thresh_perneuron_std):.2f}"
            )
            percentiles = np.percentile(all_differences, [0, 25, 50, 75, 100], axis=(1, 2))
            for i_n in range(n_neurons):
                axes[0].plot(vals_threshold, percentiles[2, :, i_n], color=f"C{i_n%10}")
                axes[0].fill_between(vals_threshold,
                                     percentiles[1, :, i_n],
                                     percentiles[3, :, i_n],
                                     alpha=0.3, color=f"C{i_n%10}")
                axes[0].axvline(optimal_thresh_perneuron_mean[i_n], alpha=0.1, color=f"C{i_n%10}")
            axes[0].set_ylabel("difference\nmedian±IQR [tau]")
            axes[0].set_xlabel("threshold values")
            axes[0].set_yscale('log')
            axes[0].set_ylim(1e-3, 2e0)

            for i_n in range(n_neurons):
                tmp_val = optimal_thresh[:, :, i_n].flatten()
                axes[1].scatter(np.full_like(tmp_val, i_n) + np.linspace(-0.3, 0.3, len(tmp_val)),
                                tmp_val,
                                color=f"C{i_n%10}", marker="+")
                violin = axes[1].violinplot(tmp_val, [i_n], widths=1.0)
                utils.make_violin_color(violin, 'black', zorder=1)  # f"C{i_n%10}")
            axes[1].set_ylabel("optimal thresh vals\neverything")

            for i_n in range(n_neurons):
                tmp_val = np.nanmean(optimal_thresh[:, :, i_n], axis=0)
                axes[2].scatter(np.full_like(tmp_val, i_n) + np.linspace(-0.3, 0.3, len(tmp_val)),
                                tmp_val,
                                color=f"C{i_n%10}", marker="+")
                violin = axes[2].violinplot(tmp_val, [i_n], widths=1.0)
                utils.make_violin_color(violin, 'black', zorder=1)  # f"C{i_n%10}")
            axes[2].set_ylabel(f"optimal thresh vals\navg per {optimal_thresh.shape[0]} reps")

            for i_n in range(n_neurons):
                tmp_val = np.nanmean(optimal_thresh[:, :, i_n], axis=1)
                axes[3].scatter(np.full_like(tmp_val, i_n) + np.linspace(-0.3, 0.3, len(tmp_val)),
                                tmp_val,
                                color=f"C{i_n%10}", marker="+")
            axes[3].set_ylabel(f"optimal thresh vals\navg per {optimal_thresh.shape[1]} patterns")

            axes[-1].set_xlabel("neuron idcs")

            fig.tight_layout()
            fig.savefig(f"dbg_tryThresholds_{os.environ.get('SLURM_HARDWARE_LICENSES')}.png")

            sys.exit()
        # ##############################

        tmp = copy.deepcopy(differences)
        n_inf_p = (differences == np.inf).sum()
        n_inf_m = (differences == -np.inf).sum()
        n_nonnan = np.logical_not(np.isnan(differences)).sum()

        differences[np.isinf(differences)] = np.nan
        axes[0, -2].set_title(f"+∞:{n_inf_p};"
                              f"-∞:{n_inf_m};"
                              f"nonnan:{n_nonnan};"
                              f"mean±std:{np.nanmean(differences):.3f}±{np.nanstd(differences):.3f}",
                              fontsize=10,
                              )

        with np.printoptions(precision=3, suppress=True):
            print(differences.shape)
            diff_per_neuron_means = np.nanmean(differences, axis=(0, 1))
            diff_per_neuron_stds = np.nanstd(differences, axis=(0, 1))
            print(f"diff per neurons, means: {np.nanmean(diff_per_neuron_means):.3f}"
                  f"±{np.nanstd(diff_per_neuron_means):.3f}")
            print(diff_per_neuron_means)
            print(f"diff per neurons, stds: {np.nanmean(diff_per_neuron_stds):.3f}"
                  f"±{np.nanstd(diff_per_neuron_stds):.3f}")
            print(diff_per_neuron_stds)

            print(f"Percentiles {np.nanpercentile(differences, [0, 10, 25, 50, 75, 90, 100])}")

        fig.tight_layout(h_pad=0, w_pad=0)
        fig.savefig(f"test_hardwareParameters_hid{record_idx_hidden}"
                    f"_{os.environ.get('SLURM_HARDWARE_LICENSES')}.png", dpi=300)
        plt.close(fig)
        print("saved hid fig")

        fig, ax = plt.subplots(1, 1)
        for i_n in range(tmp.shape[-1]):
            tmptmp = tmp[:, :, i_n]
            tmptmp = tmptmp[np.logical_not(np.isnan(tmptmp))]
            tmptmp = tmptmp[np.logical_not(np.isinf(tmptmp))]
            if len(tmptmp) > 0:
                violin = ax.violinplot(tmptmp, [i_n], widths=1.0)
                utils.make_violin_color(violin, "C1")
        tmptmp = tmp[np.logical_not(np.isnan(tmp))]
        tmptmp = tmptmp[np.logical_not(np.isinf(tmptmp))]
        violin = ax.violinplot(tmptmp, [-1], widths=1.0)
        utils.make_violin_color(violin, "C0")
        ax.set_xticks([-1] + list(range(tmp.shape[-1])))
        ax.set_xticklabels(["all"] + list(range(tmp.shape[-1])))
        ax.set_xlabel("neurons")
        ax.set_ylabel("missmatch spike times (ideal - measured) [taus]")
        fig.tight_layout()
        fig.savefig('dbg_distr_spiketimes_hidden.png')
        plt.close(fig)

    if record_idx_label is not None:
        # #################### label
        fig, axes = plt.subplots(2, len(plotted_patterns), sharex=True)
        axes[0, 0].set_title(f"label neuron {record_idx_label}")
        net_hxpynn.hx_record_neuron = net_hxpynn.network.pynn_pop_out[record_idx_label:record_idx_label+1]
        for i_pat, plotted_pattern in enumerate(plotted_patterns):
            # inputs
            for t in outputs_eb[1][1].detach().numpy()[plotted_pattern]:
                axes[1, i_pat].axvline(t, ymin=0.0, ymax=0.1, c='black')

            # formatting
            if i_pat==0:
                axes[0, i_pat].set_ylabel("hx-pynn")
                axes[1, i_pat].set_ylabel("integ./ev.-b.")
            else:
                axes[0, i_pat].sharey(axes[0, 0])
                axes[1, i_pat].sharey(axes[1, 0])
                plt.setp(axes[0, i_pat].get_yticklabels(), visible=False)
                plt.setp(axes[1, i_pat].get_yticklabels(), visible=False)
            axes[0, i_pat].set_xlim(6.5, 9.5)
            axes[-1, i_pat].set_xlabel("time [taus]")

            # simulations
            axes[1, i_pat].plot(
                np.arange(0, training_params['sim_time'], training_params['resolution']),
                trace_lab[:, plotted_pattern, record_idx_label,]
            )
            axes[1, i_pat].axhline(default_neuron_params_hw['threshold'], color="black", ls=":", alpha=0.4)
            spike = outputs_int[0].detach().numpy()[plotted_pattern, record_idx_label]
            axes[1, i_pat].axvline(spike, ymin=0.5)
            spike = outputs_int[0].detach().numpy()[plotted_pattern, record_idx_label]
            axes[1, i_pat].axvline(spike, ymax=0.5, color="C1", ls="--")

        # hardware plots
        differences = np.zeros((n_repetitions, *outputs_eb[0].shape))
        for i_rep in range(n_repetitions):
            outputs_hw = net_hxpynn(inputs)
            print(f"Done hw rep {i_rep}/{n_repetitions} for label {record_idx_label}")
            for i_pat, plotted_pattern in enumerate(plotted_patterns):
                for t in outputs_hw[1][1].detach().numpy()[plotted_pattern]:
                    axes[0, i_pat].axvline(t, 
                                           ymin=np.linspace(0, 0.1, n_repetitions + 1)[i_rep],
                                           ymax=np.linspace(0, 0.1, n_repetitions + 1)[i_rep + 1],
                                           c=f'C{i_rep}', alpha=0.1)
                axes[0, i_pat].plot(
                    np.array((net_hxpynn.trace.times / net_hxpynn.hx_settings['scale_times'])) - plotted_pattern * net_hxpynn.hx_settings['single_simtime'],
                    net_hxpynn.trace,
                    color=f"C{i_rep}", alpha=0.3,
                )
                spike = outputs_hw[0].detach().numpy()[plotted_pattern, record_idx_label]
                axes[0, i_pat].axvline(spike, color=f"C{i_rep}", alpha=0.3,)
                differences[i_rep] = (
                    outputs_eb[0].detach().numpy() -
                    outputs_hw[0].detach().numpy()
                )
        tmp = copy.deepcopy(differences)
        n_inf_p = (differences == np.inf).sum()
        n_inf_m = (differences == -np.inf).sum()
        n_nonnan = np.logical_not(np.isnan(differences)).sum()

        differences[np.isinf(differences)] = np.nan
        axes[0, -2].set_title(f"+∞:{n_inf_p};"
                              f"-∞:{n_inf_m};"
                              f"nonnan:{n_nonnan};"
                              f"mean±std:{np.nanmean(differences):.3f}±{np.nanstd(differences):.3f}",
                              fontsize=10,
                              )

        with np.printoptions(precision=3, suppress=True):
            print(differences.shape)
            diff_per_neuron_means = np.nanmean(differences, axis=(0, 1))
            diff_per_neuron_stds = np.nanstd(differences, axis=(0, 1))
            print(f"diff per neurons, means: {np.nanmean(diff_per_neuron_means):.3f}"
                  f"±{np.nanstd(diff_per_neuron_means):.3f}")
            print(diff_per_neuron_means)
            print(f"diff per neurons, stds: {np.nanmean(diff_per_neuron_stds):.3f}"
                  f"±{np.nanstd(diff_per_neuron_stds):.3f}")
            print(diff_per_neuron_stds)

            print(f"Percentiles {np.nanpercentile(differences, [0, 10, 25, 50, 75, 90, 100])}")

        fig.tight_layout(h_pad=0, w_pad=0)
        fig.savefig(f"test_hardwareParameters_lab{record_idx_label}"
                    f"_{os.environ.get('SLURM_HARDWARE_LICENSES')}.png", dpi=300)
        plt.close(fig)
        print("saved label fig")

        fig, ax = plt.subplots(1, 1)
        for i_n in range(tmp.shape[-1]):
            tmptmp = tmp[:, :, i_n]
            tmptmp = tmptmp[np.logical_not(np.isnan(tmptmp))]
            tmptmp = tmptmp[np.logical_not(np.isinf(tmptmp))]
            if len(tmptmp) > 0:
                violin = ax.violinplot(tmptmp, [i_n], widths=1.0)
                utils.make_violin_color(violin, "C1")
        tmptmp = tmp[np.logical_not(np.isnan(tmp))]
        tmptmp = tmptmp[np.logical_not(np.isinf(tmptmp))]
        violin = ax.violinplot(tmptmp, [-1], widths=1.0)
        utils.make_violin_color(violin, "C0")
        ax.set_xticks([-1] + list(range(tmp.shape[-1])))
        ax.set_xticklabels(["all"] + list(range(tmp.shape[-1])))
        ax.set_xlabel("neurons")
        ax.set_ylabel("missmatch spike times (ideal - measured) [taus]")
        fig.tight_layout()
        fig.savefig('dbg_distr_spiketimes_label.png')
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing current hardware parametrisation")

    parser.add_argument('--hidden-neurons', nargs='*', type=int, default=[])
    parser.add_argument('--label-neurons', nargs='*', type=int, default=[])

    parser.add_argument('--plotted-patterns', nargs='*', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--n-repetitions', type=int, default=5)
    args = parser.parse_args()

    print(f"Going to run for hiddens {args.hidden_neurons} and labels {args.label_neurons}, "
          f"so {len(args.hidden_neurons)+len(args.label_neurons)}x{args.n_repetitions} times")

    for hidden_neuron in args.hidden_neurons:
        record_and_plot(
            record_idx_hidden=hidden_neuron,
            record_idx_label=None,
            plotted_patterns=args.plotted_patterns,
            n_repetitions=args.n_repetitions,
        )
    for label_neuron in args.label_neurons:
        record_and_plot(
            record_idx_hidden=None,
            record_idx_label=label_neuron,
            plotted_patterns=args.plotted_patterns,
            n_repetitions=args.n_repetitions,
        )
