#!/bin/python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import os.path as osp
import sys
import torch

import training
import utils


def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + f'^{{{suffix}}}'


def rasterplot(dirname, plotted_sample):
    _, neuron_params, network_layout, training_params = training.load_config(osp.join(dirname, "config.yaml"))

    # inputs_u = np.load(osp.join(dirname, f"spiketimes_inputs_untrained.npy"))
    # inputs_t = np.load(osp.join(dirname, f"spiketimes_inputs_trained.npy"))
    hiddens_u = np.load(osp.join(dirname, f"spiketimes_hiddens_untrained.npy"))
    hiddens_t = np.load(osp.join(dirname, f"spiketimes_hiddens_trained.npy"))
    outputs_u = np.load(osp.join(dirname, f"spiketimes_outputs_untrained.npy"))
    outputs_t = np.load(osp.join(dirname, f"spiketimes_outputs_trained.npy"))

    # plotted_samples = 5
    fig, axes = plt.subplots(len(hiddens_t) + 1, 2,
                             # sharey=True,
                             sharex=True,
                             figsize=(3.2, 3.7))

    n_bins = 30
    scaling = 10

    # print(f"median, spread between 25, 75 percentile of non-inf spiketimes per sample, averaged over {len(inputs)} samples")
    # for j, dat in enumerate(plottedata):
    #     tmp = dat.clone().numpy()
    #     tmp[np.isinf(tmp)] = np.nan
    #     stds, means = np.nanstd(tmp, axis=1), np.nanmean(tmp, axis=1)
    #     quantiles = np.nanquantile(tmp, q=[0.25, 0.50, 0.75], axis=1)
    #     medians = quantiles[1]
    #     spreads = quantiles[2] - quantiles[0]
    #     print("{}\tavgd. median,IQR {:.3f}, {:.3f}; \tavgd. mean±std {:.3f}±{:.3f}".format(
    #         "input" if j == 0 else ("output" if (j == len(plottedata) - 1) else f"hidden{j}"),
    #         np.mean(medians),
    #         np.mean(spreads),
    #         np.mean(means),
    #         np.mean(stds),
    #     ))
    for i_trai, trai in enumerate(["Untrained", "Trained"]):
        if trai == "Untrained":
            plotted_data = [hiddens_u[i] for i in range(len(hiddens_u))] + [outputs_u, ]
        elif trai == "Trained":
            plotted_data = [hiddens_t[i] for i in range(len(hiddens_t))] + [outputs_t, ]
        else:
            raise NotImplementedError()
        # plotted_data = plotted_data[::-1]
        # offset = 0
        artists = []
        for i_lay, dat in enumerate(plotted_data):
            ax = axes[len(plotted_data) - 1 - i_lay][i_trai]
            islast = i_lay == len(plotted_data) - 1
            tmp = dat[plotted_sample][:1000].reshape((-1, 1))
            # SCALING
            tmp *= scaling
            # tmp[np.isinf(tmp)] = 4.0
            # linelengths = 1 if not islast else 3
            # offsets = np.arange(len(tmp)) * linelengths  # + offset
            # offset += len(tmp) * linelengths
            print(f"  {i_lay} {trai} {len(tmp)}")
            # colour = f"C{i_lay}" if not islast else "C9",
            colour = 'black'
            ax.eventplot(
                tmp,
                color=colour,
                linelengths=1 if islast else 2,
                linewidths=None if islast else 2,
                # lineoffsets=offsets,
                # linelengths=linelengths,
            )

            if not islast:
                ax.hist(
                    tmp[np.logical_not(np.isinf(tmp))],
                    bins=np.linspace(0, 3.5 * scaling, n_bins),
                    histtype='step',
                    color="C1",
                    alpha=0.8,
                )

            # label = "input" if i_lay == 0 else ("output" if islast else f"hidden{i_lay}")
            label = "output" if islast else f"hidden{i_lay}"
            artists.append(mpatches.Patch(
                color=colour,
                label=label,
            ))

            ax.set_ylim(0, len(tmp))
            ax.set_xlim(-0.0, 3.5 * scaling)
            if i_trai == 0:
                ax.set_ylabel(
                    # f"layer\n\n${ordinal(i_lay + 1)}$" if i_lay == 2 else f"${ordinal(i_lay + 1)}$"
                    f"layer number\n${i_lay + 1}$" if i_lay == 2 else f"${i_lay + 1}$"
                    # f"{label} layer"
                )
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if i_lay == 0:
                ax.set_xlabel("time (ms)")
            if islast:
                ax.set_title(f"{trai}")

    fig.tight_layout(
        w_pad=0.50,
        h_pad=1.00,
        rect=(-0.03, -0.03, 1.0, 1.0),
    )
    fn = f'spikeraster'
    for ex in ['.png', ]:  #  '.pdf', '.svg']:
        for d in ['', ]:  # dirname + '/', ]:  # '']:
            print(f"saving plot as {d + fn + ex}")
            fig.savefig(d + fn + ex, dpi=300)

    return


if __name__ == "__main__":
    print('hi')
    dirname = sys.argv[1]
    sample = int(sys.argv[2])
    rasterplot(dirname, sample)
