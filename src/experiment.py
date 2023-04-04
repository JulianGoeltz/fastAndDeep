#!python3
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from pprint import pprint
import sys
import time
import torch
import yaml

import training
import datasets
import evaluation
import utils


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Script has to be called with mode and parameter file/folder")
    mode = sys.argv[1]

    # set up datasets, configs
    if mode == 'train':
        config_path = sys.argv[2]
    else:
        config_path = osp.join(sys.argv[2], "config.yaml")
    dataset, neuron_params, network_layout, training_params = training.load_config(config_path)

    if dataset == "yin_yang":
        multiply_input_layer = 1 if not training_params['use_hicannx'] else 5
        multiply_bias = 1 if not training_params['use_hicannx'] else 5
        dataset_train = datasets.YinYangDataset(size=5000, seed=42, multiply_input_layer=multiply_input_layer)
        dataset_val = datasets.YinYangDataset(size=1000, seed=41, multiply_input_layer=multiply_input_layer)
        dataset_test = datasets.YinYangDataset(size=1000, seed=40, multiply_input_layer=multiply_input_layer)
    elif dataset == "bars":
        multiply_input_layer = 1 if not training_params['use_hicannx'] else 5
        dataset_train = datasets.BarsDataset(3, noise_level=0, multiply_input_layer=multiply_input_layer)
        dataset_val = datasets.BarsDataset(3, noise_level=0, multiply_input_layer=multiply_input_layer)
        dataset_test = datasets.BarsDataset(3, noise_level=0, multiply_input_layer=multiply_input_layer)
    elif dataset == "xor":
        dataset_train = datasets.XOR()
        dataset_val = datasets.XOR()
        dataset_test = datasets.XOR()
    elif dataset == "mnist":
        dataset_train = datasets.FullMnist('train')
        dataset_val = datasets.FullMnist('val')
        dataset_test = datasets.FullMnist('test')
    elif dataset == "16x16_mnist":
        width_pixel = network_layout.get('width_pixel', 16)
        if width_pixel != 16:
            print("*********************************************")
            print(f"using mnist reduced to {width_pixel}x{width_pixel}, not usual 16x16")
            print("*********************************************")
            network_layout['n_inputs'] = width_pixel**2
        dataset_train = datasets.HicannMnist('train', late_at_inf=True, width_pixel=width_pixel)
        dataset_val = datasets.HicannMnist('val', late_at_inf=True, width_pixel=width_pixel)
        dataset_test = datasets.HicannMnist('test', late_at_inf=True, width_pixel=width_pixel)
    else:
        sys.exit("data set given in parameter file is unknown")
    filename = dataset

    # main code
    if mode == 'train':
        if training_params['use_hicannx']:
            assert os.environ.get('SLURM_HARDWARE_LICENSES') is not None
            with open('py/hx_settings.yaml') as f:
                hx_settings = yaml.load(f, Loader=yaml.SafeLoader)
            hx_setup_no = os.environ.get('SLURM_HARDWARE_LICENSES')
            if hx_setup_no not in hx_settings:
                raise OSError(f"Setup no {hx_setup_no} is not described in hx settings file, only {hx_settings.keys()}")
            print("Using hardware settings:")
            pprint(hx_settings[hx_setup_no])
            neuron_params = hx_settings[hx_setup_no]['neuron_params']

            # modify network layout with multiplication of input for YY on hw
            if dataset == "yin_yang":
                network_layout['n_inputs'] = network_layout['n_inputs'] * multiply_input_layer
                network_layout['n_biases'] = [multiply_bias, 0]
                network_layout['bias_times'] = [
                    np.array(times).repeat(multiply_bias).tolist()  # list for yaml dump
                    for times in network_layout['bias_times']
                ]
            elif dataset == "bars":
                network_layout['n_inputs'] = network_layout['n_inputs'] * multiply_input_layer
        else:
            if os.environ.get('SLURM_HARDWARE_LICENSES') is not None:
                sys.exit("There are SLURM_HARDWARE_LICENSES available "
                         f"({os.environ.get('SLURM_HARDWARE_LICENSES')}), but 'use_hicannx' is False. \n"
                         "Either execute without hw resources, or set 'use_hicannx'")

        config_name = osp.splitext(osp.basename(sys.argv[2]))[0]
        t_start = time.perf_counter()
        dirname = '{0}_{1:%Y-%m-%d_%H-%M-%S}'.format(config_name, datetime.datetime.now())
        net = training.train(training_params, network_layout, neuron_params,
                             dataset_train, dataset_val, dataset_test, dirname, filename)
        t_end = time.perf_counter()
        duration = t_end - t_start
        print('Training {0} epochs -> duration: {1} seconds'.format(training_params['epoch_number'], duration))

        dirname = '../experiment_results/' + dirname + f"/epoch_{training_params['epoch_number']}/"
        device = utils.get_default_device()
        untrained = False
    elif mode in ['eval', 'inference']:
        dirname = sys.argv[2]
        device = utils.get_default_device()
        untrained = False
        if len(sys.argv) > 3 and sys.argv[3] == 'eval_untrained':
            untrained = True
        net = None
    elif mode in ['continue', 'fast_eval']:
        dirname = sys.argv[2]
        start_epoch = int(sys.argv[3])
        savepoints = sys.argv[4].split(',')
        savepoints = [int(item) for item in savepoints]
        t_start = time.time()
        net = training.continue_training(dirname, filename, start_epoch, savepoints,
                                         dataset_train, dataset_val, dataset_test)
        t_end = time.time()
        duration = t_end - t_start
        print('Training {0} epochs -> duration: {1} seconds'.format(
            savepoints[-1] - start_epoch, duration))

        dirname = dirname + f"/epoch_{savepoints[-1]}/"
        device = utils.get_default_device()
        untrained = False
    else:
        raise IOError("argument must be train, eval, continue or inference")

    if mode == 'inference':
        outputs, selected_classes, labels, _, _ = evaluation.run_inference(
            dirname, filename, 'test', dataset_test, untrained=False, reference=False,
            return_hidden=True,  # to run new inference
            device=device, net=net, wholeset=False)
        correct = torch.eq(torch.tensor(labels), selected_classes.detach().cpu()).sum().numpy()
        acc = correct / len(selected_classes)
        print(f"After inference, the accuracy is {acc * 100:.2f}%.")
    elif mode == 'eval' or (net.use_hicannx and mode != 'fast_eval'):
        # generic plots:
        evaluation.confusion_matrix('test', dataset_test, dirname=dirname, filename=filename, device=device, net=net)
        evaluation.confusion_matrix('train', dataset_train, dirname=dirname, filename=filename,
                                    device=device, net=net)
        evaluation.sorted_outputs('test', dataset_test, dirname=dirname, filename=filename, device=device, net=net)
        evaluation.sorted_outputs('train', dataset_train, dirname=dirname, filename=filename, device=device, net=net)

        evaluation.summary_plot(dataset, dirname=dirname, filename=filename, net=net)

        # special plots
        if dataset == "yin_yang":
            evaluation.yin_yang_classification('test', dataset_test, dirname=dirname, filename=filename,
                                               device=device, net=net)
            evaluation.yin_yang_classification('train', dataset_train, dirname=dirname, filename=filename,
                                               device=device, net=net)
            evaluation.yin_yang_spiketimes('train', dataset_train, dirname=dirname, filename=filename,
                                           device=device, net=net)
            evaluation.yin_yang_spiketimes('test', dataset_test, dirname=dirname, filename=filename,
                                           device=device, net=net)
            evaluation.yin_yang_hiddentimes('train', dataset_train, dirname=dirname, filename=filename,
                                            device=device, net=net)
            evaluation.yin_yang_hiddentimes('test', dataset_test, dirname=dirname, filename=filename,
                                            device=device, net=net)
            evaluation.yin_yang_spiketime_diffs('train', dataset_train, dirname=dirname, filename=filename,
                                                device=device, net=net)
            evaluation.yin_yang_spiketime_diffs('test', dataset_test, dirname=dirname, filename=filename,
                                                device=device, net=net)

        # potential plotting (needs a manual switch in utils.py)
        # evaluation.compare_voltages(dataset=dataset_train, dirname=dirname, filename=filename, device=device)

        evaluation.loss_accuracy(dataset, dirname=dirname, filename=filename)
        if untrained:
            evaluation.confusion_matrix('test', dataset_test, dirname=dirname, filename=filename,
                                        untrained=True, device=device, net=net)
            evaluation.sorted_outputs('test', dataset_test, dirname=dirname, filename=filename,
                                      untrained=True, device=device, net=net)
        if net is None:
            evaluation.weight_histograms(dirname=dirname, filename=filename, device=device)
            evaluation.weight_matrix(dirname=dirname, filename=filename, device=device)
            # evaluation.spiketime_hist('train', dataset_train, dirname=dirname, filename=filename, device=device)
            # evaluation.spiketime_hist('test', dataset_test, dirname=dirname, filename=filename, device=device)
