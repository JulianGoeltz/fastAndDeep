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
    dataset_params, default_neuron_params, network_layout, training_params = training.load_config(config_path)
    dataset_name = dataset_params['name']
    if dataset_name == "yin_yang":
        # TODO: the following five lines should be replaced by a multiplex layer in the network def
        multiply_input_layer = 1 if not training_params['substrate'].startswith('hx') else 5
        multiply_bias = 1 if not training_params['substrate'].startswith('hx') else 5
        dataset_params['multiply_input_layer'] = multiply_input_layer
        dataset_train = datasets.YinYangDataset({**dataset_params, 'size':5000, 'seed':42})
        dataset_val = datasets.YinYangDataset({**dataset_params, 'size':1000, 'seed':41})
        dataset_test = datasets.YinYangDataset({**dataset_params, 'size':1000, 'seed':40})
        # dataset_train = datasets.YinYangDataset(size=5000, seed=42)
        # dataset_val = datasets.YinYangDataset(size=1000, seed=41)
        # dataset_test = datasets.YinYangDataset(size=1000, seed=40)
    elif dataset_name == "bars":
        dataset_train = datasets.BarsDataset(3, noise_level=0)
        dataset_val = datasets.BarsDataset(3, noise_level=0)
        dataset_test = datasets.BarsDataset(3, noise_level=0)
    elif dataset_name == "xor":
        dataset_train = datasets.XOR()
        dataset_val = datasets.XOR()
        dataset_test = datasets.XOR()
    elif dataset_name == "mnist":
        dataset_train = datasets.FullMnist('train')
        dataset_val = datasets.FullMnist('val')
        dataset_test = datasets.FullMnist('test')
    elif dataset_name == "16x16_mnist":
        width_pixel = network_layout.get('width_pixel', 16)
        if width_pixel != 16:
            print("*********************************************")
            print(f"using mnist reduced to {width_pixel}x{width_pixel}, not usual 16x16")
            print("*********************************************")
            network_layout['n_inputs'] = width_pixel**2
        dataset_train = datasets.HicannMnist('train', late_at_inf=True, width_pixel=width_pixel)
        dataset_val = datasets.HicannMnist('val', late_at_inf=True, width_pixel=width_pixel)
        dataset_test = datasets.HicannMnist('test', late_at_inf=True, width_pixel=width_pixel)
    elif dataset_name == "single_in_single_out":
        dataset_train = datasets.Single_in_Single_out(dataset_params)
        dataset_val = datasets.Single_in_Single_out(dataset_params)
        dataset_test = datasets.Single_in_Single_out(dataset_params)
    elif dataset_name == "pattern":
        dataset_params['n_inputs'] = network_layout['n_inputs']
        dataset_train = datasets.Pattern(dataset_params)
        dataset_val = datasets.Pattern(dataset_params)
        dataset_test = datasets.Pattern(dataset_params)
    else:
        sys.exit("data set given in parameter file is unknown")

    if mode == 'train':
        if training_params['substrate'].startswith('hx'):
            assert os.environ.get('SLURM_HARDWARE_LICENSES') is not None
            hx_settings = training.get_hx_settings()
            print("Using hardware settings:")
            pprint(hx_settings)
            default_neuron_params.update(hx_settings['neuron_params'])
            pprint(default_neuron_params)
            # modify network layout with multiplication of input for YY on hw
            if dataset_name == "yin_yang":
                if network_layout['n_inputs'] == 4:
                    network_layout['n_inputs'] = network_layout['n_inputs'] * multiply_input_layer
                    if network_layout['layers'][0]['name'] == 'Biases':
                        network_layout['layers'][0]['times'] *= multiply_input_layer
        elif training_params['substrate'] == 'sim':
            if os.environ.get('SLURM_HARDWARE_LICENSES') is not None:
                sys.exit("There are SLURM_HARDWARE_LICENSES available "
                         f"({os.environ.get('SLURM_HARDWARE_LICENSES')}), but 'substrate' is 'sim'. \n"
                         "Either execute without hw resources, or set 'substrate'")

    # sanity checks of network configuration
    assert network_layout['layers'][-1].get('name', None) == 'NeuronLayer', "Last layer has to be a NeuronLayer"
    assert len(dataset_val.vals[0]) == network_layout['n_inputs'], \
        f"number of inputs found in dataset ({len(dataset_val.vals[0])}) " \
        f"must match number of input neurons ({network_layout['n_inputs']})"
    assert max(dataset_val.cs) + 1 == network_layout['layers'][-1].get('size', -1), \
        f"number of classes found in dataset must match number of output neurons"

    filename = dataset_name

    # main code
    if mode == 'train':
        config_name = osp.splitext(osp.basename(sys.argv[2]))[0]
        t_start = time.perf_counter()

        if len(sys.argv) == 3 :
            basename = '../experiment_results/'
            dirname = basename + '{0}_{1:%Y-%m-%d_%H-%M-%S}'.format(config_name, datetime.datetime.now())
        elif len(sys.argv) == 4 :
            dirname = sys.argv[3]
        else :
            raise IOError("Train only accept 1 or 2 arguments")
        print(f'Results will be stored in {dirname}')

        net = training.train(training_params, dataset_params, network_layout, default_neuron_params,
                             dataset_train, dataset_val, dataset_test, dirname, dataset_name)
        t_end = time.perf_counter()
        duration = t_end - t_start
        print('Training {0} epochs -> duration: {1} seconds'.format(training_params['epoch_number'], duration))

        dirname = dirname + f"/epoch_{training_params['epoch_number']}/"
        device = utils.get_default_device()
        untrained = False
    elif mode in ['eval', 'inference', 'inference_repeated']:
        dirname = sys.argv[2]
        device = utils.get_default_device()
        untrained = False
        if mode == "eval":
            net = None
        else:
            start_epoch = int(sys.argv[3])
            net = training.continue_training(dirname, dataset_name, start_epoch, [start_epoch],
                                             dataset_train, dataset_val, dataset_test)
        if len(sys.argv) > 3 and sys.argv[3] == 'eval_untrained':
            untrained = True
    elif mode in ['continue', 'fast_eval']:
        dirname = sys.argv[2]
        start_epoch = int(sys.argv[3])
        savepoints = sys.argv[4].split(',')
        savepoints = [int(item) for item in savepoints]
        t_start = time.time()
        net = training.continue_training(dirname, dataset_name, start_epoch, savepoints,
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
            dirname, dataset_name, 'test', dataset_test, untrained=False, reference=False,
            return_hidden=True,  # to run new inference
            device=device, net=net, wholeset=False)
        correct = torch.eq(torch.tensor(labels), selected_classes.detach().cpu()).sum().numpy()
        acc = correct / len(selected_classes)
        print(f"After inference, the accuracy is {acc * 100:.2f}%.")
    elif mode == 'eval' or (net.substrate.startswith('hx') and mode not in ['fast_eval', 'inference_repeated']):
        evaluation.monitored_plot(title=dataset_name, dirname=dirname, filename=filename, net=net)
        evaluation.summary_plot(title=dataset_name, dirname=dirname, filename=filename, net=net)
        evaluation.loss_accuracy(title=dataset_name, dirname=dirname, filename=filename)
        # generic plots:
        evaluation.confusion_matrix('test', dataset_test, dirname=dirname, filename=filename, device=device, net=net)
        evaluation.confusion_matrix('train', dataset_train, dirname=dirname, filename=filename,
                                    device=device, net=net)
        evaluation.sorted_outputs('test', dataset_test, dirname=dirname, filename=filename, device=device, net=net)
        evaluation.sorted_outputs('train', dataset_train, dirname=dirname, filename=filename, device=device, net=net)

        # special plots
        if dataset_name == "yin_yang":
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
        # evaluation.compare_voltages(dataset_params=dataset_train, dirname=dirname, filename=filename, device=device)

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
    if mode == 'inference_repeated' or (net is not None and net.substrate.startswith('hx') and mode != 'fast_eval'):
        print("starting repeated inference")
        accs = []
        for _ in range(10):
            outputs, selected_classes, labels, _, _ = evaluation.run_inference(
                dirname, dataset_name, 'test', dataset_test, untrained=False, reference=False,
                return_hidden=True,  # to run new inference
                device=device, net=net, wholeset=False)
            correct = torch.eq(torch.tensor(labels), selected_classes.detach().cpu()).sum().numpy()
            acc = correct / len(selected_classes)
            accs.append(acc)
            print(f"After inference, the accuracy is {acc * 100:.2f}%.")
        accs = np.array(accs) * 100.
        print(accs)
        print(np.mean(accs), np.std(accs))
