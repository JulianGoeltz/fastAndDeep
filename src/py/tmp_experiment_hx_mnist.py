#!python3
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
import yaml

import utils as u
import training as t
import datasets as d
import evaluation as e


# ########speed up for YY
# lst = []
# for i in range(10):
#     lst.append(classify(dirname=dirname, show=False,
#                         device=device, net=net,
#                         datasets_list=[['test', dataset_test], ]))
# lst = np.array(lst) * 100
# print(f"ACCURACY {lst.mean():.2f}+-{lst.std():.2f}")
#
# def eval(batch_size):
#     neuron_params, training_params, network_layout = e.load_config(dirname, filename)
#     training_params['batch_size_eval'] = batch_size
#     loader = e.setup_data(dataset_test, training_params)
#     all_accs = []
#     for i, data in enumerate(loader):
#         inputs, labels = data
#         with torch.no_grad():
#             outputs, hiddens = net(inputs)
#         firsts = outputs.argmin(1)
#         firsts_reshaped = firsts.view(-1, 1)
#         nan_mask = torch.isnan(torch.gather(outputs, 1, firsts_reshaped)).flatten()
#         inf_mask = torch.isinf(torch.gather(outputs, 1, firsts_reshaped)).flatten()
#         firsts[nan_mask] = -1
#         firsts[inf_mask] = -1
#         wrongs = torch.logical_not(torch.eq(labels.to(device), firsts)).detach().cpu().numpy()
#         acc = (len(outputs) - wrongs.sum()) / len(outputs)
#         all_accs.append(acc)
#     # print(all_accs)
#     print(np.mean(all_accs), np.std(all_accs))
#
# def EVAL(batch_size):
#     for i in range(3):
#         eval(batch_size)


if __name__ == '__main__':
    mode = sys.argv[1]

    width_pixel = 16

    dataset_train = d.HicannMnist('train', late_at_inf=True, width_pixel=width_pixel)
    dataset_val = d.HicannMnist('val', late_at_inf=True, width_pixel=width_pixel)
    dataset_test = d.HicannMnist('test', late_at_inf=True, width_pixel=width_pixel)

    filename = '16x16_mnist'
    detailed_timing = False

    def fasteval_loads(dirname, filename):
        dataset, neuron_params, network_layout, training_params = t.load_config(dirname + "/config.yaml")
        training_params['batch_size_eval'] = 10000
        loader = u.DeviceDataLoader(torch.utils.data.DataLoader(
            dataset_test, batch_size=training_params.get('batch_size_eval', None), shuffle=False), device)
        for i, data in enumerate(loader):
            inputs, labels = data
        return inputs, labels

    def fasteval_doit(inputs, labels, net):
        with torch.no_grad():
            if detailed_timing:
                start = time.time()
            outputs, hiddens = net(inputs)
            if detailed_timing:
                print(f"net took {time.time() - start}")
        firsts = outputs.argmin(1)
        firsts_reshaped = firsts.view(-1, 1)
        nan_mask = torch.isnan(torch.gather(outputs, 1, firsts_reshaped)).flatten()
        inf_mask = torch.isinf(torch.gather(outputs, 1, firsts_reshaped)).flatten()
        firsts[nan_mask] = -1
        firsts[inf_mask] = -1
        wrongs = torch.logical_not(torch.eq(labels.to(device), firsts)).detach().cpu().numpy()
        acc = (len(outputs) - wrongs.sum()) / len(outputs)
        # print(f"### test accuracy {acc}")
        return acc

    def classify(dirname='tmp', show=False, device=None, net=None,
                 datasets_list=[
                     ['test', dataset_test],
                     # ['train', dataset_train],
                 ]):
        """eases re-evaluation with alive net on hx"""
        # neuron_params, training_params, network_layout = load_config(dirname, filename)
        for datatype, dataset in datasets_list:
            outputs, labels, _, inputs = e.run_inference(dirname, filename, dataset, False, False, device,
                                                         return_all=True, net=net)
            firsts = outputs.argmin(1)
            firsts_reshaped = firsts.view(-1, 1)
            nan_mask = torch.isnan(torch.gather(outputs, 1, firsts_reshaped)).flatten()
            inf_mask = torch.isinf(torch.gather(outputs, 1, firsts_reshaped)).flatten()
            firsts[nan_mask] = -1
            firsts[inf_mask] = -1
            wrongs = torch.logical_not(torch.eq(torch.tensor(labels).to(device), firsts)).detach().cpu().numpy()
            acc = (len(outputs) - wrongs.sum()) / len(outputs)
            print(f"{datatype} accuracy {acc}")
        if len(datasets_list) == 1:
            return acc

    if mode == 'train':
        config_path = sys.argv[2]
        neuron_params, training_params, network_layout = e.load_config('', config_path)

        if width_pixel != 16:
            print("*********************************************")
            print(f"using mnist reduced to {width_pixel}x{width_pixel}, not usual 16x16")
            print("*********************************************")
            network_layout['n_inputs'] = width_pixel**2

        # training_params['use_hicannx'] = False

        if os.environ.get('SLURM_HARDWARE_LICENSES') is not None:
            with open('py/hx_settings.yaml') as f:
                hx_settings = yaml.load(f, Loader=yaml.SafeLoader)
            hx_setup_no = os.environ.get('SLURM_HARDWARE_LICENSES')
            if hx_setup_no not in hx_settings:
                raise OSError(f"Setup no {hx_setup_no} is not described in hx settings file")
            neuron_params = hx_settings[hx_setup_no]['neuron_params']
        else:
            if os.environ.get('SLURM_HARDWARE_LICENSES') is not None:
                sys.exit("There are SLURM_HARDWARE_LICENSES available "
                         f"({os.environ.get('SLURM_HARDWARE_LICENSES')}), but 'use_hicannx' is False. \n"
                         "Either execute without hw resources, or set 'use_hicannx'")

        config_name = config_path.split('/')[-1]
        config_name = config_name.split('.')[0]
        dirname = '{0}_{1:%Y-%m-%d_%H-%M-%S}'.format(config_name, datetime.datetime.now())
        t_start = time.time()
        net = t.train(training_params, network_layout, neuron_params, dataset_train, dataset_val, dataset_test,
                      dirname, filename)
        t_end = time.time()
        duration = t_end - t_start
        print('Training {0} epochs -> duration: {1} seconds'.format(training_params['epoch_number'], duration))

        dirname = '../experiment_results/' + dirname + f"/epoch_{training_params['epoch_number']}/"
        device = u.get_default_device()
        untrained = False
    elif mode in ['continue', 'fast_eval']:
        if len(sys.argv) < 5:
            raise IOError("arguments that are needed in order: dirname, epoch to start at, "
                          "savepoints (separated by ','; absolute values, not incremental)")
        dirname = sys.argv[2]
        start_epoch = int(sys.argv[3])
        savepoints = [int(item) for item in sys.argv[4].split(',')]
        t_start = time.time()
        net = t.continue_training(dirname, filename, start_epoch, savepoints, dataset_train, dataset_val, dataset_test)
        t_end = time.time()
        duration = t_end - t_start
        print(f"continuation from startepoch {start_epoch} the savepoints {savepoints} epochs -> "
              f"duration: {duration} seconds")

        dirname = dirname + f"/epoch_{savepoints[-1]}/"
        device = u.get_default_device()
        untrained = False
    elif mode == 'eval':
        dirname = sys.argv[2]
        untrained = False
        if len(sys.argv) > 3 and sys.argv[3] == 'eval_untrained':
            untrained = True
        device = u.get_default_device()
        net = None
    else:
        raise IOError("argument must be train or eval")

    # do eval if eval, or if on HX
    if (mode == 'eval' or net.use_hicannx) and mode != 'fast_eval':
        e.confusion_matrix('test', dataset_test, dirname=dirname, filename=filename,
                           show=False, device=device, net=net)
        if untrained:
            e.confusion_matrix('test', dataset_test, dirname=dirname, filename=filename,
                               show=False, untrained=True, device=device, net=net)
        e.confusion_matrix('train', dataset_train, dirname=dirname, filename=filename,
                           show=False, device=device, net=net)
        e.sorted_outputs('test', dataset_test, dirname=dirname, filename=filename,
                         show=False, device=device, net=net)
        if untrained:
            e.sorted_outputs('test', dataset_test, dirname=dirname, filename=filename,
                             show=False, untrained=True, device=device, net=net)
        e.sorted_outputs('train', dataset_train, dirname=dirname, filename=filename,
                         show=False, device=device, net=net)
        e.loss_accuracy('16x16 Mnist', dirname=dirname, filename=filename, show=False)
        e.summary_plot('16x16 Mnist', dirname=dirname, filename=filename, show=False, net=net)

        if net is None:
            e.weight_histograms(dirname=dirname, filename=filename, show=False, device=device)
            e.weight_matrix(dirname=dirname, filename=filename, device=device)

    inputs, labels = fasteval_loads(dirname, filename)
    net.fast_eval = True
    net.hx_settings['single_simtime'] = 12.  # change this
    acc = fasteval_doit(inputs, labels, net)
    print(f"initial accuracy {acc}")

    net.hx_settings['single_simtime'] = 7.
    # print(f"****think about adjusting initial wait time in backend that is {net.hx_backend._timing_offset}")
    # net.hx_backend._timing_offset = 0.
    print("setting initial wait time to 0")
    acc = fasteval_doit(inputs, labels, net)
    print(f"initial accuracy {acc}")

    # print("one run with detailed timing:")
    # net._record_timings = True
    # acc = fasteval_doit(inputs, labels, net)
    # net._record_timings = False

    # print("reconfiguring for power reduction")
    # net.hx_backend.configure(reduce_power=True)

    print("############STARTING")
    accs, walls = [], []
    for i in range(2):
        start = time.time()
        acc = fasteval_doit(inputs, labels, net)
        end = time.time()
        accs.append(acc)
        walls.append(end - start)
    print(f"accs {np.mean(accs):.3f}+-{np.std(accs):.3f}  median {np.median(accs):.3f}"
          f"walls {np.mean(walls):.3f}+-{np.std(walls):.3f} median {np.median(walls):.3f}")
    print(accs)
    print([f"{wall:.3f}" for wall in walls])

    print("##############below is the procedure after the power reduction ")
    net.hx_settings['single_simtime'] = 8.
    # print(f"****think about adjusting initial wait time in backend that is {net.hx_backend._timing_offset}")
    # net.hx_backend._timing_offset = 0.
    # print("setting initial wait time to 0")
    acc = fasteval_doit(inputs, labels, net)
    print(f"initial accuracy {acc}")

    print("reconfiguring for power reduction")
    net.hx_backend.configure(reduce_power=True)

    print("one run with detailed timing:")
    net._record_timings = True
    net._record_power = True
    acc = fasteval_doit(inputs, labels, net)
    net._record_timings = False
    print(acc)

    net.hx_backend.configure(reduce_power=True)
    # time.sleep(2)
    print("############STARTING")
    accs, walls = [], []
    for i in range(5):
        net.hx_backend.configure(reduce_power=True)
        start = time.time()
        acc = fasteval_doit(inputs, labels, net)
        end = time.time()
        accs.append(acc)
        walls.append(end - start)
    print(f"accs {np.mean(accs):.3f}+-{np.std(accs):.3f}  median {np.median(accs):.3f}"
          f"walls {np.mean(walls):.3f}+-{np.std(walls):.3f} median {np.median(walls):.3f}")
    print(accs)
    print([f"{wall:.3f}" for wall in walls])
