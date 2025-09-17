import itertools
import copy
import yaml
import datetime
import os
import shutil
import multiprocessing as mp
import subprocess
import torch
from functools import reduce
import operator
import utils
import argparse
import time

import concurrent.futures
import subprocess
import time
import os
import multiprocessing

def extract_updates(sweep_config):
    for entry in sweep_config:
        values = entry.get('value', [])
        values = values if isinstance(values, list) else [values]
        path = entry.get('path', None)
        assert path is not None, "Each entry in the sweep configuration must have a 'path' key."
        if len(values) != 0:
            if isinstance(path, str):
                path = [path]
            paths = [[int(key) if key.isdigit() else key for key in p.split('/')] for p in path]
            yield paths, values
        else:
            print(f"WARNING : You left '{path}' empty in the sweep_config.yaml file")


def apply_updates(original_config, updates):
    new_config = copy.deepcopy(original_config)
    for update_paths, update_value in updates:
        for update_path in update_paths:
            set_nested_dict(new_config, update_path, update_value)
    return new_config


def get_nested_dict(root_dict, path_list):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, path_list, root_dict)

def set_nested_dict(root_dict, path_list, value):
    """Set a value in a nested object in root by item sequence."""
    key = path_list[-1]
    if key in get_nested_dict(root_dict, path_list[:-1]):
        get_nested_dict(root_dict, path_list[:-1])[key] = value
    else:
        raise KeyError(f"Key '{key}' does not exist in the dictionary")


def create_configs(original_config, sweep_config):
    """
    Input:
    - original_config: dictionary with the original configuration
    - sweep_config: dictionary with the sweep configuration
    Output:
    - list_of_configs: list of dictionaries with the complete configurations
    - combinations_of_updates: list of lists with the combinations of updates
        Note: an update is a tuple (path, value) where path is list of keys to access the value in the original_config
    """

    list_of_paths, list_of_values = zip(*extract_updates(sweep_config))
    combinations_of_updates = [list(zip(list_of_paths, comb)) for comb in itertools.product(*list_of_values)]
    list_of_configs = [apply_updates(original_config, updates) for updates in combinations_of_updates]
    return list_of_configs, combinations_of_updates

def create_sweep(sweep_results_dirname, initial_config_path, sweep_config_path):
    """"""
    # Load initial and sweep configurations
    with open(initial_config_path, 'r') as file:
        initial_config = yaml.safe_load(file)

    with open(sweep_config_path, 'r') as file:
        sweep_config = yaml.safe_load(file)

    sweep_id = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
    base_dir = os.path.join(sweep_results_dirname, f"sweep_yin_yang_{sweep_id}")
    os.makedirs(base_dir, exist_ok=True)

    # Copy the initial and sweep_config files into the sweep directory
    shutil.copy(initial_config_path, os.path.join(base_dir, 'initial_config.yaml'))
    shutil.copy(sweep_config_path, os.path.join(base_dir, 'sweep_config.yaml'))

    list_of_dirname = []
    list_of_configs, combinations_of_updates = create_configs(initial_config, sweep_config['parameters'])

    for run_config, run_updates in zip(list_of_configs, combinations_of_updates):
        run_updates = [(path, value) for list_of_paths, value in run_updates for path in list_of_paths]
        run_id = "run_" + "_".join(["-".join(map(str, [path[-1]])) + "=" + str(value) for path, value in run_updates])
        run_id = "run_" + "_".join([ str(idx) + "=" + str(value) for idx,(path, value) in enumerate(run_updates)]) # If OS does not accept too long dirnames
        run_id = ''.join(c for c in run_id if c not in [" ", "{", "}", ":", "'", "]", "[", ","])
        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Save the specific configuration for this run
        config_filename = os.path.join(run_dir, 'config.yaml')
        with open(config_filename, 'w') as file:
            yaml.safe_dump(run_config, file)

        update_filename = os.path.join(run_dir, 'sweep_updates.yaml')
        with open(update_filename, 'w') as file:
            yaml.safe_dump([{'path': path, 'value': value} for path, value in run_updates], file)

        list_of_dirname.append(run_dir)

    # sweep_file_structure = {'list_of_dirname': list_of_dirname,
    #                    'list_of_configs': list_of_configs,
    #                    'combinations_of_updates': combinations_of_updates}
    return list_of_dirname

def get_gpu_memory_usage():
    """Get the memory usage of each GPU."""
    try:
        # Use nvidia-smi to get the GPU memory usage
        command = "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits"
        result = subprocess.check_output(command, shell=True).decode('utf-8').strip()
        total, used = zip(*[tuple(map(int, line.split(', '))) for line in result.split('\n')])
        return [100*u/t for u, t in zip(used, total)]
    except subprocess.CalledProcessError:
        print("Error accessing nvidia-smi. Please check your NVIDIA driver installation.")
        return []


def get_gpu_utilization():
    """Get the current GPU utilization percentage for each GPU."""
    try:
        # Use nvidia-smi to get the GPU utilization
        command = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
        result = subprocess.check_output(command, shell=True).decode('utf-8').strip()
        gpu_utilization = [int(line.strip()) for line in result.split('\n')]
        return gpu_utilization
    except subprocess.CalledProcessError:
        print("Error accessing nvidia-smi. Please check your NVIDIA driver installation.")
        return []


def find_available_gpu(available_GPU, mem_threshold=95, util_threshold=95):
    mem_usage = get_gpu_memory_usage()
    gpu_util = get_gpu_utilization()
    for gpu_id in available_GPU:
        if mem_usage[gpu_id] < mem_threshold and gpu_util[gpu_id] < util_threshold:
            return gpu_id
    return None


def launch_run(destination_path, gpu_id=None, verbose=True, use_cpu=False):
    command = f"python3 experiment.py train {os.path.join(destination_path, 'config.yaml')} {destination_path}"
    if use_cpu:
        command = f"python3 experiment.py train {os.path.join(destination_path, 'config.yaml')} {destination_path}"
        if verbose:
            print(f"Launching on CPU {multiprocessing.current_process().name}", flush=True)
    else:
        order = [0, 2, 3, 1] # Specific for EIS's GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = str(order[gpu_id])
        if verbose:
            print(f"Launching on GPU {gpu_id}", flush=True)

    try:
        with open(os.path.join(destination_path, "output.log"), 'w') as f:
            result = subprocess.run(command, shell=True, check=True,
                                    stdout=f,
                                    stderr=subprocess.STDOUT,
                                    text=True)
        return result.stdout if verbose else "Success", None
    except subprocess.CalledProcessError as e:
        return None, e.stderr if verbose else "Error"

def launch_sweep(list_of_dirname, available_CPU=None, available_GPU=None, device=None, verbose=False):
    futures = []

    def handle_result(future):
        stdout, stderr = future.result()
        if stdout:
            print(stdout, flush=True)
        if stderr:
            print(stderr, flush=True)
    if device==None:
        device = utils.get_default_device().type

    if device=='cuda':
        if available_GPU is None:
            available_GPU = list(range(torch.cuda.device_count()))
        def submit_task(dirname):
            while True:
                time.sleep(1)
                gpu_id = find_available_gpu(available_GPU)
                if gpu_id is not None:
                    future = executor.submit(launch_run, dirname, gpu_id, verbose, use_cpu=False)
                    if verbose : future.add_done_callback(handle_result)
                    futures.append(future)
                    return

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dirname in list_of_dirname:
                submit_task(dirname)

    else: # CPU
        if available_CPU is None:
            available_CPU = int(multiprocessing.cpu_count()*0.95)

        def submit_task(dirname):
            future = executor.submit(launch_run, dirname, use_cpu=True, verbose=verbose)
            futures.append(future)

        with concurrent.futures.ProcessPoolExecutor(max_workers=available_CPU) as executor:
            for dirname in list_of_dirname:
                submit_task(dirname)

    concurrent.futures.wait(futures)


def restricted_condition(root, dir, files):
    # return 'scale=1' in root
    return True

def list_runs(sweep_results_dirname, verbose):
    list_of_finished = []
    list_of_unfinished = []
    if verbose: print("Listing the files...")
    walk = list(os.walk(sweep_results_dirname, followlinks=True))
    for idx, (root, dirs, files) in enumerate(walk):
        if verbose and ((idx+1)%100==0 or idx+1==len(walk))  :print(f'\r{idx + 1}/{len(walk)} files', end='', flush=True)
        if root.split(os.sep)[-1].startswith('run_') and restricted_condition(root, dirs, files):
            config_path = os.path.join(root, "config.yaml")
            if not os.path.exists(config_path):
                continue

            if any(dir.startswith('epoch_') for dir in dirs):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)

                epoch_dir = 'epoch_' + str(config['training_params']['epoch_number'])
                if epoch_dir in dirs:
                    epoch_path = os.path.join(root, epoch_dir)
                    if any(file.name.endswith('_test_labels.npy') for file in os.scandir(epoch_path)):
                        list_of_finished.append(root)
                    else:
                        list_of_unfinished.append(root)
                else:
                    list_of_unfinished.append(root)
            else:
                list_of_unfinished.append(root)
    if verbose: print()
    return list_of_finished, list_of_unfinished




def sweep(sweep_results_dirname, initial_config_path, sweep_config_path, available_CPU, available_GPU, verbose):
    list_of_dirname = create_sweep(sweep_results_dirname, initial_config_path, sweep_config_path)
    if verbose:
        print(list_of_dirname)
    launch_sweep(list_of_dirname, available_CPU=available_CPU, available_GPU=available_GPU, device=utils.get_default_device().type, verbose=verbose)

def continue_sweep(sweep_results_dirname, available_CPU, available_GPU, verbose):
    list_of_finished, list_of_unfinished = list_runs(sweep_results_dirname, verbose)
    if verbose:
        print(f'Will continue {len(list_of_unfinished)} runs out of the total {len(list_of_finished) + len(list_of_unfinished)}')
    launch_sweep(list_of_unfinished, available_CPU=available_CPU, available_GPU=available_GPU, device=utils.get_default_device().type, verbose=verbose)

def states_of_sweeps(sweep_results_dirname):
    list_of_finished, list_of_unfinished = list_runs(sweep_results_dirname, verbose = True)
    print(f'Finished runs : {len(list_of_finished)}/{len(list_of_finished) + len(list_of_unfinished)} at {"{0:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep tool")
    subparsers = parser.add_subparsers(dest='command', required=True)

    default_sweep_results_dirname = os.path.join('..', 'sweep_results')


    # Subparser for the 'sweep' command
    parser_sweep = subparsers.add_parser('sweep', help='Start a new sweep')
    default_initial_config_path = os.path.join('..', 'experiment_configs', 'yin_yang_small.yaml')
    default_sweep_config_path = os.path.join('..', 'sweep_config', 'sweep_yin_yang_dev.yaml')
    parser_sweep.add_argument('--sweep_results_dirname', type=str, default=default_sweep_results_dirname,
                              help='Directory for the sweep results')
    parser_sweep.add_argument('--initial_config_path', type=str, default=default_initial_config_path,
                              help='Path to the initial configuration file')
    parser_sweep.add_argument('--sweep_config_path', type=str, default=default_sweep_config_path,
                              help='Path to the sweep configuration file')
    parser_sweep.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser_sweep.add_argument('--available_CPU', type=int, default=None, help='Number of available CPUs')
    parser_sweep.add_argument('--available_GPU', type=int, nargs='+', default=None, help='List of available GPUs')

    # Subparser for the 'continue' command
    parser_continue = subparsers.add_parser('continue', help='Continue an existing sweep')
    parser_continue.add_argument('--sweep_results_dirname', type=str, default=default_sweep_results_dirname,
                                 help='Directory for the sweep results')
    parser_continue.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser_continue.add_argument('--available_CPU', type=int, default=None, help='Number of available CPUs')
    parser_continue.add_argument('--available_GPU', type=int, nargs='+', default=None, help='List of available GPUs')


    # Subparser for the 'print_state' command
    parser_print_state = subparsers.add_parser('print_state', help='Print the state of sweeps')
    parser_print_state.add_argument('--sweep_results_dirname', type=str, default=default_sweep_results_dirname,
                                    help='Directory for the sweep results')

    args = parser.parse_args()

    if args.command == 'sweep':
        sweep(args.sweep_results_dirname, args.initial_config_path, args.sweep_config_path, args.available_CPU, args.available_GPU, args.verbose)
    elif args.command == 'continue':
        continue_sweep(args.sweep_results_dirname, args.available_CPU, args.available_GPU, args.verbose)
    elif args.command == 'print_state':
        states_of_sweeps(args.sweep_results_dirname)



### Lauching the delay_comparison
# For W
# python3 sweep.py sweep --sweep_results_dirname ../delay_comparison/demo_comparison/W --initial_config_path ../delay_comparison/demo_comparison/W/yin_yang.yaml --sweep_config_path ../delay_comparison/demo_comparison/W/sweep_yin_yang.yaml --verbose
# python3 sweep.py continue --sweep_results_dirname ../delay_comparison/demo_comparison/W --verbose

# For WA
# python3 sweep.py sweep --sweep_results_dirname ../delay_comparison/demo_comparison/WA --initial_config_path ../delay_comparison/demo_comparison/WA/yin_yang.yaml --sweep_config_path ../delay_comparison/demo_comparison/WA/sweep_yin_yang.yaml --verbose
# python3 sweep.py continue --sweep_results_dirname ../delay_comparison/demo_comparison/WA --verbose

# For WAD
# python3 sweep.py sweep --sweep_results_dirname ../delay_comparison/demo_comparison/WAD --initial_config_path ../delay_comparison/demo_comparison/WAD/yin_yang.yaml --sweep_config_path ../delay_comparison/demo_comparison/WAD/sweep_yin_yang.yaml --verbose
# python3 sweep.py continue --sweep_results_dirname ../delay_comparison/demo_comparison/WAD --verbose

# For WD
# python3 sweep.py sweep --sweep_results_dirname ../delay_comparison/demo_comparison/WD --initial_config_path ../delay_comparison/demo_comparison/WD/yin_yang.yaml --sweep_config_path ../delay_comparison/demo_comparison/WD/sweep_yin_yang.yaml --verbose
# python3 sweep.py continue --sweep_results_dirname ../delay_comparison/demo_comparison/WD --verbose

# For WS
# python3 sweep.py sweep --sweep_results_dirname ../delay_comparison/demo_comparison/WS --initial_config_path ../delay_comparison/demo_comparison/WS/yin_yang.yaml --sweep_config_path ../delay_comparison/demo_comparison/WS/sweep_yin_yang.yaml --verbose
# python3 sweep.py continue --sweep_results_dirname ../delay_comparison/demo_comparison/WS --verbose



### Check the state of the sweeps
# python3 sweep.py print_state --sweep_results_dirname delay_comparison/demo_comparison/
# python3 sweep.py continue --sweep_results_dirname delay_comparison/demo_comparison/ --verbose
