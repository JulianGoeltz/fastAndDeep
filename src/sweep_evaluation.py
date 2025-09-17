import os
from sweep import list_runs
import yaml
import pandas as pd
import numpy as np
import training
import argparse
import utils
import torch



def to_list_recursive(data, np_type):
    if isinstance(data, (str, int)):
        return data
    elif isinstance(data, (float)) and data.is_integer():
        return int(data)
    elif isinstance(data, (float)):
        return np_type(data)
    else:
        return [to_list_recursive(item, np_type) for item in data]


def extract_result_df(list_of_dirname, include_err_by_epoch=True, include_parameters=True, include_tout = True, include_metadata=True):
    assert len(list_of_dirname) > 0,'No epoch directory found in the sweep results directory, you should let finish at least one training'

    list_of_configs = [yaml.safe_load(open(os.path.join(dirname, 'config.yaml'), 'r')) for dirname in list_of_dirname]
    list_of_updates = [yaml.safe_load(open(os.path.join(dirname, 'sweep_updates.yaml'), 'r')) for dirname in list_of_dirname]
    list_of_last_epoch_dir = [os.path.join(base_dir, 'epoch_' + str(config['training_params']['epoch_number'])) for base_dir, config in zip(list_of_dirname, list_of_configs)]

    # Load the result_dict
    list_of_result_dict = [training.load_result_dict(last_epoch_dir, config['dataset_params']['name'])
                           for last_epoch_dir, config in zip(list_of_last_epoch_dir, list_of_configs)]

    device = utils.get_default_device()
    list_of_criterion = [utils.GetLoss(config['training_params'],
                                       config['network_layout']['layers'][-1]['size'],
                                       config['default_neuron_params']['tau_syn'], device)
                         for config in list_of_configs]
    # Add the test_acc
    def compute_error(criterion, test_spiketimes, test_labels):
        selected_classes = criterion.select_classes(test_spiketimes)
        correct_predictions = 1.0*(selected_classes == test_labels)
        accuracy = correct_predictions.mean()
        return accuracy

    list_of_test_labels = [torch.tensor(np.load(os.path.join(last_epoch_dir, "yin_yang_test_labels.npy")))
                           for last_epoch_dir in list_of_last_epoch_dir]
    list_of_test_spiketimes = [torch.tensor(np.load(os.path.join(last_epoch_dir, "yin_yang_test_spiketimes.npy")))
                           for last_epoch_dir in  list_of_last_epoch_dir]
    list_of_test_acc = [compute_error(criterion, test_spiketimes, test_labels)
                        for criterion, test_spiketimes, test_labels in zip(list_of_criterion, list_of_test_spiketimes, list_of_test_labels)]
    [list_of_result_dict[i].update({"final_test_accuracy": acc}) for i, acc in enumerate(list_of_test_acc)]

    list_of_final_val_err = 1.-np.array([result_dict['all_validate_accuracy'][-1] for result_dict in list_of_result_dict])
    list_of_final_train_err = 1.-np.array([result_dict['all_train_accuracy'][-1] for result_dict in list_of_result_dict])
    list_of_final_test_err = 1.-np.array([result_dict['final_test_accuracy'] for result_dict in list_of_result_dict])

    list_of_number_of_parameters = np.array([
        sum([np.size(layer['params'][-1]) for layer in result_dict['all_parameters'].values()])
        for result_dict in list_of_result_dict])

    list_of_params_name = ["_".join(str(item) for item in update['path']) for update in list_of_updates[0]]
    list_of_params_values = np.array([[update['value'] for update in updates] for updates in list_of_updates])

    # Incorporate the additional list data as object type columns
    data = {
        **{(f'params/{name}'): values for name, values in zip(list_of_params_name, list_of_params_values.T) if name.split('_')[-1] != 'seed'},
        **{(f'seed/{name}'): values for name, values in zip(list_of_params_name, list_of_params_values.T) if name.split('_')[-1] == 'seed'},
        ('metrics/number_of_parameters'): list_of_number_of_parameters,
        ('metrics/final_val_err'): list_of_final_val_err,
        ('metrics/final_train_err'): list_of_final_train_err,
        ('metrics/final_test_err'): list_of_final_test_err,
    }

    if include_metadata:
        data['metadata/path'] = list_of_dirname

    if include_err_by_epoch:
        data['metrics_epoch/val_err'] = list(1.-np.array([np.array(result_dict['all_validate_accuracy'][1:])
                                                          for result_dict in list_of_result_dict]))
        data['metrics_epoch/train_err'] = list(1.-np.array([np.array(result_dict['all_train_accuracy'])
                                                            for result_dict in list_of_result_dict]))
    if include_parameters:
        # list of params is a list (size layers) of array (size : n_experiments) of list (size number of epochs)) of list of parameters (size : n_pre x n_post)
        # list_of_parameters = to_list_recursive([[list(result_dict['all_parameters'].values())[layer_idx]['params']
        #                                          for result_dict in list_of_result_dict]
        #                                         for layer_idx in range(len(list_of_result_dict[0]['all_parameters']))],
        #                                        float)

        # Only save the last epoch
        # list of parameters is a list (size : layers) of list (size : n_experiments) of list of parameters (size : n_pre x n_post)
        list_of_parameters = to_list_recursive([[np.array(list(result_dict['all_parameters'].values())[layer_idx]['params'][-1]).ravel()
                                                 for result_dict in list_of_result_dict]
                                                for layer_idx in range(len(list_of_result_dict[0]['all_parameters']))], float)

        list_of_layer_name = [f"parameters_epoch/{str(i)}_{layer['name']}" for i, layer in enumerate(list_of_result_dict[0]['all_parameters'].values())]
        data.update({key: list(value) for key, value in zip(list_of_layer_name, list_of_parameters)})

    if include_tout:
        data[('output_epoch/mean_validate_outputs_sorted')] = to_list_recursive([result_dict['mean_validate_outputs_sorted'] for result_dict in list_of_result_dict], float)
        data[('output_epoch/std_validate_outputs_sorted')] = to_list_recursive([result_dict['std_validate_outputs_sorted'] for result_dict in list_of_result_dict], float)

    return pd.DataFrame(data, columns=data.keys())

def select_experiments(result_df, n_experiments):
    # Get the list of categories
    categories = ['params', 'seed', 'metrics', 'metrics_epoch', 'parameters_epoch', 'output_epoch', 'metadata']
    params_list, seed_list, metrics_list, metrics_epoch_list, parameters_epoch_list, output_epoch_list, metadata_list = \
        ([col_name for col_name in result_df.columns if col_name.split('/')[0] == category] for category in categories)


    # Check for if there are different results for the same seed and params
    check_df = result_df.drop_duplicates(params_list + seed_list + metrics_list)
    error_duplication_df = check_df[check_df.duplicated(params_list + seed_list, keep=False)]
    for group, group_df in error_duplication_df.groupby(params_list + seed_list): # More detailed information
        print(f"Warning: Different results for the same seed and params :\n {group_df[('metadata','path')].to_string(index=False)}\n")
    if not error_duplication_df.empty: # Makes sure to fail
        raise ValueError("Different results for the same seed and params, "
                         "you should check that you did not change the initial configuration without using sweep.py")

    # Select only one unique experiment for the same setup (seed, params)
    unique_df = result_df.drop_duplicates(params_list + seed_list)

    # Select only min(max_num_seed, n_experiemnts) seeds
    seeds_per_experiments = [list(df[seed_list[0]]) for df in list(zip(*unique_df.groupby(params_list)))[1]]
    n_seeds_per_experiments = [len(seeds) for seeds in seeds_per_experiments]
    n_max_experiments, ind_max_experiements = max(n_seeds_per_experiments), np.argmax(n_seeds_per_experiments)

    n_selected = min(n_max_experiments, n_experiments)
    seeds_selected = seeds_per_experiments[ind_max_experiements][:n_selected]
    if n_selected != n_experiments:
        print(f"Warning: Only {n_selected} seeds {seeds_selected} are selected for the analysis")

    filtered_df = unique_df[unique_df[seed_list[0]].isin(seeds_selected)]

    # Check if there are missing seeds for a given experiment
    for group, group_df in filtered_df.groupby(params_list):
        missing_seeds = set(seeds_selected) - set(group_df[seed_list[0]])
        if missing_seeds:
            print(f"Missing seeds {missing_seeds} for the following experiment :\n {group_df[params_list].iloc[[0]]}")

    return filtered_df


def aggregate_experiments(result_df):
    # Get the list of categories
    categories = ['params', 'seed', 'metrics', 'metrics_epoch', 'parameters_epoch', 'output_epoch', 'metadata']
    params_list, seed_list, metrics_list, metrics_epoch_list, parameters_epoch_list, output_epoch_list, metadata_list = \
        ([col_name for col_name in result_df.columns if col_name.split('/')[0] == category] for category in categories)

    # Perform groupby and aggregation over the seeds for the metrics and metrics_epoch
    agg_funcs = {
        'mean': lambda x: np.mean(np.array(x.tolist()), axis=0),
        'std': lambda x: np.std(np.array(x.tolist()), axis=0),
        'median': lambda x: np.quantile(np.array(x.tolist()), 0.5, axis=0),
        '0': lambda x: np.quantile(np.array(x.tolist()), 0.00, axis=0),
        '25': lambda x: np.quantile(np.array(x.tolist()), 0.25, axis=0),
        '75': lambda x: np.quantile(np.array(x.tolist()), 0.75, axis=0),
        '100': lambda x: np.quantile(np.array(x.tolist()), 1.00, axis=0),
    }

    agg_df = pd.DataFrame({
        # Aggregate metrics and metrics epoch using the agg_func
        **{metric+f'_{func}': result_df.groupby(params_list)[metric].agg(agg_func)
           for metric in metrics_list + metrics_epoch_list
           for func, agg_func in agg_funcs.items()},

        # Aggregate the parameters, outputs and metadata while stacking them
        **{col: result_df.groupby(params_list)[col].agg(lambda x: np.stack(x.tolist()))
           for col in parameters_epoch_list + output_epoch_list + metadata_list},

        # Frequency of each combination
        ('metrics/frequency'): result_df.groupby(params_list).size()
    }).reset_index()

    return agg_df

def print_top_n(df, columns, metric = 'metrics/final_val_err_mean', n=5):
    df = df.sort_values(by= metric, ascending=True)
    counter = 1
    for ind, row in df.head(n=n).iterrows():
        print(f"Top {counter} :")
        for col in [metric]+ columns :
            value = f"{row[col]:.4f}" if isinstance(row[col], float) else row[col]
            print(f"{col} : {value}")
        print()  # For spacing between rows
        counter += 1

def eval_sweep(sweep_results_dirname, n_experiments, verbose=True, include_err_by_epoch=True, include_parameters=True,
               include_tout = True, include_metadata=True, aggregate=True):
    pd.set_option('display.max_colwidth', None)
    list_of_finished, list_of_unfinished = list_runs(sweep_results_dirname, verbose)

    if len(list_of_unfinished) > 0:
        print(f"Warning: {len(list_of_unfinished)} runs are unfinished:\n {list_of_unfinished} \n")

    if verbose: print("Extracting the results from the files...")
    result_df = extract_result_df(list_of_finished, include_err_by_epoch, include_parameters, include_tout, include_metadata)

    if verbose: print("Selecting a reduced number of experiments...")
    filtered_df = select_experiments(result_df, n_experiments=n_experiments)

    if verbose: print("Aggregatin the experiments...")
    if aggregate:
        agg_df = aggregate_experiments(filtered_df)
    else:
        agg_df = filtered_df
        agg_df['metrics/frequency'] = 1
        for datatype in ['train', 'val', 'test']:
            for metric in ['median', 'mean']:
                agg_df[f'metrics/final_{datatype}_err_{metric}'] = agg_df[f'metrics/final_{datatype}_err']
            for metric in ['std', '0', '25', '75', '100']:
                agg_df[f'metrics/final_{datatype}_err_{metric}'] = np.nan
        for datatype in ['train', 'val']:
            for metric in ['median', 'mean']:
                agg_df[f'metrics_epoch/{datatype}_err_{metric}'] = agg_df[f'metrics_epoch/{datatype}_err']
            for metric in ['std', '0', '25', '75', '100']:
                agg_df[f'metrics_epoch/{datatype}_err_{metric}'] = np.nan


    if verbose:
        print_top_n(agg_df, columns= [col for col in agg_df.columns if col.split('/')[0] == 'params'],  metric = 'metrics/final_val_err_mean', n=5)

    # Save the aggreated dataframe
    agg_df = pd.DataFrame({col: [to_list_recursive(item, np.float16) for item in agg_df[col]] for col in agg_df.columns})
    file_path = os.path.join(sweep_results_dirname, 'sweep_results_df.csv')
    agg_df.to_csv(file_path, index=False)
    print(f"Result Dataframe (with {agg_df.shape[0]} runs and {agg_df.shape[1]} attributes) saved in :"
          f"\n {file_path} (size: {os.path.getsize(file_path) / 1e6} MB)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep evaluation")
    default_sweep_results_dirname = os.path.join('..', 'sweep_results')
    parser.add_argument('--sweep_results_dirname', type=str, default=default_sweep_results_dirname,
                       help='Directory for the sweep results')
    parser.add_argument('--n_experiments', type=int, default=np.inf, help='Number of experiments to select')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--include_err_by_epoch', action='store_true', help='Include the error by epoch')
    parser.add_argument('--include_parameters', action='store_true', help='Include the parameters')
    parser.add_argument('--include_tout', action='store_true', help='Include the output')
    parser.add_argument('--include_metadata', action='store_true', help='Include the metadata')

    parser.add_argument('--dont_aggregate', action='store_true', help='Skip aggregation over seeds')


    args = parser.parse_args()

    eval_sweep(
        args.sweep_results_dirname, n_experiments=args.n_experiments,
        verbose=args.verbose, include_err_by_epoch=args.include_err_by_epoch,
        include_parameters=args.include_parameters, include_tout=args.include_tout,
        include_metadata=args.include_metadata,
        aggregate=not args.dont_aggregate,
    )

### Lauching the delay_comparison evaluation

# For W
# python3 sweep_evaluation.py --sweep_results_dirname ../delay_comparison/demo_comparison/W --verbose

# For WA
# python3 sweep_evaluation.py --sweep_results_dirname ../delay_comparison/demo_comparison/WA --verbose

# For WAD
# python3 sweep_evaluation.py --sweep_results_dirname ../delay_comparison/demo_comparison/WAD --verbose

# For WD
# python3 sweep_evaluation.py --sweep_results_dirname ../delay_comparison/demo_comparison/WD --verbose

# For WS
# python3 sweep_evaluation.py --sweep_results_dirname ../delay_comparison/demo_comparison/WS --verbose
