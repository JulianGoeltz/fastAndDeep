# Description
Training networks of LIF neurons including delays to classify patterns using `pytorch`, for explanations see publications below.

## References
* Fast and deep: energy-efficient neuromorphic learning with first-spike times; **Julian Göltz∗, Laura Kriener∗**, *Andreas Baumbach, Sebastian Billaudelle, Oliver Breitwieser, Bejamin Cramer, Dominik Dold, Ákos F. Kungl, Walter Senn, Johannes Schemmel, Karlheinz Meier, Mihai A. Petrovici*; https://www.nature.com/articles/s42256-021-00388-x (https://arxiv.org/abs/1912.11443)
* DelGrad: exact event-based gradients for training delays and weights on spiking neuromorphic hardware; **Julian Göltz∗, Jimmy Weber∗, Laura Kriener∗**, *Sebastian Billaudelle, Peter Lake, Johannes Schemmel, Melika Payvand, Mihai A. Petrovici*; https://www.nature.com/articles/s41467-025-63120-y (https://arxiv.org/abs/2404.19165)

If you run into any problems, have any comments or question, please feel free to create issues or pull requests, or contact us directly (julian.goeltz@kip.uni-heidelberg.de, laura.kriener@unibe.ch, jimmy.weber@ini.uzh.ch).

> [!NOTE]
> The manuscript ["Synchronization and semantization in deep spiking networks"](https://doi.org/10.48550/arXiv.2508.12975) analyses networks trained with an earlier version of this code base.
> Both the [trained networks](https://github.com/JulianGoeltz/fastAndDeep/tree/WeHaveToGoDeeper/experiment_pretrained) and the [minor adaptations](https://github.com/JulianGoeltz/fastAndDeep/blob/WeHaveToGoDeeper/experiment_pretrained/mnist_fourlayer_seed1_daleslaw/daleslaw.patch) to the training code can be found in the branch [`WeHaveToGoDeeper`](https://github.com/JulianGoeltz/fastAndDeep/tree/WeHaveToGoDeeper).

> [!NOTE]
> For the code version that was used for the Fast&Deep manuscript, including the MNIST trainings, see the [previous release](https://github.com/JulianGoeltz/fastAndDeep/releases/tag/v1.0.0).

## How to run stuff
### Simulation
```python
python3 experiment.py train ../experiment_configs/yin_yang.yaml
python3 experiment.py eval ../experiment_results/<subfolder>
```
### Hardware with strobe
```python
# after waf building
export PYTHONPATH="${PWD}/py:$PYTHONPATH";
# and probably with slurm and singularity:
# srun -p cube --wafer 74 --fpga-without 3 --pty --time 2-0:0:0 singularity exec --app dls $build_container zsh
# (after running calibration python py/generate_calibration.py --output calibrations/${SLURM_HARDWARE_LICENSES}.npz)
python3 experiment.py train ../experiment_configs/yin_yang.yaml
python3 experiment.py eval ../experiment_results/<subfolder>
```
### Hardware with pynn
```python
module load pynn-brainscales
python delay_utils.py doAll
python experiment.py train ../experiment_configs/yin_yang_hxpynn.yaml
```

## Requirements
Clone the repository with
```
git clone git@github.com:JulianGoeltz/fastAndDeep.git
```
install the requirements (or use singularity/apptainer from https://openproject.bioai.eu/containers/).
### Python packages
It is often advisable to install packages in a new virtual environment (`python -m venv <path and name of venv>`).
In this venv, execute (the specific versions are ones that can be installed with `Python 3.13.7`; if you run a different python version, everything should work without specifying the versions of the packages)
```
pip install -r requirements.txt
```

### GPU support
If you want a native GPU implementation of the LambertW function for speed up, you can use the code available in the folder `pytorch_cuda_lambertw`.
In `pytorch_cuda_lambertw` run `python setup.py install --user` to install the function (tested for `cudatoolkit 10.2` and an older `python` and `pytorch`).

### Datasets
`pytorch` has functionality that automatically downloads the standard data sets (for us MNIST), for this an internet connection is necessary (when executing on compute nodes on a HPC there might not be an internet connection, execute once on the frontend in that case), e.g. with
```
python -c "import torchvision; print(torchvision.datasets.MNIST('../data/mnist', train=True, download=True))"
```

The [`yin_yang`](https://github.com/lkriener/yin_yang_data_set) is included as a submodule, to initialise execute once
```
git submodule update --init
```

## Functionality
### Run demo notebook
Assuming the software was installed in a virtualenv, the environment must be provided as a kernel for jupyter:
```
ipython kernel install --user --name=<venvname>
```
The demo notebook is located in the `src` directory.

### Training
Test e.g. the `yin_yang` data set by going into `src` and running
```
python3 experiment.py train ../experiment_configs/yin_yang.yaml
```
(training for 300 epochs takes about 30 minutes on a T470 Thinkpad without GPU; setting `live_plot` plots the current training accuracy, check `src/live_accuracy.png`).
Results are then saved in a subfolder of the `experiment_results` directory. The name of the subfolder is the name of the config file and a timestamp.
### Inference
See the accuracy of a saved network by running
```
python3 experiment.py inference ../experiment_results/<subfolder> [number of epochs]
```
e.g., to check the accuracy of the given `yin_yang` network, do
```
python experiment.py inference ../experiment_pretrained/yin_yang_H30_WA 300
```
(This also works for the other given networks in `experiment_pretrained`, i.e., for `fullmnist_et_150epochs` and `yinyang_et_300epochs`)
### Plotting
To plot the results run
```
python3 experiment.py eval ../experiment_results/<subfolder>
```
This will create plots similar to the ones found in the paper.
The plots are saved in the same subfolder as the data.
### Continue Training
To continue training from a savepoint (epochs that are savepoints are specified in the configs) run
```
python3 experiment.py continue "../experiment_results/<subfolder>/" <savepoint (epoch nr)> "<new savepoint 1>, <new savepoint 2>, ..."
```
### Network setup
* in [`experiment_configs/yin_yang_H30_WS.yaml`](experiment_configs/yin_yang_H30_WS.yaml) you can find an example setup with one hidden layer of 30 hidden neurons and synaptic delays
* between two neuron layers there needs to be one or more `DelayLayer` (or `BroadcastLayer` if you don't need delays)
* the layers can be of type `NeuronLayer`, `DelayAxonal`, `DelaySynaptic`, `DelayDendritic`, `BroadcastLayer`, `Biases`, `Multiplex`. For details on each, check their definition in `utils.py`
* `neuron_params` can be set layer specific, which means you can set `tau_m` or `model_tau_ratio` etc per layer.

### Sweep Execution Guide

#### Running Sweeps
1. Execute sweeps on your cluster with:
   ```
   python3 sweep.py sweep --sweep_results_dirname <results_dir> --initial_config_path <config_path> --sweep_config_path <sweep_config_path> [--available_CPU <n>] [--available_GPU <gpu_ids>] [--verbose]
   ```
   - This runs the `sweep` function, which initiates a new sweep based on the specified configuration files. You can specify the number of available CPUs and GPUs if needed.

1.1. Continue an existing sweep with:
   ```
   python3 sweep.py continue --sweep_results_dirname <results_dir> [--available_CPU <n>] [--available_GPU <gpu_ids>] [--verbose]
   ```
   - This runs the `continue_sweep` function, allowing you to resume a previously interrupted sweep.

1.2. Check the state of current sweeps with:
   ```
   python3 sweep.py print_state --sweep_results_dirname <results_dir>
   ```
   - This runs the `states_of_sweeps` function, which prints the current status of the ongoing or completed sweeps in the specified directory.

#### Evaluating Sweeps
2. Extract the results into `sweep_result_df.csv`:
   ```
   python3 sweep_evaluation.py --sweep_results_dirname <results_dir> --n_experiments <n> [--verbose] [--include_err_by_epoch] [--include_parameters] [--include_tout] [--include_metadata]
   ```

#### Transferring Results
3. Transfer the `sweep_result_df.csv` to your local machine:
   ```
   scp <source> <destination>
   ```

#### Plotting Results

4. Plot the results on your local machine with `sweep_plot.py`:
   ```
   python3 sweep_plot.py --sweep_results_dirname <results_dir> [--plot_err_by_epoch]
   ```

## Repository structure
* `experiment_configs` has the configuration files for the experiments, they are given as an argument for the `experiment.py` calls
* `experiment_pretrained` includes some trained models to allow for faster inferences without the need to train yourself
* `pytorch_cuda_lambertw` includes source files that enable GPU execution of the lambertW function, see [above](#gpu-support).
* `src` is where the `python` source code is located:
	* especially `experiment.py` that is used for training, inference and evaluation (it depends on `evaluation.py`, `networks.py`, `training.py`, `utils.py` and the `utils_spiketime.py`)
	* in `networks.py` the `pytorch` network is defined
	* `py/` has additional source files necessary for the execution on BrainScaleS-2
	* `calibration/` includes a calibration file for chip
