# Description
Training a network of LIF neurons to classify patterns using `pytorch` as used in Goeltz, Kriener, et. al (see below).

If you run into any problems, have any comments or question, please feel free to create issues or pull requests, or contact us directly (julian.goeltz@kip.uni-heidelberg.de and laura.kriener@unibe.ch).

> [!NOTE]
> We are currently working on expanding the existing public code base to include (clean) code to train transmission delays alongside synaptic weights ([a preprint is available](https://arxiv.org/abs/2404.19165)).
This code will be publicly available soon.


## Requirements
Clone the repository with
```
git clone git@github.com:JulianGoeltz/fastAndDeep.git
```
install the requirements and you are ready to go.
### Python packages
It is often advisable to install packages in a new virtual environment (`python -m venv <path and name of venv>`).
In this venv, execute (the version numbers are the ones used for the publication; if you run a newer python version, everything should work with newer versions of the packages)
```
pip install -r requirements.txt
```


Depending on your internet connection, this should take some minutes to an hour.
### GPU support
If you want GPU support for speed up, you will have to get your `pytorch` to work with cuda.
The above `pytorch` and `cudatoolkit 10.2` can be used to install the special lambertw function for GPU execution (check folder `pytorch_cuda_lambertw`, run `python setup.py install --user` in there to install the function).
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
(training for 300 epochs takes about 30 minutes on a T470 Thinkpad without GPU; the current training accuracy is plotted by default, check `src/live_accuracy.png`).
Results are then saved in a subfolder of the `experiment_results` directory. The name of the subfolder is the name of the config file and a timestamp.
### Inference
See the accuracy of a saved network by running
```
python3 experiment.py inference ../experiment_results/<subfolder>
```
e.g. to check the accuracy of the given `16x16_mnist` network, do (this will take approximately 5 minutes; first mnist execution will download the dataset, and process the images by resizing them)
```
python experiment.py inference ../experiment_pretrained/16x16mnist_et_200epochs
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
### Training with doubletime
There are two learning rules, one for `tau_mem=tau_syn` (default) and one for `tau_mem=2*tau_syn`.
In order to use the latter, in `utils.py` change `import utils_spiketime_et as utils_spiketime` to `import utils_spiketime_dt as utils_spiketime`, adapt the neuron parameters in the configuration `.yaml` file (`tau_mem` has to be specified to `2 * tau_syn`) and you can start training.

## Repository structure
* `experiment_configs` has the configuration files for the experiments, they are given as an argument for the `experiment.py` calls
* `experiment_pretrained` includes some trained models to allow for faster inferences without the need to train yourself
* `pytorch_cuda_lambertw` includes source files that enable GPU execution of the lambertW function, see [above](#gpu-support).
* `src` is where the `python` source code is located:
	* especially `experiment.py` that is used for training, inference and evaluation (it depends on `evaluation.py`, `training.py` and the `util*.py`)
	* in `training.py` the `pytorch` network is defined
	* `py/` has additional source files necessary for the execution on BrainScaleS-2
	* `calibration/` includes a calibration file for chip

## References
* Fast and deep: energy-efficient neuromorphic learning with first-spike times; *J. Göltz∗, L. Kriener∗, A. Baumbach, S. Billaudelle, O. Breitwieser, B. Cramer, D. Dold, A. F. Kungl, W. Senn, J. Schemmel, K. Meier, M. A. Petrovici*; https://www.nature.com/articles/s42256-021-00388-x (https://arxiv.org/abs/1912.11443)
