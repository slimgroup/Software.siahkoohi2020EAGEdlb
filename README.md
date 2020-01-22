# A deep-learning based Bayesian approach to seismic imaging and uncertainty quantification

Codes for generating results in Siahkoohi, A., Rizzuti, G., and Herrmann, F.J., Jan. 2020. A deep-learning based Bayesian approach to seismic imaging and uncertainty quantification. arXiv preprint [arXiv:2001.04567](https://arxiv.org/pdf/2001.04567.pdf).


## Prerequisites

This code has been tested on Deep Learning AMI (Amazon Linux 2) Version 26.0 on Amazon Web Services (AWS), using `c5.4xlarge` and `g3s.xlarge` instances. Using GPU is not essential since PDE solves dominate the computation. Also, we use GCC compiler version 7.3.1.

This software is based on [Devito-3.5](https://github.com/devitocodes/devito/releases/tag/v3.5) and [PyTorch-1.4.0](https://github.com/pytorch/pytorch/releases/tag/v1.4.0). Additionally, we borrow `JAcoustic_codegen.py`\, `PyModel.py`\, `PySource.py`\, `utils.py`\, and `checkpoint.py` from [JUDI](https://github.com/slimgroup/JUDI.jl), a framework for large-scale seismic modeling and inversion that abstracts forward/adjoint nonlinear and Born modeling Devito operators.

Follow the steps below to install the necessary libraries:

```bash
cd $HOME
git clone https://github.com/alisiahkoohi/seismic-imaging-with-SGLD.git
git clone --branch v3.5 https://github.com/devitocodes/devito.git

cd $HOME/devito
conda env create -f environment.yml
source activate devito
pip install -e .
export DEVITO_ARCH=gnu
export OMP_NUM_THREADS=16 # or any other number of threads you prefer
export DEVITO_OPENMP=1

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install matplotlib
pip install tqdm
pip install h5py
pip install tensorboardX
```

## Dataset

The 2D Overthrust velocity model we use is obtained from [JUDI](https://github.com/slimgroup/JUDI.jl)'s GitHub repository and will be automatically downloaded and placed at `vel_dir/` directory upon running `run-training.sh`. See below for more details. The observed data will be also downloaded into `vel_dir/data/` upon starting the inversion. See below for more details.

## Script descriptions

`run-training.sh`\: script for running inversion/training. It downloads the velocity model into `vel_dir/` and creates `checkpoint/`, `log/`, and `sample/` directories in `$HOME/seismic-imaging-with-SGLD` for storing intermediate parameters, loss function log, and samples for monitoring, respectively. The drawn samples from the posterior will be stored at `training-logs.pt` located at `checkpoint/`.

`run-test.sh`\: script for generating figures after (while) training by calling `src/sample.py`\,

`src/main.py`\: constructs `LearnedImaging` class using given arguments in `run-training.sh`\, defined in `model.py` and calls `train` function in the defined  `LearnedImaging` class.

`src/model.py`: includes `LearnedImaging` class definition, which involves `train` and `test` functions.

`src/sample.py`: script for loading the obtained samples after (while) training and creating figures in the manuscript. Not that the script will throw an assertation error if there are no samples drawn yet.

### Running the code

To perform inversion/training, run:

```bash

bash run-training.sh

```

To generate and save figures shown in the manuscript, run:

```bash

bash run-test.sh

```

The figures will be saved in `sample/` directory.


## Questions

Please contact alisk@gatech.edu for further questions.

## Acknowledgments

The authors thank Zezhou Cheng for his open-access [GitHub repository](https://github.com/ZezhouCheng/GP-DIP). We also thank Philipp Witte for his contribuitions in integrating Devito operators in PyTorch.


## Author

Ali Siahkoohi
