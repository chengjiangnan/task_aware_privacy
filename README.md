# Task-aware Privacy Preservation for Multi-dimensional Data #
This is the code release for our ICML 2022 paper "Task-aware Privacy Preservation for Multi-dimensional Data".

arXiv: https://arxiv.org/abs/2110.02329

### Setup ###

* export `LDP_CODESIGN_DIR` in your bashrc to be the path of this repo
    * example `export LDP_CODESIGN_DIR=~/Desktop/task_aware_privacy/`

* have `python3` installed. My version is 3.7.4

* have other python packages installed, including `torch`, `numpy`, `pandas`, etc


### Models ###

There are 5 subdirectories for evaluation under `./codesign/`:

* `eval_theoretical` computes some theoretical results under Assumption 1;

* `eval_household`, `eval_valuation` and `eval_cancer` correspond to the hourly household power consumption, real estate valuation, breast cancer detection applications, respectively; `eval_mnist` corresponds to an evaluation on high-dimensional image dataset -- MNIST.

The parameters should be specified in the `get_model_specific_info.sh` first.

### Approaches ###

There are 3 approaches: `codesign`, `sep_design`, and `benchmark`, corresponding to `task-aware`, `privacy-agnostic` and `task-agnostic` approaches mentioned in our paper.  

### Theoretical Result ###
```
cd ./codesign/eval_theoretical

# Compute the theoretical losses
./laplace_theoretical_loss.sh

# Plot loss under different privacy budgets
./laplace_loss_privacy_plot.sh 
```

### Simulation ###

* [linear model and l2-loss] For `household` application:
```
cd ./codesign/eval_household

# Read raw data
./household_data_generation.sh

# Co-design, sep-design, benchmark
./household_codesign_computation.sh 
./household_sep_design_computation.sh
./household_benchmark_computation.sh

# Plot loss under different privacy budgets / MSE for each x_i
./household_loss_privacy_plot.sh
./household_mse_plot.sh 
```

Caveat: due to randomness running the algorithms twice may produce slightly different results.

* [General Settings, regression] For `valuation` application:
```
cd ./codesign/eval_valuation

# Read raw data and train regressor
./valuation_data_generation.sh 
./valuation_train_regressor.sh

# Co-design, sep-design, benchmark
./valuation_codesign_train_encoder.sh
./valuation_sep_design_train_encoder.sh
./valuation_benchmark_train_encoder.sh

# Plot loss under different privacy budgets
./valuation_loss_privacy_plot.sh
```

* [General Settings, classification] For `cancer` application:
```
cd ./codesign/eval_cancer

# Read raw data and train classifier
./cancer_data_generation.sh
./cancer_train_classifier.sh 

# Co-design, sep-design, benchmark
./cancer_codesign_train_encoder.sh
./cancer_sep_design_train_encoder.sh 
./cancer_benchmark_train_encoder.sh

# Plot loss under different privacy budgets
./cancer_loss_privacy_plot.sh
```

* [General Settings, classification, minibatch] For `mnist` application:
```
cd ./codesign/eval_mnist

# Read raw data and train classifier
./mnist_data_generation.sh
./mnist_train_classifier.sh

# Co-design, sep-design, benchmark
./mnist_codesign_train_encoder.sh
./mnist_sep_design_train_encoder.sh
./mnist_benchmark_train_encoder.sh

# Plot loss under different privacy budgets
./mnist_loss_privacy_plot.sh
```

We also have a few scripts that visualize data/result in other ways, including (take `cancer` as an example):
```
# The distribution of data for each x_i and each label y (only for classification) 
./cancer_data_plot.sh

# Train loss evolution (for all the applications)
./cancer_train_losses_plot.sh
```

### Tuning Hyperparameters and Models ###

The number of back-propagation steps in each epoch can be specified in `./codesign/ML_functions/LDP_encoder_train.py` (or `LDP_encoder_train_minibatch.py`).

The encoder/decoder, classifier and regressor model can be further specified in the corresponding files under `codesign/models`. (For our paper, to repeat our experiment given in the appendix, the activation functions of the encoder/decoder model shall be changed manually.) 
