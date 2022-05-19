[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Stable and interpretable unrolled dictionary learning

## PUDLE (Provable Unfolded Dictionary LEarning)

### Results

Trained models and results are stored in `results/save_results`.

Figures are stored in `results/figures`. To generate them run predict script on the trained model and its corresponding visualization script.

### PATH

For any scripts to run, make sure you are in `src` directory.

### Configuration


Check `init_params()` inside each of the script to find the detailed parameters. For example, below is the default parameters of `train_dense.py` scirpt.

```
params = {
    "exp_name": "mnist/exp1",
    "network": "AE",
    "class_list": [0, 1, 2, 3, 4],
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "random_date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    "dataset_name": "mnist",
    "shuffle": True,
    "batch_size": 32,
    "num_workers": 4,
    "overfit_to_only": None,
    # data processing
    "data_normalize": False,
    "data_whiten": False,
    "blackandwhite": True,
    # related to the Network
    "num_class": 5,
    "beta": 0,
    "init_model_path": None,
    "m": 784,
    "p": 500,
    "lam": 0.7,
    "step": 1,
    "num_layers": 15,
    "twosided": False,
    # related to the optimizer
    "lr": 1e-4,
    "lr_step": 200,
    "lr_decay": 0.1,
    "adam_eps": 1e-15,
    # related to DLLoss
    "lam_loss": 0.7,
    "rho_loss": 0,
    "noise_std": 0,
    #
    "normalize": False,
    "num_epochs": 200,
    #
    "train_val_split": 1,
    "log_info_epoch_period": 10,
    "log_model_epoch_period": 200,
    "log_fig_epoch_period": 25,
    "tqdm_prints_disable": False,
    "code_reshape": (25, 20),
    "data_reshape": (28, 28),
    #
    "random_split_manual_seed": 1099,
}

```

### Training

To train/run for an experiment with a dense/matrix dictionary:

`python train_dense.py`

To train/run for convolutional model using black and white images:

`python train_conv_bw.py`

To train/run for convolutional model using color images such as CIFAR:

`python train_conv_color.py`

### Results

When training is done, the results are saved in `results/{experiment_name}_{random_date}`.

`random_date` is a datetime string generated at the beginning of the training.


### Dependencies

* torch-1.7.1
* tqdm-4.35.0
* matplotlib-3.4.0
* sacred-0.7.5
* torchvision-0.8.2
* numpy-1.20.2
* sporco-0.1.11
* scipy-1.6.2
* tensorboardX-2.2
