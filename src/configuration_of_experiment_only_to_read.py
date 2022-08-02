"""
Copyright (c) 2021 Bahareh Tolooshams

conf for multi purposes

:author: Bahareh Tolooshams
"""

import torch

from sacred import Experiment, Ingredient

config_ingredient = Ingredient("cfg")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@config_ingredient.config
def cfg():
    hyp = {
        "experiment_name": "default",
        "dataset": "simulated",
        "data_filename_path": "../data/simulated_data.pt",
        "task": "train_ae",
        "network": "AE",
        "num_layers": 200,
        "n": 10000,
        "m": 50,  # x dimension
        "p": 100,  # z dimension
        "s": 5,  # sparsity
        "lam": 0.2,
        "beta": 1,
        "step": 0.2,
        "threshold": 0.45,
        "normalize": True,
        "shuffle": False,
        "code_dist": "uniform",
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 10000,
        "init_close": True,
        "dict_init": 0.1,
        "grad_method": "g_dec",
        "num_epochs": 600,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-8,
        "manual_seed": True,
        "num_workers": 0,
        "min_num_layers": 1,
        "max_num_layers": 100,
        "netstar_num_layers": 1000,
        "info_period": 1,
        "info_save_period": 500,
        "save_period": 500,
        "sporco_altmin_iter": 200,
        "sporco_rho": 1,
        "sporco_L": 1,
        "crop_dim": (129, 129),
        "dictionary_dim": 9,
        "stride": 4,
        "num_conv": 64,
        "noise_std": 0,
        "train_path": "../data/CBSD432/",
        "test_path": "../data/BSD68/",
        "supervised": True,
        "device": device,
    }


####################################
####### Image Denoising Task #######
####################################

############# T 15

### ht


@config_ingredient.named_config
def denoising_g_aels_T15_htp02():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/htp02",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAEhard",
        "num_layers": 15,
        "threshold": 0.02,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


@config_ingredient.named_config
def denoising_g_aels_T15_htp05():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/htp05",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAEhard",
        "num_layers": 15,
        "threshold": 0.05,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


@config_ingredient.named_config
def denoising_g_aels_T15_htp08():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/htp08",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAEhard",
        "num_layers": 15,
        "threshold": 0.08,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


@config_ingredient.named_config
def denoising_g_aels_T15_htp1():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/htp1",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAEhard",
        "num_layers": 15,
        "threshold": 0.1,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


### lam 0.08


@config_ingredient.named_config
def denoising_g_dec_T15_lamp08():
    hyp = {
        "experiment_name": "denoising/g_dec/T15/lamp08",
        "grad_method": "g_dec",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.08,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


@config_ingredient.named_config
def denoising_g_aels_T15_lamp08():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/lamp08",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.08,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


### lam 0.12


@config_ingredient.named_config
def denoising_g_dec_T15_lamp12():
    hyp = {
        "experiment_name": "denoising/g_dec/T15/lamp12",
        "grad_method": "g_dec",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.12,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


@config_ingredient.named_config
def denoising_g_aels_T15_lamp12():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/lamp12",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.12,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


### lam 0.16


@config_ingredient.named_config
def denoising_g_dec_T15_lamp16():
    hyp = {
        "experiment_name": "denoising/g_dec/T15/lamp16",
        "grad_method": "g_dec",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.16,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


@config_ingredient.named_config
def denoising_g_aels_T15_lamp16():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/lamp16",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.16,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


### lam 0.2


@config_ingredient.named_config
def denoising_g_dec_T15_lamp2():
    hyp = {
        "experiment_name": "denoising/g_dec/T15/lamp2",
        "grad_method": "g_dec",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.2,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


@config_ingredient.named_config
def denoising_g_aels_T15_lamp2():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/lamp2",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.2,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


### lam 0.24


@config_ingredient.named_config
def denoising_g_dec_T15_lamp24():
    hyp = {
        "experiment_name": "denoising/g_dec/T15/lamp24",
        "grad_method": "g_dec",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.24,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


@config_ingredient.named_config
def denoising_g_aels_T15_lamp24():
    hyp = {
        "experiment_name": "denoising/g_aels/T15/lamp24",
        "grad_method": "g_aels",
        "dataset": "folder",
        "task": "denoising",
        "network": "CAE",
        "num_layers": 15,
        "lam": 0.24,
        "step": 0.1,
        "batch_size": 1,
        "noise_std": 25,
        "num_epochs": 400,
        "lr": 1e-4,
        "eps": 1e-3,
        "shuffle": True,
        "init_close": False,
        "manual_seed": False,
        "info_period": 1,
        "save_period": 100,
    }


####################################
####### Baseline Comparisons #######
####################################

### noodl specification

# sporco
@config_ingredient.named_config
def dictionarylearning_baselines_sporco_s10():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/sporco/s10",
        "task": "sporco",
        "data_filename_path": None,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 10,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "dict_init": 0.005,
        "lr": 1e-3,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_sporco_s20():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/sporco/s20",
        "task": "sporco",
        "data_filename_path": None,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 20,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "dict_init": 0.005,
        "lr": 1e-3,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_sporco_s40():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/sporco/s40",
        "task": "sporco",
        "data_filename_path": None,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 40,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "dict_init": 0.005,
        "lr": 1e-3,
    }


# AE
@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s10():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_aels/s10",
        "task": "baseline_comparison_ae",
        "data_filename_path": None,
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 10,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s20():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_aels/s20",
        "task": "baseline_comparison_ae",
        "data_filename_path": None,
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 20,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s40():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_aels/s40",
        "task": "baseline_comparison_ae",
        "data_filename_path": None,
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 40,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


# AE hard
@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s10_hard():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_aels/s10_hard",
        "task": "baseline_comparison_ae",
        "network": "AEhard",
        "data_filename_path": None,
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 10,  # sparsity
        "threshold": 0.1,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s20_hard():
    hyp = {
        "experiment_name": "dictionary_learning/noodl/baselines/s20_hard",
        "task": "baseline_comparison_ae",
        "network": "AEhard",
        "data_filename_path": None,
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 20,  # sparsity
        "threshold": 0.1,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s40_hard():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_aels/s40_hard",
        "task": "baseline_comparison_ae",
        "network": "AEhard",
        "data_filename_path": None,
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 40,  # sparsity
        "threshold": 0.1,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


# AE decay
@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s10_decay():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_aels/s10_decay",
        "task": "baseline_comparison_ae",
        "data_filename_path": None,
        "network": "AEdecay",
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 10,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s20_decay():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_aels/s20_decay",
        "task": "baseline_comparison_ae",
        "data_filename_path": None,
        "network": "AEdecay",
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 20,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_g_aels_s40_decay():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_aels/s40_decay",
        "task": "baseline_comparison_ae",
        "data_filename_path": None,
        "network": "AEdecay",
        "grad_method": "g_aels",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 40,  # sparsity
        "lam": 0.2,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 1e-3,
        "optim": "ADAM",
        "eps": 1e-3,
    }


# noodl
@config_ingredient.named_config
def dictionarylearning_baselines_g_noodl_s10():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_noodl/s10",
        "task": "baseline_comparison_ae",
        "network": "AEhard",
        "data_filename_path": None,
        "grad_method": "g_noodl",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 10,  # sparsity
        "threshold": 0.1,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 20,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_g_noodl_s20():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_noodl/s20",
        "task": "baseline_comparison_ae",
        "network": "AEhard",
        "data_filename_path": None,
        "grad_method": "g_noodl",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 20,  # sparsity
        "threshold": 0.1,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 20,
    }


@config_ingredient.named_config
def dictionarylearning_baselines_g_noodl_s40():
    hyp = {
        "experiment_name": "dictionary_learning/baselines/g_noodl/s40",
        "task": "baseline_comparison_ae",
        "network": "AEhard",
        "data_filename_path": None,
        "grad_method": "g_noodl",
        "num_layers": 100,
        "n": 50000,
        "m": 1000,  # x dimension
        "p": 1500,  # z dimension
        "s": 40,  # sparsity
        "threshold": 0.1,
        "step": 0.2,
        "c_min": 1.0,
        "c_max": 2.0,
        "twosided": True,
        "batch_size": 50,
        "dict_init": 0.005,
        "num_epochs": 1,
        "lr": 20,
    }


####################################
####### Dictionary Learning ########
####################################

### T 100


@config_ingredient.named_config
def dictionarylearning_g_dec_T100():
    hyp = {
        "experiment_name": "dictionary_learning/g_dec/T100",
        "grad_method": "g_dec",
        "num_layers": 100,
        "dict_init": 0.1,
    }


@config_ingredient.named_config
def dictionarylearning_g_aelasso_T100():
    hyp = {
        "experiment_name": "dictionary_learning/g_aelasso/T100",
        "grad_method": "g_aelasso",
        "num_layers": 100,
        "dict_init": 0.1,
    }


@config_ingredient.named_config
def dictionarylearning_g_aels_T100():
    hyp = {
        "experiment_name": "dictionary_learning/g_aels/T100",
        "grad_method": "g_aels",
        "num_layers": 100,
        "dict_init": 0.1,
    }


### T 50


@config_ingredient.named_config
def dictionarylearning_g_dec_T50():
    hyp = {
        "experiment_name": "dictionary_learning/g_dec/T50",
        "grad_method": "g_dec",
        "num_layers": 50,
        "dict_init": 0.1,
    }


@config_ingredient.named_config
def dictionarylearning_g_aelasso_T50():
    hyp = {
        "experiment_name": "dictionary_learning/g_aelasso/T50",
        "grad_method": "g_aelasso",
        "num_layers": 50,
        "dict_init": 0.1,
    }


@config_ingredient.named_config
def dictionarylearning_g_aels_T50():
    hyp = {
        "experiment_name": "dictionary_learning/g_aels/T50",
        "grad_method": "g_aels",
        "num_layers": 50,
        "dict_init": 0.1,
    }


### T 25


@config_ingredient.named_config
def dictionarylearning_g_dec_T25():
    hyp = {
        "experiment_name": "dictionary_learning/g_dec/T25",
        "grad_method": "g_dec",
        "num_layers": 25,
        "dict_init": 0.1,
    }


@config_ingredient.named_config
def dictionarylearning_g_aelasso_T25():
    hyp = {
        "experiment_name": "dictionary_learning/g_aelasso/T25",
        "grad_method": "g_aelasso",
        "num_layers": 25,
        "dict_init": 0.1,
    }


@config_ingredient.named_config
def dictionarylearning_g_aels_T25():
    hyp = {
        "experiment_name": "dictionary_learning/g_aels/T25",
        "grad_method": "g_aels",
        "num_layers": 25,
        "dict_init": 0.1,
    }


####################################
###### gradient estimation D #######
####################################

### init close A 0.02


@config_ingredient.named_config
def gradient_g_dec_initAp02():
    hyp = {
        "experiment_name": "gradient/g_dec/initAp02",
        "task": "gradient_ae",
        "grad_method": "g_dec",
        "dict_init": 0.02,
    }


@config_ingredient.named_config
def gradient_g_aelasso_initAp02():
    hyp = {
        "experiment_name": "gradient/g_aelasso/initAp02",
        "task": "gradient_ae",
        "grad_method": "g_aelasso",
        "dict_init": 0.02,
    }


@config_ingredient.named_config
def gradient_g_aels_initAp02():
    hyp = {
        "experiment_name": "gradient/g_aels/initAp02",
        "task": "gradient_ae",
        "grad_method": "g_aels",
        "dict_init": 0.02,
    }


### init close A 0.05


@config_ingredient.named_config
def gradient_g_dec_initAp05():
    hyp = {
        "experiment_name": "gradient/g_dec/initAp05",
        "task": "gradient_ae",
        "grad_method": "g_dec",
        "dict_init": 0.05,
    }


@config_ingredient.named_config
def gradient_g_aelasso_initAp05():
    hyp = {
        "experiment_name": "gradient/g_aelasso/initAp05",
        "task": "gradient_ae",
        "grad_method": "g_aelasso",
        "dict_init": 0.05,
    }


@config_ingredient.named_config
def gradient_g_aels_initAp05():
    hyp = {
        "experiment_name": "gradient/g_aels/initAp05",
        "task": "gradient_ae",
        "grad_method": "g_aels",
        "dict_init": 0.05,
    }


### init close A 0.1


@config_ingredient.named_config
def gradient_g_dec_initAp1():
    hyp = {
        "experiment_name": "gradient/g_dec/initAp1",
        "task": "gradient_ae",
        "grad_method": "g_dec",
        "dict_init": 0.1,
    }


@config_ingredient.named_config
def gradient_g_aelasso_initAp1():
    hyp = {
        "experiment_name": "gradient/g_aelasso/initAp1",
        "task": "gradient_ae",
        "grad_method": "g_aelasso",
        "dict_init": 0.1,
    }


@config_ingredient.named_config
def gradient_g_aels_initAp1():
    hyp = {
        "experiment_name": "gradient/g_aels/initAp1",
        "task": "gradient_ae",
        "grad_method": "g_aels",
        "dict_init": 0.1,
    }
