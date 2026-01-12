import argparse
import time
from pathlib import Path
import random
from typing import Union
import json
from collections import defaultdict
import numpy as np
from torch.backends import cudnn
from utils import train, loss, util_fn, eval
import torch
import logging
import os
from utils.model import DPLGlacierModel
from utils.dataloader import Loader
from utils.params_range import params_range

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=Union[str, None], default=None,
                    help='the path of the model configure file')
args = parser.parse_args()

# prepare configures
if args.config_file is not None:
    with open(args.config_file, 'r') as f:
        config = json.load(f)
else:
    config = defaultdict()
    # arguments for dataset
    config['data'] = {'forcing_dir': '../../data/forcing/glacier',
                      'attr_dir': '../../data/attrs/glacier',
                      'periods': [['1955-1-1', '1999-12-31'], ['2000-1-1', '2009-12-31'],
                                  ['2010-1-1', '2019-12-31']],
                      'init_s_depth_path': None,  # initial snow depth path
                      'acr_path':'../../data/init/glac_acr.json',  # the path of glacier area change rate
                      'g_area_path': '../../data/val_data/glacier/glac_area_basin.txt',  # RGI-6 glacier area and observed date
                      'gvol_path': '../../data/val_data/glacier/glac_vol_basin.txt',  # glacier volume from Millan et al. (2022)
                      'd_gvol_path': '../../data/val_data/glacier/glac_dvol_basin_WGMS.txt',
                      # glacier volume change from Hugonnet et al. (2021)
                      's_depth_path': '../../data/val_data/glacier/glac_sdep_basin.txt',  # snow depth on glacier
                      'seq_len': 1096, # the length of sequence
                      'win_sz': 1096,  # the window size to generate the sequence
                      'spin_up_len': 365,  # the length of spin-up period
                      }
    # arguments for model training
    config['train'] = {'glac_w_loss': {'g_area': 1, 'g_vol': 0, 'd_gvol': 1, 't_gvol': 1, 's_depth': 0},  # weight of each loss
                       'sdep_scale': 'monthly', # daily or monthly, the scale to calculate loss for  snow depth
                       'metric': 'MSE', # the metric to calculate loss, NSE or KGE
                       'patience': 10,  # the patience of early stopping
                       'lr': 0.001,  # the learning rate
                       'clip': 5,   # the gradient clip
                       'epochs': 200, # the number of epochs
                       'gpu': False,  # whether to use gpu
                       'pretrained_model': None,  # the path of pretrained model
                       'gwe_swe_t0': True, # whether to use glacier water equivalent at t0 to constrain glacier mass balance
                       'seed': 19} # 42, 19, 36, 27, 70
    device = torch.device('cuda:0' if config['train']['gpu'] and torch.cuda.is_available() else 'cpu')
    # fix the random seed
    seed = config['train']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # arguments for model
    config['model'] = {'dropout': 0.5,
                       'mul_comp': {'n_mul_comp': 1, 'weights_method': 'mean'},  # multi-components
                       'snow': {
                           'sinusoidal_ddf': True,  # whether to use sinusoidal equation to calculate degree-day factor
                           'snow2ice': True,  # whether to consider snow transferring to ice
                           'snow_slm': False,  # whether to consider snow sublimation
                           'swe2sd': True,  # whether to calculate snow depth from snow water equivalent
                           'swe2sd_nn': False,
                           # whether to use neural network to calculate snow depth from snow water equivalent
                           'swe2sd_nn_hidden': [128, 32, 8],  # the hidden size of neural network for snow depth
                       },
                       'glacier': {
                           'sinusoidal_ddf': True,  # whether to use sinusoidal equation to calculate degree-day factor
                           'glacier_slm': False,  # whether to consider glacier sublimation
                           'glacier_shift': False,  # whether to consider glacier shift
                           'glacier_shift_nn': False,  # whether to use neural network to calculate glacier shift
                           'glacier_shift_nn_hidden': [128, 32, 8],  # the hidden size of neural network for glacier shift
                           'update_freq': 'y', # 'd', 'sm', 'm' or 'y' for daily, semi-monthly, monthly or yearly
                       },
                       'nn_params': {
                           'type': 'mlp',  # the type of neural network conv_mlp, mlp, or lstm_mlp
                           'hidden_fc': 256,  # the hidden size of mlp
                           'hidden_lstm': 128,  # the hidden size of lstm
                           'out_lstm': 64,  # the output size of lstm
                           'in_length': config['data']['spin_up_len'],  # the input length for conv 1d
                           'n_conv_kernel': [10, 5, 1],  # the number of convolutional kernel
                           'conv_kernel_size': [7, 5, 3],  # the size of convolutional kernel
                           'stride': [1, 1, 1],  # the stride of convolutional kernel
                           'pool_kernel_size': [3, 2, 1],  # the size of pooling kernel
                           'params_range': params_range,
                       }}
    # configure for output directory
    now = time.strftime('%m%d-%H%M', time.localtime())
    w_loss = config['train']['glac_w_loss']
    seq_len, win_sz = config['data']['seq_len'], config['data']['win_sz']
    metric = config['train']['metric']
    update_freq = config['model']['glacier']['update_freq']
    config['out'] = (f"./checkpoints/seed_{seed}_seqL_{seq_len}_winSz_{win_sz}_freq_{update_freq}_loss_{metric}_wLoss_"
                     f"{'_'.join(str(v) for v in w_loss.values())}_t_{now}")
    Path(config['out']).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config['out'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)  # type: ignore

    # log file
    log_file = os.path.join(config['out'], 'log.txt')
    if os.path.exists(log_file):
        os.remove(log_file)
    util_fn.setup_logger(log_file)
    logging.info(f'{device} is used in training.')
    logging.info(f'The output path is {config["out"]}')

    # get the loader
    loader = Loader(forcing_dir=config['data']['forcing_dir'],
                    attr_dir=config['data']['attr_dir'],
                    acr_path=config['data']['acr_path'],
                    periods=config['data']['periods'],
                    spin_up_len=config['data']['spin_up_len'],
                    seq_len=config['data']['seq_len'],
                    win_sz=config['data']['win_sz'],
                    device=device,
                    logger=True)
    # get initial snow depth
    s_dep_t0 = loader.cal_init_snow_depth(s_dep_path=config['data']['init_s_depth_path'], device=device) if (
            config['data']['init_s_depth_path'] is not None) else None
    # get the model
    config['model']['glacier']['slope'] = torch.tensor(loader.band_slope, device=device)
    model = DPLGlacierModel(bsn_band_ids_dict=loader.bsn_band_ids_dict, n_attrs=len(loader.attr_vars),
                            n_forc=len(loader.forc_vars) - 1, snow_config=config['model']['snow'],
                            gla_config=config['model']['glacier'], mul_comp_config=config['model']['mul_comp'],
                            nn_params=config['model']['nn_params'], glac_area_band_t0=loader.g_area_band_t0,
                            snow_depth_t0=s_dep_t0, dropout=config['model']['dropout'], device=device)
    model = model.to(device)
    if config['train']['pretrained_model'] is not None:
        model.load_state_dict(torch.load(config['train']['pretrained_model'], weights_only=True))
        logging.info(f'Load the pretrained model from {config["train"]["pretrained_model"]}')

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    early_stop = util_fn.EarlyStopping(save_path=os.path.join(config['out'], 'model.pt'),
                                     patience=config['train']['patience'], delta=0.0002)
    loss_fn = loss.LossFn(g_area_path=config['data']['g_area_path'], gvol_path=config['data']['gvol_path'],
                          d_gvol_path=config['data']['d_gvol_path'], s_depth_path=config['data']['s_depth_path'],
                          w_loss=config['train']['glac_w_loss'], basin_codes=loader.bsn_codes,
                          band_codes=loader.band_codes, snow_scale=config['train']['sdep_scale'],
                          metric=config['train']['metric'])

    train.train_model(model=model, loader=loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                      early_stop=early_stop, config=config)
    eval.SaveEval(loader_all=loader.loader_all, model=model, config=config, basin_codes=loader.bsn_codes,
                  eval_vars=['g_area', 't_gvol', 'd_gvol', 'g_vol', 's_depth'])

