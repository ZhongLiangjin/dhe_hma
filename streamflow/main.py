import time
from pathlib import Path
import random
import json
from collections import defaultdict
import numpy as np
from torch.backends import cudnn
from utils import train, loss, util_fn, eval
import torch
import logging
import os
from utils.model import DPLGlacierModel, DPLRainfallRunoffModel
from utils.dataloader import RainfallRunoffLoader
from utils.params_range import params_range

config = defaultdict()
# arguments for dataset
config['data'] = {'glac_forc_dir': '../../data/forcing/glacier',
                  'bsn_forc_dir': '../../data/forcing/basin',
                  'glac_attr_dir': '../../data/attrs/glacier',
                  'bsn_riv_attr_dir': '../../data/attrs/bsn_riv',
                  'periods': {'train': [['1955-1-1', '1974-12-31'], ['1991-1-1', '2009-12-31']],
                              'valid': [['1975-1-1', '1982-12-31'], ['2010-1-1', '2014-12-31']],
                              'test': [['1983-1-1', '1990-12-31'], ['2015-1-1', '2019-12-31']]},
                  'sim_bsn_head': 'all',
                  # 'glac_init_sdep_path': '../../data/init/glac_sdep_t0.txt',  # initial snow depth path
                  'glac_init_sdep_path': None,  # initial snow depth path
                  # 'bsn_init_sdep_path': '../../data/init/bsn_sdep_t0.txt',  # initial snow depth path
                  'bsn_init_sdep_path': None,  # initial snow depth path
                  'pretrain_glac_model_path': '../../data/pretrain',
                  # snow cover fraction on glacier
                  'seq_len': 1096,  # the length of sequence, unit: days
                  'win_sz': 1096,  # the window size to generate the sequence for training, unit: days
                  'spin_up_len': 730,  # the length of spin-up period, unit: days
                  'padding': -9999,  # padding value for river routing order
                  'seq_len_eval': 10,  # the length of sequence for evaluation, unit: years
                  }
# arguments for model training
config['train'] = {'metric': 'MSE',  # the metric to calculate loss, NSE or KGE
                   'patience': 15,  # the patience of early stopping
                   'clip': 2,  # the gradient clip
                   'epochs': 150,  # the number of epochs
                   'gpu': False,  # whether to use gpu
                   'seed': 19, # 42, 19, 36, 27, 70
                   'glac_loss': {
                       'weight': {'glac_area': 1, 'glac_vol': 0, 'glac_dvol': 1, 'glac_tvol': 1, 'glac_sdep': 0},  # weight of each loss
                       'snow_scale': 'monthly',  # daily or monthly, the scale to calculate loss for  snow depth
                       'retrain': False,  # whether to retrain the glacial model
                       'lr': 0.001,  # learning rate for retraining
                       'glac_area_path': '../../data/val_data/glacier/glac_area_basin.txt',
                       # RGI-6 glacier area and observed date
                       'glac_vol_path': '../../data/val_data/glacier/glac_vol_basin.txt',
                       # glacier volume from Millan et al. (2022)
                       'glac_dvol_path': '../../data/val_data/glacier/glac_dvol_basin_WGMS.txt',
                       'glac_sdep_path': '../../data/val_data/glacier/glac_sdep_basin.txt',  # snow depth on glacier
                   },
                   'rr_loss': {
                       'weight': {'bsn_Q': 1, 'bsn_LAI': 1, 'bsn_sdep': 1},
                       # weight of loss
                       'q_daily_weight': 0.8,  # the weight of daily discharge loss
                       'only_gs_LAI': False,  # whether to only use growing season LAI to calculate LAI loss
                       'snow_scale': 'monthly',  # daily or monthly, the scale to calculate loss for  snow depth
                       'lr': 0.005,  # the learning rate
                       'bsn_Q_path': '../../data/val_data/basin/streamflow.xlsx',  # observed discharge
                       'bsn_sdep_path': '../../data/val_data/basin/bsn_sdep.txt',  # observed snow depth
                       # observed snow cover fraction
                       'bsn_LAI_path': '../../data/val_data/basin/GIMMS_LAI.txt',  # observed leaf area index
                       # 'held_out_gauges': ['Maqu', 'Yajiang', 'Daofu', 'Nugesha', 'Gilgit', 'Shatial'],
                       'held_out_gauges': []
                   },
                   'glac_rr_weight': [0.2, 0.8]}
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
                   'glac_model': {
                       'snow': {
                           'sinusoidal_ddf': True,
                           # whether to use sinusoidal equation to calculate degree-day factor
                           'snow2ice': True,  # whether to consider snow transferring to ice
                           'snow_slm': False,  # whether to consider snow sublimation
                           'swe2sd': True,  # whether to calculate snow depth from snow water equivalent
                           'swe2sd_nn': False,
                           # whether to use neural network to calculate snow depth from snow water equivalent
                           'swe2sd_nn_hidden': [128, 32, 8],  # the hidden size of neural network for snow depth
                           # the hidden size of neural network for snow cover fraction
                           's_dep_t0': None,  # the initial snow depth
                       },
                       'glacier': {
                           'sinusoidal_ddf': True,
                           # whether to use sinusoidal equation to calculate degree-day factor
                           'glacier_slm': False,  # whether to consider glacier sublimation
                           'glacier_shift': False,  # whether to consider glacier shift
                           'glacier_shift_nn': False,  # whether to use neural network to calculate glacier shift
                           'glacier_shift_nn_hidden': [128, 32, 8],
                           # the hidden size of neural network for glacier shift
                           'update_freq': 'y'  # 'd', 'sm', 'm' or 'y' for daily, semi-monthly, monthly or yearly
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
                           'params_range': params_range['glac_params']},
                       },

                   'rr_model': {
                       'snow': {
                           'sinusoidal_ddf': True,
                           # whether to use sinusoidal equation to calculate degree-day factor
                           'snow2ice': False,  # whether to consider snow transferring to ice
                           'slm': False,  # whether to consider snow sublimation
                           'swe2sd': True,  # whether to calculate snow depth from snow water equivalent
                           'swe2sd_nn': True,
                           # whether to use neural network to calculate snow depth from snow water equivalent
                           'swe2sd_nn_hidden': [256, 64, 8],  # the hidden size of neural network for snow depth
                           's_dep_t0': None,  # the initial snow depth
                       },
                       'soil': {
                           'freeze_thaw_nn': False,  # whether to use neural network (True) or dynamic parameters (False)
                           'freeze_thaw_nn_type': 'lstm',  # the type of neural network, mlp or lstm
                           'freeze_thaw_nn_hidden': 128,  # the hidden size of neural network for snow depth
                       },
                       'veg': {
                           'LAI_t0': '../../data/init/GIMMS_LAI_t0.txt',  # the initial leaf area index
                           'init_LAI_max': True, # whether to use the maximum leaf area index
                           'init_LAI_min': True, # whether to use the minimum leaf area index
                           'alloc_func': 'linear',  # the function to calculate carbon allocation, linear or exponential
                       },
                       # if streamflow is not required for calculating loss, set False to save time
                       'cal_riv_rout': True if config['train']['rr_loss']['weight']['bsn_Q'] > 0 else False,
                       'nn_params': {
                           'static': {
                               'type': 'mlp',  # the type of neural network conv_mlp, mlp, or lstm_mlp
                               'hidden_fc': [512, 128, 64],  # the hidden size of mlp
                               'hidden_lstm': 128,  # the hidden size of lstm
                               'out_lstm': 64,  # the output size of lstm
                               'in_length': config['data']['spin_up_len'],  # the input length for conv 1d
                               'n_conv_kernel': [10, 5, 1],  # the number of convolutional kernel
                               'conv_kernel_size': [7, 5, 3],  # the size of convolutional kernel
                               'stride': [1, 1, 1],  # the stride of convolutional kernel
                               'pool_kernel_size': [3, 2, 1],  # the size of pooling kernel
                           },
                           'dynamic': {
                               'type': 'lstm',  # the type of neural network
                               'hidden_lstm': 256,  # the hidden size of lstm
                               'out_lstm': 64,  # the output size of lstm
                               'params': ['param_freCoef_rr', 'param_F_rr', 'param_Ksg_rr']
                           },
                           'riv_rout': {
                               'type': 'mlp',  # the type of neural network conv_mlp, mlp, or lstm_mlp
                               'hidden_fc': [128, 32],  # the hidden size of mlp
                           },
                           'params_range': params_range['rr_params']
                       }
                   }
                   }
config['model']['riv_rout'] = True if config['train']['rr_loss']['weight']['bsn_Q'] > 0 else False
# configure for output directory
now = time.strftime('%m%d-%H%M', time.localtime())
rr_w_loss = config['train']['rr_loss']['weight']
glac_retrain = config['train']['glac_loss']['retrain']
seq_len = config['data']['seq_len']
sta_net_type = config['model']['rr_model']['nn_params']['static']['type']
sim_bsn_head = '_'.join(str(v) for v in config['data']['sim_bsn_head']) if isinstance(config['data']['sim_bsn_head'], list) else config['data']['sim_bsn_head']
metric = config['train']['metric']
config['out'] = (f"./checkpoints/bsn_{sim_bsn_head}_seed_{seed}_wLoss_{'_'.join(str(v) for v in rr_w_loss.values())}_"
                 f"metric_{metric}_glacRetrain_{glac_retrain}_t_{now}")
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
loader = RainfallRunoffLoader(bsn_forc_dir=config['data']['bsn_forc_dir'],
                              bsn_riv_attr_dir=config['data']['bsn_riv_attr_dir'],
                              glac_forc_dir=config['data']['glac_forc_dir'],
                              glac_attr_dir=config['data']['glac_attr_dir'],
                              glac_sim_path=config['data']['pretrain_glac_model_path'],
                              periods=config['data']['periods'],
                              seq_len=config['data']['seq_len'],
                              spin_up_len=config['data']['spin_up_len'],
                              win_sz=config['data']['win_sz'],
                              seq_len_eval=config['data']['seq_len_eval'],
                              padding=config['data']['padding'],
                              device=device, logger=True,
                              sim_bsn_head=config['data']['sim_bsn_head'])
# get river routing configuration
config['model']['rr_model']['riv_rout'] = {'padding': config['data']['padding'],
                                           'bsn_total_area': loader.bsn_total_area,
                                           'riv_up_bsn_idx': loader.riv_up_bsn_idx,
                                           'rout_order': loader.rout_order_idx}

# get initial snow depth
if config['data']['glac_init_sdep_path'] is not None:
    glac_sdep_t0 = loader.glac_loader.cal_init_snow_depth(s_dep_path=config['data']['glac_init_sdep_path'],
                                                          device=device)
else:
    glac_sdep_t0 = None
config['model']['glac_model']['snow']['s_dep_t0'] = glac_sdep_t0
if config['data']['bsn_init_sdep_path'] is not None:
    bsn_sdep_t0 = loader.cal_init_snow_depth(s_dep_path=config['data']['bsn_init_sdep_path'], device=device)
else:
    bsn_sdep_t0 = None

# configure the glacier model
config['model']['glac_model']['glacier']['slope'] = torch.tensor(loader.glac_loader.glac_band_slope, device=device)
glac_model = DPLGlacierModel(bsn_band_ids_dict=loader.glac_loader.glac_bsn_band_ids_dict,
                             n_attrs=len(loader.glac_loader.attr_vars), n_forc=len(loader.glac_loader.forc_vars) - 1,
                             snow_config=config['model']['glac_model']['snow'],
                             glac_config=config['model']['glac_model']['glacier'],
                             mul_comp_config=config['model']['mul_comp'],
                             nn_params=config['model']['glac_model']['nn_params'], dropout=config['model']['dropout'],
                             device=device)
glac_model = glac_model.to(device)
glac_model.load_state_dict(torch.load(config['data']['pretrain_glac_model_path'] + '/model.pt', weights_only=True))
logging.info(f"Load the pretrained glacier model from {config['data']['pretrain_glac_model_path']}")
glac_loss_config = config['train']['glac_loss']
glac_loss_config['bsn_code'] = loader.glac_loader.glac_bsn_codes
glac_loss_config['band_code'] = loader.glac_loader.glac_band_codes
if config['train']['glac_loss']['retrain']:
    glac_optimizer = torch.optim.Adam(glac_model.parameters(), lr=glac_loss_config['lr'])
    glac_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(glac_optimizer, factor=0.5, patience=5)
else:
    for param in glac_model.parameters():
        param.requires_grad = False
    glac_optimizer, glac_scheduler = None, None

# configure the rainfall runoff model
rr_config = config['model']['rr_model']
if rr_config['veg']['LAI_t0'] is not None:
    lai_t0, lai_min, lai_max = loader.cal_init_LAI(LAI_path=config['model']['rr_model']['veg']['LAI_t0'],
                                                            init_LAI_min = config['model']['rr_model']['veg']['init_LAI_min'],
                                                            init_LAI_max = config['model']['rr_model']['veg']['init_LAI_max'],
                                                            device=device)
    config['model']['rr_model']['veg']['LAI_t0'] = lai_t0
    config['model']['rr_model']['veg']['LAI_min'] = lai_min
    config['model']['rr_model']['veg']['LAI_max'] = lai_max

rr_model = DPLRainfallRunoffModel(n_forc=len(loader.forc_vars) - 1,
                                  n_bsn_attrs=len(loader.bsn_attr_vars),
                                  n_riv_attrs=len(loader.riv_attr_vars),
                                  rr_config=config['model']['rr_model'],
                                  mul_comp_config=config['model']['mul_comp'],
                                  dropout=config['model']['dropout'],
                                  glac_bsn_idx=loader.glac_bsn_idx,
                                  device=device)
rr_model = rr_model.to(device)
# configure the loss function, optimizer and scheduler
rr_loss_config = config['train']['rr_loss']
rr_loss_config['bsn_code'] = loader.bsn_codes
rr_optimizer = torch.optim.Adam(rr_model.parameters(), lr=rr_loss_config['lr'], betas=(0.9, 0.999))
rr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(rr_optimizer, factor=0.5, patience=5)

loss_fn = loss.LossFn(glac_loss_config=glac_loss_config, rr_loss_config=rr_loss_config,
                      glac_rr_weight=config['train']['glac_rr_weight'], metric=config['train']['metric'])

early_stop = util_fn.EarlyStopping(save_path=os.path.join(config['out'], 'model.pt'),
                                   patience=config['train']['patience'], delta=0.0002)


train.train_model(glac_model=glac_model, glac_optimizer=glac_optimizer, glac_scheduler=glac_scheduler,
                  rr_model=rr_model, rr_optimizer=rr_optimizer, rr_scheduler=rr_scheduler,
                  loader=loader, loss_fn=loss_fn, early_stop=early_stop, config=config)

# run the best model to get simulation results for evaluation
loader = RainfallRunoffLoader(bsn_forc_dir=config['data']['bsn_forc_dir'],
                              bsn_riv_attr_dir=config['data']['bsn_riv_attr_dir'],
                              glac_forc_dir=config['data']['glac_forc_dir'],
                              glac_attr_dir=config['data']['glac_attr_dir'],
                              glac_sim_path=config['data']['pretrain_glac_model_path'],
                              periods=config['data']['periods'],
                              seq_len=config['data']['seq_len'],
                              spin_up_len=config['data']['spin_up_len'],
                              win_sz=config['data']['win_sz'],
                              seq_len_eval=config['data']['seq_len_eval'],
                              padding=config['data']['padding'],
                              device=device, logger=True, mode='eval',
                              sim_bsn_head=config['data']['sim_bsn_head'])
eval.SaveEval(loader_all=loader.loader_all, glac_model=glac_model, rr_model=rr_model, config=config,
              glac_loss_config=glac_loss_config, rr_loss_config=rr_loss_config,
              eval_vars=['glac_area', 'glac_tvol', 'glac_dvol', 'bsn_Q', 'bsn_sdep', 'bsn_LAI'])