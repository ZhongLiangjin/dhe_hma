import pandas as pd
from collections import defaultdict
from typing import List, Union, Dict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
try:
    from torch_scatter import scatter_sum, scatter_mean
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False


class DPLRainfallRunoffModel(nn.Module):
    def __init__(self, n_forc: int, n_bsn_attrs: int, n_riv_attrs: int, rr_config: dict, glac_bsn_idx: List[int],
                 mul_comp_config: dict, dropout: float = 0.5, device: Union[str, torch.device] = 'cpu'):
        super(DPLRainfallRunoffModel, self).__init__()

        self.n_forc = n_forc
        self.n_bsn_attrs = n_bsn_attrs
        self.n_riv_attrs = n_riv_attrs
        self.device = device

        # define multiple components
        self.n_mul_comp = mul_comp_config['n_mul_comp']
        self.mul_comp_weights_method = mul_comp_config['weights_method']

        # define the snow and glacier
        self.rain_runoff_model = RainfallRunoffDynCell(rr_config=rr_config, n_mul_comp=self.n_mul_comp,
                                                       glac_bsn_idx=glac_bsn_idx,
                                                       params_range=rr_config['nn_params']['params_range'],
                                                       n_bsn_attrs=n_bsn_attrs, device=device, dropout=dropout)
        self.cal_riv_rout = rr_config['cal_riv_rout']
        # determine the number of dynamic, static, and river routing parameters
        self.n_dynamic_params = len(self.rain_runoff_model.dynamic_params)
        self.n_riv_rout_params = len(self.rain_runoff_model.riv_rout_params)
        self.n_static_params = len(self.rain_runoff_model.static_params)
        self.n_hillslope_params = len(self.rain_runoff_model.hillslope_rout_params)
        # determine the number of parameters to be learned
        assert self.mul_comp_weights_method in ['dPL', 'mean'], 'Invalid method to determine the number of parameters'
        if self.mul_comp_weights_method == 'dPL' and self.n_mul_comp > 1:  # learn the weights for multiple components
            # hillslope parameters are identical for all components
            self.n_static_params = self.n_static_params * self.n_mul_comp + self.n_mul_comp + self.n_hillslope_params
        else:  # assign the same weights for multiple components
            self.n_static_params = self.n_static_params * self.n_mul_comp + self.n_hillslope_params
        self.n_dynamic_params = self.n_dynamic_params * self.n_mul_comp

        # define the static parameters learning network
        static_net_config = rr_config['nn_params']['static']
        if static_net_config['type'] == 'mlp':
            self.static_param_nn = MlpParams(in_size=n_bsn_attrs, hidden_size=static_net_config['hidden_fc'],
                                             out_size=self.n_static_params,
                                             dropout=dropout)
        elif static_net_config['type'] == 'conv_mlp':
            self.static_param_nn = ConvMlpParams(n_attrs=n_bsn_attrs, n_forc=n_forc, n_params=self.n_static_params,
                                                 in_length=static_net_config['in_length'],
                                                 hidden_size=static_net_config['hidden_fc'],
                                                 n_conv_kernel=static_net_config['n_conv_kernel'],
                                                 conv_kernel_size=static_net_config['conv_kernel_size'],
                                                 stride=static_net_config['stride'],
                                                 pool_kernel_size=static_net_config['pool_kernel_size'],
                                                 dropout=dropout)
        elif static_net_config['type'] == 'lstm_mlp':
            self.static_param_nn = LstmMlpParams(in_lstm=n_forc, hid_lstm=static_net_config['hidden_lstm'],
                                                 out_lstm=static_net_config['out_lstm'],
                                                 in_fc=static_net_config['out_lstm'] + n_bsn_attrs,
                                                 hid_fc=static_net_config['hidden_fc'],
                                                 out_fc=self.n_static_params,
                                                 dropout=dropout, device=device)
        else:
            raise ValueError('Invalid neural network type for static parameters')

        # define the dynamic parameters learning network
        dynamic_net_config = rr_config['nn_params']['dynamic']
        if dynamic_net_config['type'] == 'lstm':
            self.dynamic_param_nn = LstmParams(in_lstm=n_forc + n_bsn_attrs, hid_lstm=dynamic_net_config['hidden_lstm'],
                                               out_lstm=self.n_dynamic_params, dropout=dropout, device=device)
        else:
            raise ValueError('Invalid neural network type for dynamic parameters')

        # define the river routing parameters learning network
        riv_rout_net_config = rr_config['nn_params']['riv_rout']
        if riv_rout_net_config['type'] == 'mlp':
            self.riv_rout_param_nn = MlpParams(in_size=n_riv_attrs, hidden_size=riv_rout_net_config['hidden_fc'],
                                               out_size=self.n_riv_rout_params, dropout=dropout)
        else:
            raise ValueError('Invalid neural network type for river routing parameters')

    def forward(self, forc: torch.Tensor, forc_norm: torch.Tensor, bsn_attrs_norm: torch.Tensor,
                riv_attrs_norm: torch.Tensor, glac_sim_bsn: Dict[str, torch.Tensor], spin_up_len: int,
                mode: str = 'train', hidden_state: Union[torch.Tensor, None] = None):

        assert mode in ['train', 'eval'], 'Invalid mode'
        if mode == 'train':
            out_vars = [ 'Qriver', 'Qsub', 'Qsurf', 'Etot', 'Esnow', 'Eicp', 'Esoil', 'T', 'LAI', 'sdep', 'swe']  # output variables for training
        else:
            out_vars = ['Peff', 'Qriver', 'Qsub', 'Qsurf', 'Qrain', 'Qsnow', 'Etot', 'Esnow', 'Eicp', 'Esoil', 'T', 'LAI',
                        'biom', 'swe', 'sdep', 'swc_liq', 'swc_ice', 'param']
        # learn static parameters using neural network
        bsn_attrs_exp = bsn_attrs_norm.unsqueeze(1).expand(-1, forc_norm.size(1), -1)
        x_params_nn = torch.cat((forc_norm, bsn_attrs_exp), dim=-1)
        spin_up_len = 1 if spin_up_len < 1 else spin_up_len  # ensure spin-up length is at least 1
        static_params = self.static_param_nn(x=x_params_nn[:, :spin_up_len, :], attrs_idx=self.n_forc + 1)
        # get then hillslope routing parameters first
        hillslope_params_sigmoid = torch.sigmoid(static_params[:, -2:])
        hillslope_params_dict = {
            k: hillslope_params_sigmoid[:, i] for i, k in enumerate(self.rain_runoff_model.hillslope_rout_params)
        }
        hillslope_params_dict = self.rain_runoff_model.rescale_param_range(hillslope_params_dict)
        # reshape the rest parameters into (n_basins, n_mul_comp, n_params)
        static_params_tmp = static_params[:, :-2].reshape(static_params.size(0), self.n_mul_comp, -1)
        if self.mul_comp_weights_method == 'dPL' and self.n_mul_comp > 1:
            mul_comp_weights = torch.softmax(static_params_tmp[:, :, -1], dim=1)
        else:
            mul_comp_weights = torch.full_like(static_params_tmp[:, :, -1], 1 / self.n_mul_comp)
        static_params_sigmoid = torch.sigmoid(static_params_tmp[...,
                                              :-1] if self.mul_comp_weights_method == 'dPL' and self.n_mul_comp > 1 else static_params_tmp)
        static_params_dict = {
            k: static_params_sigmoid[:, :, i] for i, k in enumerate(self.rain_runoff_model.static_params)
        }
        static_params_dict = self.rain_runoff_model.rescale_param_range(static_params_dict)

        # learn dynamic parameters using neural network
        dynamic_params = self.dynamic_param_nn(x=x_params_nn, attrs_idx=self.n_forc + 1, mode=mode)
        # reshape the parameters into (n_basins, n_time, n_mul_comp, n_params)
        dynamic_params_tmp = dynamic_params.reshape(x_params_nn.size(0), x_params_nn.size(1), self.n_mul_comp, -1)
        # store other dynamic parameters in a dictionary and rescale the parameters to the physical range
        dynamic_params_sigmoid = torch.sigmoid(dynamic_params_tmp)
        dynamic_params_dict = {
            k: dynamic_params_sigmoid[:, :, :, i] for i, k in enumerate(self.rain_runoff_model.dynamic_params)
        }
        dynamic_params_dict = self.rain_runoff_model.rescale_param_range(dynamic_params_dict)
        # learn river routing parameters using neural network
        riv_rout_params = self.riv_rout_param_nn(riv_attrs_norm, attrs_idx=0)
        # store other river routing parameters in a dictionary and rescale the parameters to the physical range
        riv_rout_sigmoid = torch.sigmoid(riv_rout_params)
        riv_rout_params_dict = {
            k: riv_rout_sigmoid[:, i] for i, k in enumerate(self.rain_runoff_model.riv_rout_params)
        }
        riv_rout_params_dict = self.rain_runoff_model.rescale_param_range(riv_rout_params_dict)

        # merge the parameters
        params_dict_tmp = {**static_params_dict, **dynamic_params_dict, **hillslope_params_dict, **riv_rout_params_dict}
        params_dict = {k: (params_dict_tmp[k] if k in params_dict_tmp.keys() else None)
                       for k in self.rain_runoff_model.param_names}
        # # unpack parameters
        # for i, param in enumerate(self.rain_runoff_model.non_none_params):
        #     self.rain_runoff_model.params[param] = params_dict[param]

        # initialize the storage
        if hidden_state is None:
            q_t0, s_t0, ant_tas = self.rain_runoff_model.init_model(n_basins=bsn_attrs_norm.size(0),
                                                                    n_rivers=riv_attrs_norm.size(0),
                                                                    params_dict=params_dict)
        else:
            q_t0, s_t0, ant_tas = hidden_state
        if hasattr(self.rain_runoff_model, 'freeze_thaw_hidden'):  # initialize the hidden state for freeze-thaw nn
            self.rain_runoff_model.freeze_thaw_hidden = None
        # Get glacier melt as additional input to the rainfall-runoff model
        glac_melt = self.rain_runoff_model.cal_glac_input(glac_sim=glac_sim_bsn)
        # for time loop to simulate the streamflow
        output = defaultdict(list)
        for t_step in range(x_params_nn.size(1)):
            # get the initial states for the current time step
            params_dict_step = {k: (v[:, t_step, :] if k in self.rain_runoff_model.dynamic_params else v) for k, v in
                                params_dict.items()}
            tas = forc[:, t_step, :].unsqueeze(1).expand(-1, self.n_mul_comp, -1)[:, :, 1]
            ant_tas = torch.cat((ant_tas[:, :, -(ant_tas.size(-1) - 1):].clone(), tas.unsqueeze(-1)), dim=-1)
            out_t = self.rain_runoff_model(forc=forc[:, t_step, :],
                                           glac_melt=glac_melt[:, t_step],
                                           forc_norm=forc_norm[:, t_step, :],
                                           attrs_norm=bsn_attrs_norm,
                                           s_t0=s_t0,
                                           ant_tas=ant_tas,
                                           params_dict=params_dict_step)
            # update the initial states for the next time step
            Sw_tt, Si_tt, Ssl_tt, Sss_tt, Bg_tt, Tacc_tt = out_t['swe'], out_t['iwc'], out_t['swc_liq'], out_t['swc_ice'], out_t[
                'biom'], out_t['Tacc']
            s_t0 = torch.stack((Sw_tt, Ssl_tt, Sss_tt, Si_tt, Tacc_tt, Bg_tt), dim=-1)
            # store the multi-component wighted variables
            for k, v in out_t.items():
                if k in out_vars:
                    output[k].append((v * mul_comp_weights).sum(dim=1))
        output = {k: torch.stack(v, dim=-1) for k, v in output.items()}  # [n_basin, n_time]

        # calculate the basin-wide variables considering glacier area and non-glacier area
        output = self.rain_runoff_model.cal_glac_rr_weighted_state(glac_sim=glac_sim_bsn, rr_sim=output)

        # calculate hillslope routing
        output['Qhill'] = self.rain_runoff_model.cal_hillslope_routing(Q=output['Qsurf'], tmax=10,
                                                                    param_A=params_dict['param_A_rr'],
                                                                    param_B=params_dict['param_B_rr'])

        # calculate channel routing
        if self.cal_riv_rout:
            output['Qriver'], q_t0 = self.rain_runoff_model.cal_river_routing(Qh=output['Qsub'] + output['Qhill'], R0=q_t0,
                                                                          param_X=params_dict['param_X_rr'],
                                                                          param_K=params_dict['param_K_rr'])
        output['hidden_state'] = (q_t0.clone().detach(), s_t0.clone().detach(), ant_tas.clone().detach())

        # save the parameters
        if 'param' in out_vars:
            for k, v in params_dict.items():
                if v is not None:
                    output[k] = v  # type: ignore

        return output


class DPLGlacierModel(nn.Module):
    def __init__(self, bsn_band_ids_dict: dict, n_attrs: int, n_forc: int, snow_config: dict, glac_config: dict,
                 mul_comp_config: dict, nn_params: dict, dropout: float = 0.5,
                 device: Union[str, torch.device] = 'cpu'):
        super(DPLGlacierModel, self).__init__()
        # define multiple components
        self.n_mul_comp = mul_comp_config['n_mul_comp']
        self.mul_comp_weights_method = mul_comp_config['weights_method']
        self.n_forc = n_forc

        # define the snow and glacier
        self.glacier_model = GlacierDynCell(bsn_band_ids_dict=bsn_band_ids_dict, n_attrs=n_attrs,
                                            snow_config=snow_config, glac_config=glac_config,
                                            params_range=nn_params['params_range'],
                                            n_mul_comp=mul_comp_config['n_mul_comp'], dropout=dropout, device=device)
        # determine the number of parameters to be learned
        assert self.mul_comp_weights_method in ['dPL', 'mean'], 'Invalid method to determine the number of parameters'
        if self.mul_comp_weights_method == 'dPL' and self.n_mul_comp > 1:  # learn the weights for multiple components
            self.n_params = len(self.glacier_model.non_none_params) * self.n_mul_comp + self.n_mul_comp
        else:  # assign the same weights for multiple components
            self.n_params = len(self.glacier_model.non_none_params) * self.n_mul_comp

        # define the neural network to learn the static parameters
        if nn_params['type'] == 'mlp':
            self.param_nn = MlpParams(in_size=n_attrs, hidden_size=nn_params['hidden_fc'], out_size=self.n_params,
                                      dropout=dropout)
        elif nn_params['type'] == 'conv_mlp':
            self.param_nn = ConvMlpParams(n_attrs=n_attrs, n_forc=n_forc, n_params=self.n_params,
                                          in_length=nn_params['in_length'], hidden_size=nn_params['hidden_fc'],
                                          n_conv_kernel=nn_params['n_conv_kernel'],
                                          conv_kernel_size=nn_params['conv_kernel_size'],
                                          stride=nn_params['stride'], pool_kernel_size=nn_params['pool_kernel_size'],
                                          dropout=dropout)
        elif nn_params['type'] == 'lstm_mlp':
            self.param_nn = LstmMlpParams(in_lstm=n_forc, hid_lstm=nn_params['hidden_lstm'],
                                          out_lstm=nn_params['out_lstm'], in_fc=nn_params['out_lstm'] + n_attrs,
                                          hid_fc=nn_params['hidden_fc'], out_fc=self.n_params,
                                          dropout=dropout, device=device)
        else:
            raise ValueError('Invalid neural network type')

    def forward(self, forc: torch.Tensor, forc_norm: torch.Tensor, attrs_norm: torch.Tensor, ts: pd.DatetimeIndex,
                spin_up_len: int, glac_area_t0: Union[torch.Tensor, None] = None, mode: str = 'train'):
        # store the output variables
        if mode == 'train':
            out_vars = {'band': ['s_we', 'g_we', 'g_area'],
                        'basin': ['s_depth', 's_cf', 's_vol', 's_pr', 's_melt', 's_slm', 'g_vol', 'g_area', 'g_melt',
                                  'g_slm']}
        else:
            out_vars = {'band': ['s_we', 'g_we', 'g_area', 'param'],
                        'basin': ['s_pr', 's_melt', 's_slm', 's_depth', 's_cf', 's_vol', 'g_melt', 'g_slm', 'g_vol',
                                  'g_area', 'param']}

        # learn static parameters using neural network
        x_params_nn = torch.cat((forc_norm, attrs_norm.unsqueeze(1).expand(-1, forc_norm.size(1), -1)), dim=-1)
        spin_up_len = 1 if spin_up_len < 1 else spin_up_len  # ensure spin-up length is at least 1
        params = self.param_nn(x_params_nn[:, :spin_up_len, :], attrs_idx=self.n_forc + 1)
        # reshape the parameters into (n_bands, n_mul_comp, n_params)
        params_tmp = params.reshape(params.size(0), self.n_mul_comp, -1)
        if self.mul_comp_weights_method == 'dPL' and self.n_mul_comp > 1:
            mul_comp_weights = F.softmax(params_tmp[:, :, -1], dim=1)  # different weights determined by the model
        else:
            mul_comp_weights = torch.full_like(params_tmp[:, :, -1], 1 / self.n_mul_comp)  # equal weights
        # store other physical parameters in a dictionary and rescale the parameters to the physical range
        param_dict_tmp = {param_name: torch.sigmoid(params_tmp[:, :, i]) for i, param_name in
                          enumerate(self.glacier_model.non_none_params)}
        param_dict_tmp = self.glacier_model.rescale_param_range(param_dict_tmp)
        # unpack parameters
        params_dict = {param: None for param in self.glacier_model.param_names}
        for i, param in enumerate(self.glacier_model.non_none_params):
            params_dict[param] = param_dict_tmp[param]
        # transform the band parameters to the basin parameters
        for param, scale in self.glacier_model.params_scale.items():
            if scale == 'basin':
                params_dict[param] = self.glacier_model.cal_bsn_params(params_dict[param])

        # initialize the glacier area
        g_area_band_t0 = glac_area_t0.T.expand(-1, self.n_mul_comp)
        g_area_bsn_t0 = self.glacier_model.trans_var_band2bsn(var_name='g_area', var_band=g_area_band_t0)
        # initialize the glacier volume and snow water equivalent
        param_m, param_n = params_dict['param_m_glac'], params_dict['param_n_glac']
        gwe_band_t0 = self.glacier_model.init_gwe(area_bsn=g_area_bsn_t0, area_band=g_area_band_t0,
                                                  param_m=param_m, param_n=param_n)  # type: ignore
        param_Asp, param_beta = params_dict['param_Asp_glac'], params_dict['param_beta_glac']
        swe_band_t0 = self.glacier_model.init_swe(g_area_band=g_area_band_t0,
                                                  param_Asp=param_Asp, param_beta=param_beta)  # type: ignore

        # for time loop to simulate the glacier dynamics
        out_band, out_bsn = defaultdict(list), defaultdict(torch.Tensor)
        for t_step in range(x_params_nn.size(1)):
            out_band_t, out_bsn_t = self.glacier_model(forc=forc[:, t_step, :],
                                                       forc_norm=forc_norm[:, t_step, :],
                                                       attrs_norm=attrs_norm,
                                                       params_dict=params_dict,
                                                       swe_band_t0=swe_band_t0,
                                                       gve_band_t0=gwe_band_t0,
                                                       area_bsn_t0=g_area_bsn_t0,
                                                       area_band_t0=g_area_band_t0,
                                                       ts=ts[t_step])

            # update the initial states for the next time step
            swe_band_t0, g_area_band_t0, gwe_band_t0 = out_band_t['s_we'], out_band_t['g_area'], out_band_t['g_we']
            g_area_bsn_t0 = out_bsn_t['g_area']
            if torch.isnan(swe_band_t0).any() or torch.isnan(g_area_band_t0).any() or torch.isnan(gwe_band_t0).any():
                print('NaN values in the output variables')
            # store the output variables based on the mode
            for k, v in out_band_t.items():
                if k in set(out_vars['band'] + out_vars['basin']):
                    out_band[k].append((v * mul_comp_weights).sum(dim=1))

        # stack the output variables
        out_band = {k: torch.stack(v, dim=1) for k, v in out_band.items()}  # [n_band, n_time]
        # calculate the basin-wide variables
        out_bsn['g_area'] = self.glacier_model.trans_var_band2bsn(var_name='g_area', var_band=out_band['g_area'])
        for k, v in out_band.items():
            if k != 'g_area' and k in out_vars['basin']:
                out_bsn[k] = self.glacier_model.trans_var_band2bsn(var_name=k, var_band=v,
                                                                   area_band=out_band['g_area'],
                                                                   area_bsn=out_bsn['g_area'])
        # filter the output variables
        out_band = {k: v for k, v in out_band.items() if k in out_vars['band']}

        # save the parameters
        if 'param' in set(out_vars['band'] + out_vars['basin']):
            for k, v in params_dict.items():
                if v is not None:
                    if self.glacier_model.params_scale[k] == 'band':
                        out_band[k] = v  # type: ignore
                    else:
                        out_bsn[k] = v  # type: ignore

        return out_band, out_bsn


class RainfallRunoffDynCell(nn.Module):
    def __init__(self, rr_config: dict, n_mul_comp: int, glac_bsn_idx: List[int], n_bsn_attrs: int, params_range: dict,
                 device: Union[str, torch.device] = 'cpu', dropout: float = 0.5):
        super(RainfallRunoffDynCell, self).__init__()
        # get the configuration for the river routing model
        self.device = device
        riv_rout_dict = rr_config['riv_rout']
        self.pad = riv_rout_dict['padding']
        self.bsn_total_area = torch.tensor(riv_rout_dict['bsn_total_area'], device=device, dtype=torch.float32)
        self.riv_up_bsn_idx = riv_rout_dict['riv_up_bsn_idx']
        self.rout_order = riv_rout_dict['rout_order']
        self._precompute_routing_steps()
        # get the configuration for rainfall-runoff model
        self.params_range = params_range
        self.rr_config = rr_config
        self.glac_bsn_idx = glac_bsn_idx
        self.eps = 1e-6
        self.n_mul_comp = n_mul_comp
        self.LAI_t0 = rr_config['veg']['LAI_t0']
        self.s_dep_t0 = rr_config['snow']['s_dep_t0']
        if self.s_dep_t0 is not None:
            self.s_dep_t0 = self.s_dep_t0.expand(-1, self.n_mul_comp)
        if self.LAI_t0 is not None:
            self.LAI_t0 = self.LAI_t0.expand(-1, self.n_mul_comp)
        self.LAI_min = rr_config['veg']['LAI_min'].expand(-1, self.n_mul_comp) if rr_config['veg'][
                                                                                      'LAI_min'] is not None \
            else torch.zeros_like(self.LAI_t0)
        self.LAI_max = rr_config['veg']['LAI_max'].expand(-1, self.n_mul_comp) if rr_config['veg'][
                                                                                      'LAI_max'] is not None \
            else torch.full_like(self.LAI_t0, 6)

        # all parameters
        self.param_names = list(self.params_range.keys())
        # self.param_names.append('param_K_rr')
        # self.params = {param: None for param in param_names}
        self.non_none_params = self.get_not_none_params()
        self.dynamic_params = self.get_dynamic_params()
        self.riv_rout_params = ['param_X_rr', 'param_K_rr']
        self.hillslope_rout_params = ['param_A_rr', 'param_B_rr']
        self.static_params = [param for param in self.non_none_params if param not in
                              self.dynamic_params + self.riv_rout_params + self.hillslope_rout_params]

        # initialize neural networks
        snow_config = self.rr_config['snow']
        if self.rr_config['snow']['swe2sd_nn']:
            self.swe2sd_nn = MlpModules(in_features=7 + n_bsn_attrs, hidden_size=snow_config['swe2sd_nn_hidden'],
                                        out_features=1, dropout=dropout)
        if self.rr_config['soil']['freeze_thaw_nn']:
            if self.rr_config['soil']['freeze_thaw_nn_type'] == 'mlp':
                self.freeze_thaw_nn = MlpModules(in_features=8 + n_bsn_attrs, unnorm_var_num=4,
                                                 hidden_size=self.rr_config['soil']['freeze_thaw_nn_hidden'],
                                                 out_features=1, dropout=dropout)
            elif self.rr_config['soil']['freeze_thaw_nn_type'] == 'lstm':
                self.freeze_thaw_nn = LstmCellModules(in_lstm=8 + n_bsn_attrs,
                                                      hid_lstm=self.rr_config['soil']['freeze_thaw_nn_hidden'],
                                                      out_lstm=1, dropout=dropout, unnorm_var_num=4)
                self.freeze_thaw_hidden = None
            else:
                raise ValueError('Invalid neural network type for freeze-thaw process')


    def cal_snow_bucket(self, swe, forc, forc_norm, attrs_norm, param_ts, param_tm, param_ds=None, param_ds6=None,
                        param_ds12=None, param_snow2ice=None, param_Asp=None, param_beta=None):
        """
        :param swe: initial snow water equivalent.
        :param forc: forcing data including prec, tas, pet, rhu, wind, nRad, prs, and doy.
        :param forc_norm: normalized forcing data, with the shape of (n_band, n_mul_comp, 7).
        :param attrs_norm: elevation band attributes.
        :param param_ts: temperature threshold for snowfall and rainfall, °C.
        :param param_tm: temperature threshold for snow melt, °C.
        :param param_ds: degree-day factor for snow melt, mm/°C/day.
        :param param_ds6: degree-day factor for snow melt on 21 June, mm/°C/day.
        :param param_ds12: degree-day factor for snow melt on 21 December, mm/°C/day.
        :param param_snow2ice: a constant coefficient to calculate snow to ice, -.
        :param param_Asp: coefficient to calculate snow pressure, g/cm2, 0.01-0.3.
        :param param_beta: exponent coefficient to calculate snow pressure, 0.8-1.2.
        """
        # unpack forcing data
        prec, tas, rhu, wins = forc[:, :, 0], forc[:, :, 1], forc[:, :, 3], forc[:, :, 4]
        nRad, prs, doy = forc[:, :, 5], forc[:, :, 6], forc[:, :, 7]

        # partition precipitation into snow and rain
        rainfall = torch.mul(prec, (tas >= param_ts))
        snowfall = torch.mul(prec, (tas < param_ts))

        # calculate snowmelt
        if self.rr_config['snow']['sinusoidal_ddf']:
            ddf = (param_ds6 + param_ds12) / 2 + (param_ds6 - param_ds12) / 2 * torch.sin(
                2 * 3.1415926 * (doy - 81) / 365)
        else:
            ddf = param_ds
        Gs = 0.2 * nRad  # subsurface heat flux, MJ/m2/day, 20% of net radiation following GLEAM4
        lambda_fusion = 0.334  # latent heat of fusion, MJ/kg
        rho_w = 1  # density of water, g/cm3
        melt_pot = torch.clamp((nRad - Gs) * rho_w / lambda_fusion, min=0)  # potential melt, mm/day
        melt = torch.clamp(ddf * (tas - param_tm), min=torch.zeros_like(swe), max=torch.min(melt_pot, swe))

        # calculate snow transferring to ice
        if self.rr_config['snow']['snow2ice']:
            snow2ice = swe * param_snow2ice * (1 + torch.sin(2 * 3.1415926 * (doy - 81) / 365))
        else:
            snow2ice = torch.zeros_like(swe)

        # calculate snow sublimation
        if self.rr_config['snow']['slm']:
            # calculate potential evapotranspiration using Penman formula
            pet = self.cal_Penman_ET(tas=tas, prs=prs, nRad=nRad, rhu=rhu, wins=wins, melt=melt)
            E_snow = torch.clamp(pet, max=swe - melt)
        else:
            E_snow = torch.zeros_like(snowfall)

        # update snow water equivalent
        swe = swe + snowfall - melt - E_snow - snow2ice
        swe = torch.clamp(swe, min=torch.zeros_like(swe))

        # calculate snow depth
        if self.rr_config['snow']['swe2sd']:
            if self.rr_config['snow']['swe2sd_nn']:  # use neural network to calculate snow depth from snow water equivalent
                # [n_band, n_mul_comp, 1 + 6 + n_attrs]
                x = torch.cat((swe.unsqueeze(-1), forc_norm[:, :, 1:7], attrs_norm), dim=-1)
                snow_dep = F.relu(self.swe2sd_nn(x)).squeeze(-1)
            else:  # use empirical formula to calculate snow depth from snow water equivalent
                # swe = param_a * sd ** param_beta / rho_w, in which the units of swe and sd are cm.
                snow_dep = (swe / 10 / param_Asp + self.eps) ** (1 / param_beta) * 10
        else:
            snow_dep = None

        return swe, rainfall, melt, E_snow, snow2ice, snow_dep

    def cal_veg_bucket(self, rainfall, LAI, iwc, evap_snow, forc, param_Imax):
        """
        :param rainfall: precipitation falling as rain
        :param forc: forcing data including prec, tas, pet, rhu, wind, nRad, prs, and doy.
        :param iwc: interception water content
        :param LAI: leaf area index
        :param evap_snow: snow sublimation
        :param param_Imax: maximum interception storage  | Range: (0.1, 1)
        """
        pet = forc[:, :, 2]
        E_icp = torch.clamp(iwc, min=torch.zeros_like(iwc),
                            max=torch.clamp(pet - evap_snow, min=0))
        Peff = torch.clamp(rainfall + iwc - param_Imax * LAI / self.LAI_max, min=0)  # effective precipitation
        iwc = torch.clamp(rainfall + iwc - E_icp - Peff, min=torch.zeros_like(iwc),
                          max=param_Imax * LAI / self.LAI_max)

        return Peff, iwc, E_icp

    def cal_soil_bucket(self, forc, forc_norm, attrs_norm, swc_liq, swc_ice, swe, Peff, glac_melt, snow_melt, Ei, Ew, LAI,
                        ant_tas, param_Smax, param_F, param_Qmax, param_freCoef, param_Ks):
        """
        :param forc: forcing data including prec, tas, pet, rhu, wind, nRad, prs, and doy.
        :param glac_melt: glacier melt water input to the soil bucket
        :param forc_norm: normalized forcing data, with the shape of (n_band, n_mul_comp, 7).
        :param attrs_norm: basin attributes.
        :param swc_liq: Liquid water in the soil bucket.
        :param swc_ice: Solid water in the soil bucket.
        :param swe: Snow water equivalent.
        :param Peff: effective precipitation.
        :param snow_melt: Melted snow.
        :param LAI: leaf area index.
        :param Ei: Evapotranspiration for interception storage.
        :param Ew: Evapotranspiration for snow sublimation.
        :param param_Smax: Maximum storage of the catchment bucket with a range of (100, 1500)
        :param param_F: Rate of decline in flow from catchment bucket with a range of (0, 0.1)
        :param param_Qmax: Maximum subsurface flow at full bucket with a range of (10, 50)
        :param param_freCoef: A parameter determining how much soil water is freezing.
        :param param_Ks：Soil coefficient to calculate the maximum bare soil evaporation from ET0 with a range of (0.7, 1)
        """
        swc_tot = torch.clamp(swc_ice + swc_liq, max=param_Smax - self.eps)
        freeze_flag = torch.mean(ant_tas, dim=-1) < 0  # soil is freezing when flag is true and thawing otherwise.
        min_swc_ice = torch.where(freeze_flag, swc_ice, torch.zeros_like(swc_ice))
        # the solid water will decrease when thawing, so the Sus at timestep t should be no more than that at t-1
        # subtract a small number to make sure Sus is smaller than Su, so that Sul will not be 0 for gradient tracking
        max_swc_ice = torch.where(freeze_flag, swc_tot, swc_ice)
        if self.rr_config['soil']['freeze_thaw_nn']:
            # [n_basin, n_mul_comp, 9 + n_attrs]
            tas, tas_norm = forc[:, :, 1:2], forc_norm[:, :, 1:2]  # temperature
            nRad_norm, prec_norm, wins_norm = forc_norm[:, :, 5:6], forc_norm[:, :, 0:1], forc_norm[:, :, 4:5]
            x = torch.cat((swc_liq.unsqueeze(-1), swc_ice.unsqueeze(-1), swe.unsqueeze(-1), LAI.unsqueeze(-1), tas_norm,
                           nRad_norm, prec_norm, wins_norm, attrs_norm), dim=-1)
            if self.rr_config['soil']['freeze_thaw_nn_type'] == 'mlp':
                out = self.freeze_thaw_nn(x)
            elif self.rr_config['soil']['freeze_thaw_nn_type'] == 'lstm':
                out, hidden = self.freeze_thaw_nn(x, self.freeze_thaw_hidden)
                self.freeze_thaw_hidden = hidden
            param_freCoef = F.sigmoid(out).squeeze(-1)
            param_freCoef = self.rescale_param_range({'param_freCoef_rr': param_freCoef})['param_freCoef_rr']
        swc_ice = min_swc_ice + (max_swc_ice - min_swc_ice) * param_freCoef
        swc_liq = torch.clamp(swc_tot - swc_ice, min=self.eps)

        Q_sub = torch.clamp(param_Qmax * torch.exp(-1 * param_F * (param_Smax - swc_ice - swc_liq)),
                            max=torch.min(swc_liq - self.eps, param_Qmax), min=torch.zeros_like(param_Qmax))

        # calculate T and E_soil
        pet = forc[:, :, 2]
        cr = torch.clamp(swc_liq / (param_Smax - swc_ice + self.eps), max=1, min=0)
        T = torch.clamp((pet * LAI / self.LAI_max - Ei - Ew) * cr, max=swc_liq - Q_sub,
                        min=torch.zeros_like(swc_liq) + self.eps)  # transpiration
        E_soil = torch.zeros_like(pet)  # soil evaporation
        mask = LAI < self.LAI_max
        E_soil[mask] = torch.clamp(param_Ks[mask] * (pet[mask] * (1 - LAI[mask] / self.LAI_max[mask])) * cr[mask],
                                   min=torch.zeros_like(swc_liq[mask]),
                                   max=torch.clamp(torch.min(swc_liq[mask] - Q_sub[mask] - T[mask],
                                                         pet[mask] - Ei[mask] - Ew[mask] - T[mask]), min=0))
        # calculate Q_surf
        swc_liq = swc_liq + Peff + glac_melt + snow_melt - E_soil - Q_sub - T
        Q_surf = torch.clamp(swc_liq - (param_Smax - swc_ice), min=0)
        swc_liq = swc_liq - Q_surf

        return swc_liq, swc_ice, E_soil, T, Q_sub, Q_surf

    def cal_veg_grow(self, forc, ant_tas, acc_tas, LAI, E_tot, Biom, param_uWUE, param_Ksg, param_Cg, param_slp, param_lmt):
        """
        Only consider green leaves when using LAI
        :param forc: forcing data including prec, tas, pet, rhu, wind, nRad, prs, and doy.
        :param ant_tas: antecedent temperature, with a shape of (n_basin, n_mul_comp, n_time).
        :param acc_tas: accumulated temperature, with a shape of (n_basin, n_mul_comp).
        :param LAI: Green LAI
        :param E_tot: Evapotranspiration
        :param Biom: Green biomass
        :param param_uWUE: underlying water use efficiency (kg CO2/kg H2O) | Range: (0.001, 0.1)
        :param param_Ksg: Natural decay factor for green biomass (d-1) | Range: (0.001, 0.02)
        :param param_Cg: Coefficient to calculate LAI from green biomass | Range: (0.005, 0.05)
        :param param_slp: slope parameter for the carbon allocation function
        :param param_lmt: limiting parameter for the carbon allocation function
        """
        # calculate vpd
        tas, rhu, nRad = forc[:, :, 1], forc[:, :, 3], forc[:, :, 5]
        es = 0.611 * torch.exp(17.27 * tas / (tas + 237.3))
        vpd = torch.clamp(es * (1 - rhu / 100), min=torch.zeros_like(es) + self.eps,
                          max=es)  # vapor pressure deficit (kPa)
        # calculate accumulated temperature over 0 celsius
        gs_mask = ant_tas.mean(dim=-1) > 0  # growing season mask
        acc_tas_tt = torch.where(gs_mask, acc_tas + tas, torch.zeros_like(acc_tas))

        # calculate net primary production (NPP)
        rho = 999.9  # water density kg/m3
        w = 0.55  # converts CO2 gained to dry matter (kg DM kg-1 CO2)
        mu = 0.3  # the ratio of nighttime to daytime CO2 exchange
        wue = torch.clamp(param_uWUE / torch.sqrt(vpd * 10), max=self.params_range['param_uWUE_rr'][1])
        npp = 0.75 * (1 - mu) * E_tot * wue * rho * w  # using underlying wue

        # determine the potential LAI for different vegetation types
        LAI_pot = self.LAI_max * 1.2
        # determine green biomass
        if self.rr_config['veg']['alloc_func'] == 'linear':
            phi = 1 - LAI / LAI_pot
        elif self.rr_config['veg']['alloc_func'] == 'exponential':
            phi = 1 / (1 + torch.exp(param_slp * (LAI / LAI_pot - param_lmt)))
        else:
            raise ValueError('unknown alloc_func')
        Biom = Biom + npp * phi - Biom * param_Ksg

        # calculate leaf area index
        Biom = torch.clamp(Biom, min=torch.zeros_like(self.LAI_min), max=LAI_pot / param_Cg)
        # Bg = torch.clamp(Bg, min=self.LAI_min / param_Cg, max=self.LAI_max / param_Cg)
        LAI = param_Cg * Biom

        return LAI, Biom, acc_tas_tt

    def cal_hillslope_routing(self, Q, param_A, param_B, tmax=15):
        """
        Gamma distribution for routing:
                γ(t:a,b) = 1 / (Γ(a) * b^a) * t^(a-1) * e^(-t/b)
                q(t) = ∫(_0^tmax)((γ(t:a,b) * R(t-s))ds

        Reference
        ---------
        Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process‐based
        models with multiphysical outputs can approach state‐of‐the‐art hydrologic prediction accuracy. Water
        Resources Research. https://doi.org/10.1029/2022wr032404

        :param Q: simulated streamflow output by lumped model with a shape of (N, L)
        :param param_A: shape parameter for gamma distribution, with a shape of (N, L) or (N,)
        :param param_B: timescale parameter for gamma distribution, with a shape of (N, L) or (N,)
        :param tmax: maximum time length for unit hydrograph, int
        """

        def UH_conv(x, UH):
            """
            :param x: streamflow output by lumped models, with a shape of [N, F, L]
            :param UH: unit hydrograph, with a shape of [N, F, tmax]
            """
            basin_size, channel_size, sequence_length = x.shape
            kernel_size = UH.shape[-1]

            # batch and basins need to do convolution dependently and we make use of groups
            groups = basin_size
            xx = x.view([channel_size, groups, sequence_length])
            w = UH.view([groups, channel_size, kernel_size])

            # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao, conv1d does \integral(w(\tao)*x(t+\tao))d\tao, hence we flip UH
            y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=kernel_size - 1, stride=1, bias=None)
            y = y[:, :, 0:-(kernel_size - 1)]
            return y.view(x.shape)

        def UH_gamma(a, b, tmax):
            """
            :param a: shape parameter, range(0, 2.9), with a shape of (L, N, 1)
            :param b: timescale parameter, range(0, 6.5), with a shape of (L, N, 1)
            :param tmax: maximum time length for gamma distribution

            :return
                w: weights of the unit graph for different timestep, with a shape of (tmax, N, 1)
            """
            m = a.shape
            w = torch.zeros([tmax, m[1], m[2]])
            aa = F.relu(a[0:tmax, :, :]).view([tmax, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
            theta = F.relu(b[0:tmax, :, :]).view([tmax, m[1], m[2]]) + 0.5  # minimum 0.5
            t = torch.arange(0.5, tmax * 1.0).view([tmax, 1, 1]).expand([-1, m[1], m[2]])
            t = t.to(aa.device)
            denom = (aa.lgamma().exp()) * (theta ** aa)
            mid = t ** (aa - 1)
            right = torch.exp(-t / theta)
            w = 1 / denom * mid * right
            w = w / w.sum(0)  # scale to 1 for each UH

            return w

        if param_A.dim() == 1:
            param_A = param_A.unsqueeze(1).expand(-1, Q.size(1))
        if param_B.dim() == 1:
            param_B = param_B.unsqueeze(1).expand(-1, Q.size(1))

        rout_A = param_A.unsqueeze(-1).permute(1, 0, 2)  # (N, L) to # (L, N, 1)
        rout_B = param_B.unsqueeze(-1).permute(1, 0, 2)
        UH = UH_gamma(rout_A, rout_B, tmax)
        UH = UH.permute(1, 2, 0)  # (tmax, N, 1) to (N, 1, tmax)
        Qin = Q.unsqueeze(1)  # (N, L) to (N, 1, L)
        Q = UH_conv(Qin, UH)

        return Q.squeeze(1)

    def init_model(self, n_basins, n_rivers, params_dict, q_t0=None, s_t0=None, t_len=5):
        """
        :param q_t0: initial inflow and outflow of each river reach.
        :param s_t0: initial reservoir storage of model.
        :param t_len: used to initiate a tensor with the length of 'tLen' to store the past 'tLen' days of temperature.
            Soil is freezing when the average mean of the tensor is below 0 and thawing when the mean is above 0.
        """
        # initiate inflow and outflow of each river reach
        if q_t0 is None:
            q_t0 = (torch.zeros((n_rivers, 2), dtype=torch.float32) + 0.001).to(self.device)
        # initial snow water equivalent
        if s_t0 is None:
            if self.s_dep_t0 is not None:
                Sw_t0 = (self.s_dep_t0 / 10) ** params_dict['param_beta_rr'] * params_dict[
                    'param_Asp_rr'] * 10  # type: ignore
            else:
                Sw_t0 = torch.zeros([n_basins, self.n_mul_comp])  # initialize the snow water equivalent
            # initial biomass
            if self.LAI_t0 is not None:
                Bg_t0 = self.LAI_t0 / params_dict['param_Cg_rr']  # for biomass
            else:
                Bg_t0 = (torch.zeros([n_basins, self.n_mul_comp], dtype=torch.float32) + 200).to(self.device)
            # initiate liquid and solid soil water storage
            ssl_t0 = (torch.zeros([n_basins, self.n_mul_comp], dtype=torch.float32) + 0.001).to(self.device)
            sss_t0 = (torch.zeros([n_basins, self.n_mul_comp], dtype=torch.float32) + 0.001).to(self.device)

            # initiate veg interception storage and accumulated temperature
            i_t0 = (torch.zeros([n_basins, self.n_mul_comp, 2], dtype=torch.float32) + 0.001).to(
                self.device)

            # concat the initial variables
            s_t0 = torch.cat((Sw_t0.unsqueeze(-1), ssl_t0.unsqueeze(-1), sss_t0.unsqueeze(-1),
                              i_t0, Bg_t0.unsqueeze(-1)), dim=-1)

        # initiate a tensor to storage antecedent temperature
        ant_tas = torch.zeros([n_basins, self.n_mul_comp, t_len], dtype=torch.float32).to(self.device)

        return q_t0, s_t0, ant_tas

    def cal_river_routing(self, Qh, R0, param_X, param_K):
        """

        An optimized version of river network routing using the Muskingum method.
        :param Qh: simulated hillslope-routed flow (mm/d) of sub-basins with a shape of [N_basin, L].
        :param param_X: one of muskingum parameters ranges (0, 0.5) with a shape of [N_river, L] or (N_river,).
        :param param_K: another muskingum parameter satisfying 2KX<Δt≤K with a shape of [N_river, L] or (N_river,).
        :param R0: the initial inflow and outflow of river reaches with a shape of [N_river, 2].
        :return: a tuple containing two elements
            Qr: the final runoff at the outlet of each sub-basin.
        """

        # If parameters are 1D, expand them to 2D tensors with the same time length as Qh.
        if param_X.dim() == 1:
            param_X = param_X.unsqueeze(1).expand(-1, Qh.size(1))
        if param_K.dim() == 1:
            param_K = param_K.unsqueeze(1).expand(-1, Qh.size(1))

        # Calculate coefficients C1, C2, C3 for the Muskingum method.
        dT = 1  #  simulation time step: 1 day
        C1 = (dT - 2 * param_K * param_X) / (2 * param_K * (1 - param_X) + dT)
        C2 = (dT + 2 * param_K * param_X) / (2 * param_K * (1 - param_X) + dT)
        C3 = (2 * param_K * (1 - param_X) - dT) / (2 * param_K * (1 - param_X) + dT)

        # Convert hillslope runoff depth (mm/d) to discharge (m3/s).
        Qh_m3s = Qh * self.bsn_total_area.unsqueeze(1) * 1000 / (3600 * 24)
        # Initialize the final sub-basin outlet runoff tensor.
        Qr = torch.zeros_like(Qh_m3s)

        # Initialize inflow and outflow of reaches at time t-1.
        inflow_t0, outflow_t0 = R0[:, 0].clone(), R0[:, 1].clone()

        # Main time loop
        for t in range(Qh.size(1)):
            # Initialize inflow and outflow tensors for the current timestep t.
            inflow_tt = torch.zeros_like(inflow_t0)
            outflow_tt = torch.zeros_like(outflow_t0)

            # Loop through the pre-computed steps which are in the correct physical order.
            for step in self.routing_steps:
                idx = step["current_reach_idx"]
                bsn_idx = step["current_bsn_idx"]
                up_idx = step["upstream_reach_idx"]

                if idx.numel() == 0:
                    continue
                # 1. Calculate inflow for the reaches in the current step.
                #  Inflow = Hillslope runoff from the current sub-basin + Outflow from the immediate upstream reach
                #  (if it exists), which has already been calculated in this timestep.
                current_inflow = Qh_m3s[bsn_idx, t]
                if up_idx is not None:
                    current_inflow += outflow_tt[up_idx]

                # We use += to handle cases where multiple tributaries merge into the same main stem reach.
                inflow_tt[idx] += current_inflow

                # 2. Calculate outflow for the current reaches (this is the core physical process).
                outflow_tt[idx] = C1[idx, t] * inflow_tt[idx] + C2[idx, t] * inflow_t0[idx] +  C3[idx, t] * outflow_t0[idx]

                # 3. Handle tributary merges.
                #    Add the outflow calculated in this step to the inflow of the next-level main stem reach.
                merge_from = step["merge_from_idx"]
                merge_to = step["merge_to_idx"]
                if merge_from is not None and merge_to is not None and merge_from.numel() > 0:
                    inflow_tt = inflow_tt.index_add(0, merge_to, outflow_tt[merge_from])

            # Calculation of Qr and state update.
            Qr[:, t] = Qh_m3s[:, t].clone()
            Qr[self.riv_to_bsn_tensor, t] = inflow_tt
            if self.outlet_basin_indices.numel() > 0:
                Qr[:, t] = Qr[:, t].index_add(0, self.outlet_basin_indices, outflow_tt[self.outlet_inflow_indices])
            inflow_t0, outflow_t0 = inflow_tt, outflow_tt

        return Qr, torch.stack([inflow_t0, outflow_t0], dim=1)

    def get_not_none_params(self):
        """
        Determine the non-None parameters for the snow module and glacier module based on the configuration.
        """
        # parameters for snow module
        snow_config = self.rr_config['snow']
        params_partition = ['param_ts_rr']
        params_snowmelt = ['param_tm_rr', 'param_ds6_rr', 'param_ds12_rr'] if snow_config['sinusoidal_ddf'] \
            else ['param_tm_rr', 'param_ds_rr']
        params_snow2ice = ['param_snow2ice_rr'] if snow_config['snow2ice'] else []
        if (snow_config['swe2sd'] and snow_config['swe2sd_nn'] is False) or self.s_dep_t0 is not None:
            params_swe2sd = ['param_Asp_rr', 'param_beta_rr']
        else:
            params_swe2sd = []
        params_snow = params_partition + params_snowmelt + params_snow2ice + params_swe2sd

        # parameters for vegetation module
        if self.rr_config['veg']['alloc_func'] == 'exponential':
            params_veg = ['param_Imax_rr', 'param_uWUE_rr', 'param_Ksg_rr', 'param_Cg_rr', 'param_slp_rr',
                          'param_lmt_rr']
        else:
            params_veg = ['param_Imax_rr', 'param_uWUE_rr', 'param_Ksg_rr', 'param_Cg_rr']

        # parameters for soil module
        if self.rr_config['soil']['freeze_thaw_nn']:
            params_soil = ['param_Smax_rr', 'param_F_rr', 'param_Qmax_rr', 'param_Ks_rr']
        else:
            params_soil = ['param_Smax_rr', 'param_F_rr', 'param_Qmax_rr', 'param_freCoef_rr', 'param_Ks_rr']

        # parameters for hillslope routing
        params_hillslope_rout = ['param_A_rr', 'param_B_rr']

        # parameters for channel routing
        params_river_rout = ['param_X_rr', 'param_K_rr']

        return params_snow + params_soil + params_veg + params_hillslope_rout + params_river_rout

    def get_dynamic_params(self):
        dynamic_params = self.rr_config['nn_params']['dynamic']['params']
        if self.rr_config['soil']['freeze_thaw_nn'] and 'param_freCoef_rr' in dynamic_params:
            dynamic_params.remove('param_freCoef_rr')
        return dynamic_params

    def rescale_param_range(self, params: dict):
        """
        Constrain parameters into physical ranges.
        """
        for k, v in params.items():
            if k in self.params_range.keys():
                params[k] = v * (self.params_range[k][1] - self.params_range[k][0]) + self.params_range[k][0]
            if k == 'param_K_rr':  # rescale to satisfy 2𝐾𝑋<Δt≤𝐾, i.e., 1 <= K < 1/2/param_X
                params[k] = torch.clamp(params[k], min=torch.ones_like(params[k]),
                                        max=1 / 2 / (params['param_X_rr'] + self.eps))

        return params

    def cal_glac_input(self, glac_sim):
        """
        Calculates the total glacier-derived runoff (snowmelt + ice melt) and expresses it
        as an average water depth over the entire sub-basin area.
        This follows Scheme 2 (Meltwater Infiltration and Routing), preparing the glacier melt
        as an additional water input for the non-glaciated part of the basin.

        Args:
            glac_sim (dict): A dictionary containing simulation results for glaciated areas.
                             Expected keys: 'g_area', 's_melt', 'g_melt', 's_pr'.

        Returns:
            torch.Tensor: A tensor named 'glac_melt' representing the basin-averaged depth of
                          water from glaciers. Shape: [num_timesteps, num_basins].
        """
        # --- 1. Prepare Indices and Tensors ---
        # Get the indices for basins that contain glaciers
        glac_bsn_idx = torch.tensor(self.glac_bsn_idx, dtype=torch.long, device=self.device)
        # Get the area of each glacier band over time
        bsn_glac_area = glac_sim['g_area']  # Shape: [num_glacier_basins, timesteps]

        num_basins = self.bsn_total_area.size(0)
        num_timesteps = bsn_glac_area.size(1)

        # --- 2. Calculate Total Glacier Runoff Source (as depth on glacier area) ---
        glacier_runoff_source = glac_sim['s_melt'] + glac_sim['g_melt'] + glac_sim['s_pr']  # Shape: [num_glacier_basins, timesteps]

        # --- 3. Map Glacier-Specific Values to the Full Basin Dimension ---
        glacier_runoff_map = torch.zeros(num_basins, num_timesteps, device=self.device)
        glacier_runoff_map.index_copy_(0, glac_bsn_idx, glacier_runoff_source)

        # We also need the map of glacier areas for the volume calculation
        glacier_area_map = torch.zeros(num_basins, num_timesteps, device=self.device)
        glacier_area_map.index_copy_(0, glac_bsn_idx, bsn_glac_area)

        # --- 4. Convert to Basin-Average Depth (mm) ---
        # Expand total basin area to match the timestep dimension for broadcasting
        bsn_total_area_expanded = self.bsn_total_area.unsqueeze(1).expand(-1, num_timesteps)
        non_glac_area = bsn_total_area_expanded - glacier_area_map
        # A small epsilon to prevent division by zero for basins with zero area
        epsilon = 1e-9
        # Calculate the total runoff volume from glaciers for each basin
        glacier_runoff_volume = glacier_runoff_map * glacier_area_map
        # Calculate the basin-averaged depth by spreading the volume over the entire basin area
        # glac_melt_basin_avg = glacier_runoff_volume / (non_glac_area + epsilon)  # Shape: [num_basins, timesteps]
        glac_melt_basin_avg = glacier_runoff_volume / (bsn_total_area_expanded + epsilon)

        return glac_melt_basin_avg

    def cal_glac_rr_weighted_state(self, glac_sim, rr_sim):
        # --- 1. Prepare Indices and Area Tensors ---
        glac_bsn_idx = torch.tensor(self.glac_bsn_idx, dtype=torch.long, device=self.device)  # 1D tensor of basin indices that have glaciers
        bsn_glac_area = glac_sim['g_area']  # Glacier area, shape [num_glaciers, timesteps]

        num_basins = self.bsn_total_area.size(0)
        num_timesteps = bsn_glac_area.size(1)

        # Create a full map of glacier area aligned with the basin dimension
        # It will be zero everywhere except for basins that contain a glacier.
        glacier_area_map = torch.zeros(num_basins, num_timesteps, device=self.device)
        glacier_area_map.index_copy_(0, glac_bsn_idx, bsn_glac_area)

        # Calculate total and non-glacier areas for all basins in a vectorized way
        bsn_total_area = self.bsn_total_area.unsqueeze(1).expand(-1, num_timesteps)
        bsn_non_glac_area = bsn_total_area - glacier_area_map

        # --- 2. Calculate Area-Based Weights ---
        # Add a small epsilon to prevent division by zero for basins with zero area
        epsilon = 1e-9
        non_glacier_weight = bsn_non_glac_area / (bsn_total_area + epsilon)
        glacier_weight = glacier_area_map / (bsn_total_area + epsilon)

        # --- 3. Scale Non-Glacier Variables ---
        # Apply the non-glacier weight to all relevant variables from the regular simulation.
        # For basins without glaciers, this weight is 1.0, so their values remain unchanged.
        # vars_to_scale = ['Peff', 'Qsub', 'Qsurf', 'Qrain', 'Qsnow', 'Etot', 'Esnow', 'Eicp', 'Esoil', 'T', 'LAI',
        #                 'biom', 'swe', 'sdep', 'swc_liq', 'swc_ice']
        # for var in vars_to_scale:
        #     if var in rr_sim:
        #         rr_sim[var] = rr_sim[var] * non_glacier_weight

        # --- 4. Calculate and Add Weighted Glacier Variables ---
        # Calculate total glacier runoff and ET contributions
        glacier_runoff_source = glac_sim['s_melt'] + glac_sim['g_melt'] + glac_sim['s_pr']
        glacier_et_source = glac_sim['s_slm'] + glac_sim['g_slm']

        # Initialize target tensors in rr_sim
        rr_sim['Qglac'] = torch.zeros_like(rr_sim['Qsub'])
        rr_sim['Eglac'] = torch.zeros_like(rr_sim['Esnow'])

        # Scatter the glacier-specific values to the correct basin locations
        rr_sim['Qglac'].index_copy_(0, glac_bsn_idx, glacier_runoff_source)
        rr_sim['Eglac'].index_copy_(0, glac_bsn_idx, glacier_et_source)
        # Apply the glacier area weight to get the final contribution
        rr_sim['Qglac'] = rr_sim['Qglac'] * glacier_weight
        rr_sim['Eglac'] = rr_sim['Eglac'] * glacier_weight
        # Add the glacier ET contribution to the total ET
        rr_sim['Etot'] = rr_sim['Etot'] + rr_sim['Eglac']

        return rr_sim

    def _precompute_routing_steps(self):
        """
        Pre-computes tensor indices for each step of the routing process,
        faithfully replicating the physical order from the original code.

        This function should be called once during initialization.
        """
        # divide the routing order array into different level of tributary sub-basins
        self.riv_net = []
        # count the number of non-pad values in each river levels
        for i in range(0, len(self.rout_order)):
            nb = np.count_nonzero(self.rout_order[i] != self.pad, axis=1)
            riv_net = self.rout_order[i, :, :np.max(nb) + 1]
            mask = ~(riv_net == self.pad).all(axis=1)
            self.riv_net.append(riv_net[mask])

        # Create a fast lookup map (O(1) complexity) from basin index to reach index.
        # This replaces the inefficient use of list.index() (O(N) complexity).
        self.bsn_to_riv_map = {bsn_idx: i for i, bsn_idx in enumerate(self.riv_up_bsn_idx)}
        self.riv_to_bsn_tensor = torch.tensor(self.riv_up_bsn_idx, dtype=torch.long, device=self.device)

        # This list will store dictionaries, each representing a single vectorized calculation step.
        # The order of the dictionaries in this list preserves the correct physical routing sequence.
        self.routing_steps = []

        # Loop through network levels (k) and segment order (i), which defines the correct physical sequence.
        for k, trib_net in enumerate(self.riv_net):
            next_trib_net = self.riv_net[k + 1] if k + 1 < len(self.riv_net) else None

            # The loop range correctly matches the original code's focus on the river reaches (from i to i+1).
            for i in range(trib_net.shape[1] - 1):
                # --- Prepare indices for the current computation step (k, i) ---
                # Identify the valid river reaches in this step. A reach is valid only if its downstream basin is not a pad value.
                valid_mask = (trib_net[:, i + 1] != self.pad)
                # Get the upstream basins for these valid reaches.
                current_bsns = trib_net[:, i][valid_mask]
                if len(current_bsns) == 0:
                    continue
                # Convert basin indices to reach indices using the O(1) lookup.
                current_reach_idx = torch.tensor(
                    [self.bsn_to_riv_map[b] for b in current_bsns if b in self.bsn_to_riv_map],
                    dtype=torch.long, device=self.device
                )
                # We also need the basin indices themselves to extract hillslope runoff Qh.
                current_bsn_idx = torch.tensor(
                    [b for b in current_bsns if b in self.bsn_to_riv_map],
                    dtype=torch.long, device=self.device
                )

                # --- Identify the upstream source reaches for the current step's inflow ---
                upstream_reach_idx = None
                if i > 0:
                    # Get the upstream basins corresponding to the current basins (i-1 column).
                    up_bsns = trib_net[:, i - 1][valid_mask]
                    up_bsns = [b for b in up_bsns if b in self.bsn_to_riv_map]
                    if up_bsns:
                        upstream_reach_idx = torch.tensor(
                            [self.bsn_to_riv_map[b] for b in up_bsns],
                            dtype=torch.long, device=self.device
                        )

                # --- Handle merges from the current tributary to the next main stem ---
                merge_inflow_from_reach_idx = None
                merge_inflow_to_reach_idx = None

                # This logic captures the `if next_trib_net is not None...` block from the original code.
                # We check if the downstream basin of the current reach exists in the next level network.
                if next_trib_net is not None and len(current_bsns) > 0:
                    down_bsns = trib_net[:, i + 1][valid_mask]
                    is_in_next_net = np.isin(down_bsns, next_trib_net)
                    if np.any(is_in_next_net):
                        # Get the current reaches that will merge into the main stem.
                        source_reaches = current_reach_idx[is_in_next_net]
                        # Get the target basins in the next network level that receive the merge flow.
                        target_bsns = down_bsns[is_in_next_net]
                        target_reaches = torch.tensor(
                            [self.bsn_to_riv_map[b] for b in target_bsns if b in self.bsn_to_riv_map],
                            dtype=torch.long, device=self.device
                        )
                        # Ensure indices are valid and non-empty.
                        valid_source_mask = torch.tensor(
                            [b in self.bsn_to_riv_map for b in target_bsns],
                            device=self.device
                        )
                        source_reaches = source_reaches[valid_source_mask]

                        if source_reaches.numel() > 0 and target_reaches.numel() > 0:
                            merge_inflow_from_reach_idx = source_reaches
                            merge_inflow_to_reach_idx = target_reaches

                step_data = {
                    "current_reach_idx": current_reach_idx,
                    "current_bsn_idx": current_bsn_idx,
                    "upstream_reach_idx": upstream_reach_idx,
                    "merge_from_idx": merge_inflow_from_reach_idx,
                    "merge_to_idx": merge_inflow_to_reach_idx
                }
                self.routing_steps.append(step_data)

        # Pre-computation for the final outlets (this part remains correct).
        outlet_inflow_indices = []
        outlet_basin_indices = []
        out_bsns = [i for i in range(len(self.bsn_total_area)) if i not in self.bsn_to_riv_map and i in self.rout_order]
        for out_bsn in out_bsns:
            dim1_idx, dim2_idx, dim3_idx = np.where(self.rout_order == out_bsn)
            if dim1_idx.size > 0:
                up_bsns = self.rout_order[dim1_idx, dim2_idx, dim3_idx - 1]
                up_reaches_idx = [self.bsn_to_riv_map[up_bsn] for up_bsn in up_bsns if
                                  up_bsn in self.bsn_to_riv_map]
                if up_reaches_idx:
                    outlet_inflow_indices.extend(up_reaches_idx)
                    outlet_basin_indices.extend([out_bsn] * len(up_reaches_idx))

        self.outlet_inflow_indices = torch.tensor(outlet_inflow_indices, dtype=torch.long, device=self.device)
        self.outlet_basin_indices = torch.tensor(outlet_basin_indices, dtype=torch.long, device=self.device)

    def cal_Penman_ET(self, tas, prs, nRad, wins, rhu, melt):
        # convert all inputs to tensors
        _273_15 = torch.tensor(273.15, dtype=tas.dtype, device=tas.device)
        _1_01 = torch.tensor(1.01, dtype=tas.dtype, device=tas.device)
        _0_287 = torch.tensor(0.287, dtype=tas.dtype, device=tas.device)
        _4098 = torch.tensor(4098, dtype=tas.dtype, device=tas.device)
        _0_6108 = torch.tensor(0.6108, dtype=tas.dtype, device=tas.device)
        _17_27 = torch.tensor(17.27, dtype=tas.dtype, device=tas.device)
        _237_3 = torch.tensor(237.3, dtype=tas.dtype, device=tas.device)
        _0_622 = torch.tensor(0.622, dtype=tas.dtype, device=tas.device)
        _1_013e_3 = torch.tensor(1.013e-3, dtype=tas.dtype, device=tas.device)
        _4_87 = torch.tensor(4.87, dtype=tas.dtype, device=tas.device)
        _67_8 = torch.tensor(67.8, dtype=tas.dtype, device=tas.device)
        _5_42 = torch.tensor(5.42, dtype=tas.dtype, device=tas.device)
        _0_2 = torch.tensor(0.2, dtype=tas.dtype, device=tas.device)
        _2_834 = torch.tensor(2.834, dtype=tas.dtype, device=tas.device)  # Lambda
        _0_334 = torch.tensor(0.334, dtype=tas.dtype, device=tas.device)  # Lambda_fusion
        _86400 = torch.tensor(86400, dtype=tas.dtype, device=tas.device)
        _80 = torch.tensor(80, dtype=tas.dtype, device=tas.device)

        # calculate saturated vapor pressure
        Delta = _4098 * (_0_6108 * torch.exp(_17_27 * tas / (tas + _237_3))) / ((tas + _237_3) ** 2)
        Gs = _0_2 * nRad
        rho_w = 1  # density of water, g/cm3
        rad_melt = melt * _0_334 * rho_w
        gamma = (_1_013e_3 * prs) / (_0_622 * _2_834)
        cp = gamma * _0_622 * _2_834 / prs
        rho = prs / (_0_287 * (_1_01 * (_273_15 + tas)))

        # calculate aerodynamic resistance
        z = torch.tensor(10, dtype=wins.dtype, device=wins.device)
        u2 = wins * _4_87 / torch.log(_67_8 * z - _5_42)
        h0 = torch.tensor(1e-4, dtype=u2.dtype, device=u2.device)
        d = h0 * 2 / 3
        zom = 0.1 * h0
        k = torch.tensor(0.41, dtype=u2.dtype, device=u2.device)
        zm = torch.tensor(2, dtype=u2.dtype, device=u2.device)
        ra = torch.log((zm - d) / zom) * torch.log((zm - d) / zom) / k ** 2 / (u2 + self.eps)
        # mean sa
        es = _0_6108 * torch.exp(_17_27 * tas / (tas + _237_3))
        # actual vapor pressure
        ea = es * rhu / 100

        # during melt season, the energy available for potential evapotranspiration is reduced by the melt energy
        pet = ((nRad - Gs - rad_melt) * Delta + rho * cp * (es - ea) / ra * _86400) / (_2_834 * (Delta + gamma))
        # deposition occurs when pet < 0 and rhu is high
        zero_mask = (pet < 0) & (rhu <= _80)
        pet[zero_mask] = 0

        return pet

    def forward(self, forc: torch.Tensor, glac_melt: torch.Tensor, forc_norm: torch.Tensor, attrs_norm: torch.Tensor,
                s_t0: torch.Tensor, ant_tas: torch.Tensor, params_dict: dict):
        """
        :param forc: includes prec, tas, pet, rhu, wind, nRad, prs, and doy, with the shape of (n_basin, 8).
        :param glac_melt: glacier melt input, with the shape of (n_basin, 1, n_mul).
        :param forc_norm: normalized forcing data, with the shape of (n_basin, 8).
        :param attrs_norm: normalized basin attributes, with the shape of (n_basin, n_attrs).
        :param params_dict: a dictionary containing the parameters for the snow, vegetation, soil, and routing modules.
        :param ant_tas: tensor of shape [N, n_mul, t_len], including the antecedent temperature.
        :param s_t0: tensor of shape [N, n_mul, 2], including parF and parALPHA at the current timestep.
        :return
            output: a dict consisting final runoff, reservoir and river storage, as well as other hydrological
            variables at current timestep t.
        """
        forc = forc.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_basin, n_mul_comp, 8]
        forc_norm = forc_norm.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_basin, n_mul_comp, 8]
        attrs_norm = attrs_norm.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_basin, n_mul_comp, 10]
        glac_melt = glac_melt.unsqueeze(-1).expand(-1, self.n_mul_comp)  # [n_basin, n_mul_comp]

        # initiate the storages
        swe_t0, swc_liq_t0, swc_ice_t0 = s_t0[:, :, 0], s_t0[:, :, 1], s_t0[:, :, 2], # snow water equivalent, liquid and ice soil water content
        iwc_t0, acc_tas_t0, biom_t0 = s_t0[:, :, 3], s_t0[:, :, 4], s_t0[:, :, 5] # interception water content, accumulated temperature, and green biomass
        LAI_t0 = params_dict['param_Cg_rr'] * biom_t0

        # snow bucket
        swe_tt, rainfall, snow_melt, E_snow, snow2ice, snow_dep = self.cal_snow_bucket(swe=swe_t0, forc=forc, forc_norm=forc_norm,
                                                                                       attrs_norm=attrs_norm,
                                                                                       param_ts=params_dict['param_ts_rr'],
                                                                                       param_tm=params_dict['param_tm_rr'],
                                                                                       param_ds=params_dict['param_ds_rr'],
                                                                                       param_ds6=params_dict['param_ds6_rr'],
                                                                                       param_ds12=params_dict['param_ds12_rr'],
                                                                                       param_snow2ice=params_dict['param_snow2ice_rr'],
                                                                                       param_Asp=params_dict['param_Asp_rr'],
                                                                                       param_beta=params_dict['param_beta_rr'])

        # vegetation interception
        Peff, iwc_tt, E_icp = self.cal_veg_bucket(rainfall=rainfall, LAI=LAI_t0, iwc=iwc_t0,
                                                  evap_snow=E_snow, forc=forc,
                                                  param_Imax=params_dict['param_Imax_rr'])

        # soil bucket
        swc_liq_tt, swc_ice_tt, E_soil, T, Q_sub, Q_surf = self.cal_soil_bucket(forc=forc, forc_norm=forc_norm,
                                                                                attrs_norm=attrs_norm,
                                                                                swc_liq=swc_liq_t0, swc_ice=swc_ice_t0,
                                                                                swe=swe_t0, Peff=Peff,
                                                                                glac_melt=glac_melt,
                                                                                snow_melt=snow_melt, Ei=E_icp,
                                                                                Ew=E_snow, LAI=LAI_t0, ant_tas=ant_tas,
                                                                                param_Smax=params_dict['param_Smax_rr'],
                                                                                param_F=params_dict['param_F_rr'],
                                                                                param_Qmax=params_dict['param_Qmax_rr'],
                                                                                param_freCoef=params_dict[
                                                                                    'param_freCoef_rr'],
                                                                                param_Ks=params_dict['param_Ks_rr'])
        swc_tot = swc_liq_tt + swc_ice_tt

        # update vegetation growth
        LAI_tt, biom_tt, Tacc_tt = self.cal_veg_grow(forc=forc, ant_tas=ant_tas, acc_tas=acc_tas_t0, LAI=LAI_t0,
                                                     E_tot=T + E_soil, Biom=biom_t0,
                                                     param_uWUE=params_dict['param_uWUE_rr'],
                                                     param_Ksg=params_dict['param_Ksg_rr'],
                                                     param_Cg=params_dict['param_Cg_rr'],
                                                     param_slp=params_dict['param_slp_rr'],
                                                     param_lmt=params_dict['param_lmt_rr'])

        # Evapotranspiration
        E_tot = E_snow + E_icp + E_soil + T

        output = {'swe': swe_tt, 'iwc': iwc_tt, 'swc_tot': swc_tot, 'biom': biom_tt, 'Tacc': Tacc_tt,  # final reservoir storage
                  'Peff': Peff, 'Qsub': Q_sub, 'Qsurf': Q_surf, 'Qsnow': snow_melt, 'Qrain': rainfall,  # subsurface flow, surface flow, snowmelt and rainfall
                  'sdep': snow_dep,  # snow depth
                  'LAI': LAI_tt,  # leaf area index
                  'Etot': E_tot, 'Esnow': E_snow, 'Eicp': E_icp, 'Esoil': E_soil, 'T': T,  # evapotranspiration
                  'swc_liq': swc_liq_tt, 'swc_ice': swc_ice_tt}  # liquid and solid water in soil bucket

        return output


class GlacierDynCell(nn.Module):
    def __init__(self, bsn_band_ids_dict: dict, n_attrs: int, snow_config: dict, glac_config: dict, n_mul_comp: int,
                 params_range: dict, dropout: float = 0.5, device: Union[str, torch.device] = 'cpu'):
        super(GlacierDynCell, self).__init__()
        self.eps = 1e-6  # a small value to avoid division by zero
        self.device = device
        self.params_range = params_range
        self.bsn_band_ids_dict = bsn_band_ids_dict  # dictionary of basin-band ids
        self.band_ids_lst = list(self.bsn_band_ids_dict.values())  # indices of bands in each basin
        self.band_slope = glac_config['slope']

        # precalculate index for aggregating band variables to basin variables
        self.n_basins = len(bsn_band_ids_dict)
        self.n_bands = sum(len(v) for v in bsn_band_ids_dict.values())
        # create a map from band index to basin index
        self.band_to_basin_map = torch.empty(self.n_bands, dtype=torch.long, device=device)
        self.lowest_band_indices = []  # the lowest band index for each basin
        for bsn_id, band_ids in enumerate(bsn_band_ids_dict.values()):
            if band_ids:
                self.band_to_basin_map[band_ids] = bsn_id
                self.lowest_band_indices.append(band_ids[0])
        self.lowest_band_indices = torch.tensor(self.lowest_band_indices, device=device, dtype=torch.long)
        self.upper_neighbor_map = self._init_upper_neighbor_map()

        self.snow_config = snow_config  # configuration for snow module
        self.gla_config = glac_config  # configuration for glacier module
        self.rho_ice = 0.85  # density of ice, g/cm3, Huggonnet et al., 2021
        self.rho_w = 1  # density of water, g/cm3
        self.n_mul_comp = n_mul_comp  # number of multiple components
        self.snow_depth_t0 = snow_config['s_dep_t0']  # initial snow depth
        if self.snow_depth_t0 is not None:
            if self.snow_depth_t0.dim() == 1:
                self.snow_depth_t0 = self.snow_depth_t0.unsqueeze(1).expand(-1, self.n_mul_comp)
        # all parameters
        self.param_names = list(self.params_range.keys())
        self.params_scale = {param: 'band' if param not in ['param_m_glac', 'param_n_glac'] else 'basin' for param in
                             self.param_names}
        self.non_none_params = self.get_not_none_params()

        # initialize neural networks
        if self.snow_config['swe2sd_nn']:
            self.swe2sd_nn = MlpModules(in_features=7 + n_attrs, hidden_size=self.snow_config['swe2sd_nn_hidden'],
                                        out_features=1, dropout=dropout)

        if self.gla_config['glacier_shift']:
            self.shift_out_idx, self.shift_in_idx, self.lowest_band_mask = self.init_glac_shift_idx()
            if self.gla_config['glacier_shift_nn']:
                self.gla_shift_nn = MlpModules(in_features=2 + n_attrs,
                                               hidden_size=self.gla_config['glacier_shift_nn_hidden'],
                                               out_features=1, dropout=dropout, unnorm_var_num=1)

    def vol_area_curve(self, param_m, param_n, area=None, vol=None, cal_vol=False, cal_area=False):
        """
        Calculate volume or area based on the volume-area scaling relationship
        """
        if cal_vol:
            return torch.clamp(param_m * ((area + self.eps) ** param_n), min=0)
        elif cal_area:
            vol = torch.clamp(vol, min=1e-6)
            return torch.clamp(((vol + self.eps) / param_m) ** (1 / param_n), min=0)
        else:
            raise ValueError('Either cal_area or cal_vol must be True')

    def cal_snow(self, swe, forc, forc_norm, attrs_norm, area_band, param_ts, param_tm, param_ds=None, param_ds6=None,
                 param_ds12=None, param_snow2ice=None, param_Asp=None, param_beta=None):
        """
        :param swe: initial snow water equivalent.
        :param forc: forcing data including prec, tas, pet, rhu, wind, nRad, prs, and doy.
        :param forc_norm: normalized forcing data, with the shape of (n_band, n_mul_comp, 7).
        :param attrs_norm: elevation band attributes.
        :param area_band: area of the elevation band.
        :param param_ts: temperature threshold for snowfall and rainfall, °C.
        :param param_tm: temperature threshold for snow melt, °C.
        :param param_ds: degree-day factor for snow melt, mm/°C/day.
        :param param_ds6: degree-day factor for snow melt on 21 June, mm/°C/day.
        :param param_ds12: degree-day factor for snow melt on 21 December, mm/°C/day.
        :param param_snow2ice: a constant coefficient to calculate snow to ice, -.
        :param param_Asp: coefficient to calculate snow pressure, g/cm2, 0.01-0.3.
        :param param_beta: exponent coefficient to calculate snow pressure, 0.8-1.2.
        """
        # unpack forcing data
        prec, tas, rhu, wins = forc[:, :, 0], forc[:, :, 1], forc[:, :, 3], forc[:, :, 4]
        nRad, prs, doy = forc[:, :, 5], forc[:, :, 6], forc[:, :, 7]

        # partition precipitation into snow and rain
        pr = torch.mul(prec, (tas >= param_ts))
        ps = torch.mul(prec, (tas < param_ts))

        # calculate snowmelt
        if self.snow_config['sinusoidal_ddf']:
            ddf = (param_ds6 + param_ds12) / 2 + (param_ds6 - param_ds12) / 2 * torch.sin(
                2 * torch.pi * (doy - 81) / 365)
        else:
            ddf = param_ds
        Gs = 0.2 * nRad  # subsurface heat flux, MJ/m2/day, 20% of net radiation following GLEAM4
        lambda_fusion = 0.334  # latent heat of fusion, MJ/kg
        melt_pot = torch.clamp((nRad - Gs) / self.rho_w / lambda_fusion, min=0)  # potential melt, mm/day
        melt = torch.clamp(ddf * (tas - param_tm), min=torch.zeros_like(swe), max=torch.min(melt_pot, swe))

        # calculate snow transferring to ice
        if self.snow_config['snow2ice']:
            snow2ice = swe * param_snow2ice * (1 + torch.sin(2 * torch.pi * (doy - 81) / 365))
        else:
            snow2ice = torch.zeros_like(swe)

        # calculate snow sublimation
        if self.snow_config['snow_slm']:
            # calculate potential evapotranspiration using Penman formula
            pet = self.cal_Penman_ET(tas=tas, prs=prs, nRad=nRad, rhu=rhu, wins=wins, melt=melt)
            slm = torch.clamp(pet, max=swe - melt)
        else:
            slm = torch.zeros_like(ps)

        # update snow water equivalent
        swe = swe + ps - melt - slm - snow2ice
        swe = torch.clamp(swe, min=torch.zeros_like(swe))

        # calculate snow depth
        if self.snow_config['swe2sd']:
            if self.snow_config['swe2sd_nn']:  # use neural network to calculate snow depth from snow water equivalent
                # [n_band, n_mul_comp, 1 + 6 + n_attrs]
                x = torch.cat((swe.unsqueeze(-1), forc_norm[:, :, 1:7], attrs_norm), dim=-1)
                s_depth = F.relu(self.swe2sd_nn(x)).squeeze(-1)
            else:  # use empirical formula to calculate snow depth from snow water equivalent
                # swe = param_a * sd ** param_beta / rho_w, in which the units of swe and sd are cm.
                s_depth = (swe / 10 / param_Asp + self.eps) ** (1 / param_beta) * 10
        else:
            s_depth = torch.full_like(swe, torch.nan)

        # calculate snow volume (km3)
        s_vol = s_depth * area_band * 1e-6 if s_depth is not None else None

        # output variables
        out = {'s_we': swe, 's_pr': pr, 's_melt': melt, 's_slm': slm, 's_snow2ice': snow2ice, 's_depth': s_depth,
               's_vol': s_vol}

        return out

    def cal_glacier(self, swe, gwe, snow2ice, forc, attrs_norm, area_band, param_tg, param_dg=None,
                    param_dg6=None, param_dg12=None, param_rf=None, cal_shift_flag: bool = True):
        """
        :param area_band: glacier area of the elevation band, km2.
        :param swe: snow water equivalent, mm.
        :param gwe: initial glacier water equivalent, mm.
        :param snow2ice: snow transferring to ice, mm/d.
        :param forc: forcing data including prec, tas, pet, rhu, wind, nRad, and doy.
        :param attrs_norm: glacier attributes.
        :param param_tg: temperature threshold for glacier melt, °C.
        :param param_dg: day-degree factor for glacier melt, mm/°C/day.
        :param param_dg6: day-degree factor for glacier melt on 21 June, mm/°C/day.
        :param param_dg12: day-degree factor for glacier melt on 21 December, mm/°C/day.
        """
        # unpack forcing data
        prec, tas, rhu, wins = forc[:, :, 0], forc[:, :, 1], forc[:, :, 3], forc[:, :, 4]
        nRad, prs, doy = forc[:, :, 5], forc[:, :, 6], forc[:, :, 7]

        # calculate glacier melt
        melt = torch.zeros_like(gwe)
        mask = (swe <= self.eps)  # only consider glacier melt when there is no snow
        if self.gla_config['sinusoidal_ddf']:
            ddf = (param_dg6 + param_dg12) / 2 + (param_dg6 - param_dg12) / 2 * torch.sin(
                2 * torch.pi * (doy - 81) / 365)
        else:
            ddf = param_dg
        Gs = 0.2 * nRad  # subsurface heat flux, MJ/m2/day, 20% of net radiation following GLEAM4
        lambda_fusion = 0.334  # latent heat of fusion, MJ/kg
        melt_pot = torch.clamp((nRad - Gs) / self.rho_w / lambda_fusion, min=0)  # potential melt, mm/day
        melt[mask] = torch.clamp((ddf * (tas - param_tg))[mask], min=torch.zeros_like(gwe)[mask],
                                 max=torch.min(melt_pot[mask], gwe[mask]))

        # calculate glacier sublimation
        slm = torch.zeros_like(gwe)
        if self.gla_config['glacier_slm']:
            # calculate potential evapotranspiration using Penman formular
            pet = self.cal_Penman_ET(tas=tas, prs=prs, nRad=nRad, rhu=rhu, wins=wins, melt=melt)
            tmp_slm = torch.clamp(pet, max=gwe - melt)
            slm[mask] = tmp_slm[mask]  # only consider glacier sublimation when there is no snow

        # calculate net glacier shift
        if self.gla_config['glacier_shift'] and cal_shift_flag:
            H_glac = gwe * self.rho_w / self.rho_ice / 1000 # glacier thickness, m
            # Compute shift-out amount from each elevation band
            if self.gla_config['glacier_shift_nn']:
                # Input: [n_band, n_mul_comp, 2 + n_attrs]
                x = torch.cat((H_glac.unsqueeze(-1), area_band, attrs_norm), dim=-1)
                out = torch.sigmoid(self.gla_shift_nn(x)).squeeze(-1)
                frac = self.rescale_param_range({'param_flow_frac_glac': out})['param_flow_frac_glac']
                gwe_shift_out = torch.clamp(frac * gwe, min=0, max=gwe)
            else:
                # Shallow ice approximation (SIA) for glacier flow
                n = 3
                u = 2 * param_rf / (n + 2) * (self.rho_ice * 1e3 * 9.81 * self.band_slope.unsqueeze(1)) ** n * H_glac ** (n + 1)
                conversion_factor = {'d': 24 * 3600, 'sm': 15 * 24 * 3600, 'm': 30 * 24 * 3600, 'y': 365 * 24 * 3600}
                # glacier mass shift at the giver update_freq, mm
                u_freq = conversion_factor[self.gla_config['update_freq']] * u
                # Intermediate steps for clarity
                V_ice_out = u_freq * torch.sqrt(area_band  * 1e6) * H_glac  # Total ice volume shifted out this timestep [m^3]
                delta_h_ice = V_ice_out / (area_band * 1e6)  # Avg. change in ice thickness over the band [m]
                delta_h_gwe = delta_h_ice * self.rho_ice / self.rho_w  # Convert to GWE change [m]
                gwe_shift_out = delta_h_gwe * 1000  # Convert final result to mm
                gwe_shift_out = torch.clamp(gwe_shift_out, max=gwe)  # cannot shift out more than available

            g_calv = gwe_shift_out * self.lowest_band_mask # glacier calving/melt from the lowest band
            gwe_shift_out = gwe_shift_out *  (1 - self.lowest_band_mask)

            # Initialize shift-in tensor
            gwe_shift_in = torch.zeros_like(gwe)
            source_tensor = gwe_shift_out[self.shift_out_idx].to(gwe_shift_in.dtype)
            # Add shift-out from upper band to shift-in of lower band
            gwe_shift_in = gwe_shift_in.index_add(0, self.shift_in_idx, source_tensor)
            # Net glacier mass shift = incoming - outgoing
            gwe_shift_net = gwe_shift_in - gwe_shift_out - g_calv

        else:
            gwe_shift_net = torch.zeros_like(gwe)
            g_calv = torch.zeros_like(gwe)

        # update glacier water equivalent
        gwe = gwe - melt - slm + snow2ice + gwe_shift_net
        gwe = torch.clamp(gwe, min=torch.zeros_like(gwe))

        # calculate glacier volume (km3)
        gvol = gwe * 1e-6 * area_band * self.rho_w / self.rho_ice

        # output variables
        out = {'g_we': gwe, 'g_melt': melt, 'g_slm': slm, 'g_vol': gvol, 'g_calv': g_calv}

        return out

    def cal_Penman_ET(self, tas, prs, nRad, wins, rhu, melt):
        # convert all inputs to tensors
        _273_15 = torch.tensor(273.15, dtype=tas.dtype, device=tas.device)
        _1_01 = torch.tensor(1.01, dtype=tas.dtype, device=tas.device)
        _0_287 = torch.tensor(0.287, dtype=tas.dtype, device=tas.device)
        _4098 = torch.tensor(4098, dtype=tas.dtype, device=tas.device)
        _0_6108 = torch.tensor(0.6108, dtype=tas.dtype, device=tas.device)
        _17_27 = torch.tensor(17.27, dtype=tas.dtype, device=tas.device)
        _237_3 = torch.tensor(237.3, dtype=tas.dtype, device=tas.device)
        _0_622 = torch.tensor(0.622, dtype=tas.dtype, device=tas.device)
        _1_013e_3 = torch.tensor(1.013e-3, dtype=tas.dtype, device=tas.device)
        _4_87 = torch.tensor(4.87, dtype=tas.dtype, device=tas.device)
        _67_8 = torch.tensor(67.8, dtype=tas.dtype, device=tas.device)
        _5_42 = torch.tensor(5.42, dtype=tas.dtype, device=tas.device)
        _0_2 = torch.tensor(0.2, dtype=tas.dtype, device=tas.device)
        _2_834 = torch.tensor(2.834, dtype=tas.dtype, device=tas.device)  # Lambda
        _0_334 = torch.tensor(0.334, dtype=tas.dtype, device=tas.device)  # Lambda_fusion
        _86400 = torch.tensor(86400, dtype=tas.dtype, device=tas.device)
        _80 = torch.tensor(80, dtype=tas.dtype, device=tas.device)

        # calculate saturated vapor pressure
        Delta = _4098 * (_0_6108 * torch.exp(_17_27 * tas / (tas + _237_3))) / ((tas + _237_3) ** 2)
        Gs = _0_2 * nRad
        rad_melt = melt * _0_334 * self.rho_w
        gamma = (_1_013e_3 * prs) / (_0_622 * _2_834)
        cp = gamma * _0_622 * _2_834 / prs
        rho = prs / (_0_287 * (_1_01 * (_273_15 + tas)))

        # calculate aerodynamic resistance
        z = torch.tensor(10, dtype=wins.dtype, device=wins.device)
        u2 = wins * _4_87 / torch.log(_67_8 * z - _5_42)
        h0 = torch.tensor(1e-4, dtype=u2.dtype, device=u2.device)
        d = h0 * 2 / 3
        zom = 0.1 * h0
        k = torch.tensor(0.41, dtype=u2.dtype, device=u2.device)
        zm = torch.tensor(2, dtype=u2.dtype, device=u2.device)
        ra = torch.log((zm - d) / zom) * torch.log((zm - d) / zom) / k ** 2 / (u2 + self.eps)
        # mean sa
        es = _0_6108 * torch.exp(_17_27 * tas / (tas + _237_3))
        # actual vapor pressure
        ea = es * rhu / 100

        # during melt season, the energy available for potential evapotranspiration is reduced by the melt energy
        pet = ((nRad - Gs - rad_melt) * Delta + rho * cp * (es - ea) / ra * _86400) / (_2_834 * (Delta + gamma))
        # deposition occurs when pet < 0 and rhu is high
        zero_mask = (pet < 0) & (rhu <= _80)
        pet[zero_mask] = 0

        return pet

    def get_not_none_params(self):
        """
        Determine the non-None parameters for the snow module and glacier module based on the configuration.
        """
        # parameters for snow module
        params_partition = ['param_ts_glac']
        params_snowmelt = ['param_tm_glac', 'param_ds6_glac', 'param_ds12_glac'] if self.snow_config[
            'sinusoidal_ddf'] else [
            'param_tm_glac', 'param_ds_glac']
        params_snow2ice = ['param_snow2ice_glac'] if self.snow_config['snow2ice'] else []
        if (self.snow_config['swe2sd'] and self.snow_config['swe2sd_nn'] is False) or self.snow_depth_t0 is not None:
            params_swe2sd = ['param_Asp_glac', 'param_beta_glac']
        else:
            params_swe2sd = []
        params_snow = params_partition + params_snowmelt + params_snow2ice + params_swe2sd

        # parameters for glacier module
        params_gla_melt = ['param_tg_glac', 'param_dg6_glac', 'param_dg12_glac'] if self.gla_config[
            'sinusoidal_ddf'] else ['param_tg_glac', 'param_dg_glac']
        params_vol_area = ['param_m_glac', 'param_n_glac']
        params_gla_shift = ['param_rf_glac'] if self.gla_config['glacier_shift'] and (
                self.gla_config['glacier_shift_nn'] is False) else ['param_flow_frac_glac']
        params_gla = params_gla_melt + params_vol_area + params_gla_shift

        return params_snow + params_gla

    def rescale_param_range(self, params: dict):
        """
        Rescale the parameter range from the range of [0, 1] to the physical range.
        :param params: a dictionary of parameters.
        :return: a dictionary of rescaled parameters.
        """
        for k, v in params.items():
            params[k] = v * (self.params_range[k][1] - self.params_range[k][0]) + self.params_range[k][0]
        return params

    def update_area(self, gvol_band_tt: torch.Tensor, area_band_t0: torch.Tensor, area_bsn_t0: torch.Tensor,
                    param_m: torch.Tensor, param_n: torch.Tensor, update_flag: bool):

        # Step 1: Calculate new total area and area changes per band
        if not update_flag:
            return area_band_t0, area_bsn_t0, gvol_band_tt, None

        # calculate the total volume for each basin
        gvol_bsn_tt = self.trans_var_band2bsn('g_vol', gvol_band_tt)

        # calculate the total area for each basin using the volume-area scaling relationship
        area_bsn_tt = self.vol_area_curve(param_m=param_m, param_n=param_n, vol=gvol_bsn_tt, cal_area=True)
        delta_area = area_bsn_tt - area_bsn_t0

        # Calculate the area change for each band within each basin
        area_loss = torch.clamp(-delta_area, min=0)  # [n_basin, n_mul_comp]
        area_gain = torch.clamp(delta_area, min=0)

        # Calculate the area change for each band within each basin
        # Note: here we assume that the area loss occurs from the lowest elevation band to the highest
        available_area_to_lose = area_band_t0.clone()
        # Use accumulated area to determine how much area can be lost from each band
        actual_loss_per_band = torch.zeros_like(area_band_t0)
        gain_target_indices = []
        for bsn_id, band_ids in enumerate(self.bsn_band_ids_dict.values()):
            # --- Area Loss Calculation ---
            # Area available to lose from low to high elevation bands
            basin_available_area = available_area_to_lose[band_ids]
            cumulative_available = torch.cumsum(basin_available_area, dim=0)
            # basin_total_loss_needed - area available below the current band
            basin_total_loss_needed = area_loss[bsn_id]
            band_loss_contribution = torch.clamp(
                basin_total_loss_needed - (cumulative_available - basin_available_area),
                min=torch.zeros_like(basin_available_area),
                max=basin_available_area)
            actual_loss_per_band[band_ids] = band_loss_contribution

            # --- Area Gain Target Determination ---
            zero_area_mask = (basin_available_area == 0)
            if torch.any(zero_area_mask.any(dim=1)):
                local_zero_indices = torch.where(zero_area_mask.any(dim=1))[0]
                highest_zero_idx_local = local_zero_indices[-1]
                gain_target_indices.append(band_ids[highest_zero_idx_local])
            else:
                gain_target_indices.append(band_ids[0])

        # --- Final Area Gain Calculation ---
        actual_gain_per_band = torch.zeros_like(area_band_t0)
        gain_target_indices = torch.tensor(gain_target_indices, device=self.device, dtype=torch.long)
        actual_gain_per_band = actual_gain_per_band.index_add(0, gain_target_indices, area_gain)
        # update area
        area_band_tt = area_band_t0 - actual_loss_per_band + actual_gain_per_band
        area_band_tt = torch.clamp(area_band_tt, min=0)

        # ------------- Redistribute glacier mass based on the new area -------------
        # Expand basin-level variables to band-level shape for calculation
        gvol_bsn_at_bands = gvol_bsn_tt[self.band_to_basin_map]
        area_bsn_at_bands = area_bsn_tt[self.band_to_basin_map]

        # Distribute total basin volume to each band proportional to its new area share
        gvol_band_new = gvol_bsn_at_bands * area_band_tt / (area_bsn_at_bands + self.eps)
        # Calculate the new glacier water equivalent (gwe) from the new volume and area
        gwe_band_new = gvol_band_new * 1e6 * self.rho_ice / ((area_band_tt * self.rho_w) + self.eps)
        # Ensure gwe is zero where area is zero
        gwe_band_new[area_band_tt <= self.eps] = 0

        return area_band_tt, area_bsn_tt, gvol_band_new, gwe_band_new


    def cal_bsn_params(self, param_band: Union[torch.Tensor, None], area_band: Union[torch.Tensor, None] = None,
                       area_bsn: Union[torch.Tensor, None] = None, area_weighted: bool = False):
        if param_band is None:
            return None

        if TORCH_SCATTER_AVAILABLE:
            if area_weighted:
                weighted_param = param_band * area_band
                param_bsn_sum = scatter_sum(weighted_param, self.band_to_basin_map, dim=0, dim_size=self.n_basins)
                param_bsn = param_bsn_sum / (area_bsn + self.eps)
            else:
                param_bsn = scatter_mean(param_band, self.band_to_basin_map, dim=0, dim_size=self.n_basins)
        else:
            if area_weighted:
                weighted_param = param_band * area_band
                param_bsn_sum = torch.zeros((self.n_basins, self.n_mul_comp), device=self.device).index_add(0,
                                                                                                             self.band_to_basin_map,
                                                                                                             weighted_param)
                param_bsn = param_bsn_sum / (area_bsn + self.eps)
            else:
                param_bsn_sum = torch.zeros((self.n_basins, self.n_mul_comp), device=self.device).index_add(0,
                                                                                                             self.band_to_basin_map,
                                                                                                             param_band)
                band_counts = torch.bincount(self.band_to_basin_map, minlength=self.n_basins).unsqueeze(1).float()
                param_bsn = param_bsn_sum / (band_counts + self.eps)

        return param_bsn

    def init_gwe(self, area_bsn: torch.Tensor, area_band: torch.Tensor, param_m: torch.Tensor,
                 param_n: torch.Tensor):
        """
        Initialize the glacier volume and snow water equivalent.
        """
        # initialize basin and band glacier volume
        gvol_bsn = self.vol_area_curve(param_m=param_m, param_n=param_n, area=area_bsn, cal_vol=True)
        gvol_bsn_at_bands = gvol_bsn[self.band_to_basin_map]
        area_bsn_at_bands = area_bsn[self.band_to_basin_map]
        # Distribute the basin glacier volume to each band based on the area ratio
        gvol_band = gvol_bsn_at_bands * area_band / (area_bsn_at_bands + self.eps)
        gwe_band = gvol_band * self.rho_ice / self.rho_w / (area_band + self.eps) * 1e6

        return gwe_band

    def init_swe(self, g_area_band: torch.Tensor, param_Asp: torch.Tensor, param_beta: torch.Tensor):
        """
        Initialize snow water equivalent based on the snow depth.
        """
        if self.snow_depth_t0 is not None:
            swe_band = (self.snow_depth_t0 / 10 + self.eps) ** param_beta * param_Asp * 10
        else:
            swe_band = torch.zeros_like(g_area_band)  # initialize the snow water equivalent

        return swe_band

    def trans_var_band2bsn(self, var_name: str, var_band: torch.Tensor, area_band: Union[torch.Tensor, None] = None,
                           area_bsn: Union[torch.Tensor, None] = None):
        """
        Transform the band variables to the basin variables based on the area ratio.
        """
        if TORCH_SCATTER_AVAILABLE:
            if var_name in ['s_vol', 'g_vol', 'g_area']:
                var_bsn = scatter_sum(var_band, self.band_to_basin_map, dim=0, dim_size=self.n_basins)
            else:
                var_band_sum = scatter_sum(var_band * area_band, self.band_to_basin_map, dim=0, dim_size=self.n_basins)
                var_bsn = var_band_sum / (area_bsn + self.eps)
        else:
            if var_name in ['s_vol', 'g_vol', 'g_area']:

                var_bsn = torch.zeros((self.n_basins, var_band.shape[1]), device=self.device).index_add(0,
                                                                                                         self.band_to_basin_map,
                                                                                                         var_band)
            else:
                assert area_band is not None and area_bsn is not None, 'Area must be provided'

                var_band_sum = torch.zeros((self.n_basins, var_band.shape[1]), device=self.device).index_add(0,
                                                                                                              self.band_to_basin_map,
                                                                                                              var_band * area_band)
                var_bsn = var_band_sum / (area_bsn + self.eps)
        return var_bsn

    def init_glac_shift_idx(self):
        # Build mapping: from upper band (i) to lower band (i-1)
        shift_out_idx = []
        shift_in_idx = []

        for band_ids in self.bsn_band_ids_dict.values():
            if len(band_ids) < 2:
                continue  # skip if there is only one band in the basin
            for i in range(1, len(band_ids)):
                upper_band = band_ids[i]  # the upper band
                lower_band = band_ids[i - 1]  # the lower band
                shift_out_idx.append(upper_band)
                shift_in_idx.append(lower_band)

        shift_out_idx = torch.tensor(shift_out_idx, device=self.device, dtype=torch.long)
        shift_in_idx = torch.tensor(shift_in_idx, device=self.device, dtype=torch.long)

        mask = torch.zeros((self.n_bands, self.n_mul_comp), device=self.device)
        mask[self.lowest_band_indices] = 1
        return shift_out_idx, shift_in_idx, mask

    def _init_upper_neighbor_map(self):
        """
        Pre-computes a map to find the upper neighbor of each band.
        This is used for the glacier advance process.
        """
        # Initialize map with -1 (no upper neighbor)
        upper_neighbor_map = torch.full((self.n_bands,), -1, dtype=torch.long, device=self.device)

        # Iterate through the lists of band IDs from the dictionary
        for band_ids_list in self.bsn_band_ids_dict.values():
            if len(band_ids_list) < 2:
                continue
            # Convert the Python list to a PyTorch tensor before using it
            band_ids = torch.tensor(band_ids_list, dtype=torch.long, device=self.device)
            # Now, perform the assignment using tensors on both sides
            # For each band in the basin (except the highest one), its upper neighbor is the next one in the list.
            # band_ids are sorted from lowest to highest elevation.
            upper_neighbor_map[band_ids[:-1]] = band_ids[1:]

        return upper_neighbor_map


    def forward(self, forc: torch.Tensor, forc_norm: torch.Tensor, attrs_norm: torch.Tensor, swe_band_t0: torch.Tensor,
                params_dict: dict, gve_band_t0: torch.Tensor, area_bsn_t0: torch.Tensor, area_band_t0: torch.Tensor,
                ts: pd.Timestamp):
        """
        :param forc: includes prec, tas, pet, rhu, wind, nRad, and doy, with the shape of (n_band, 7).
        :param forc_norm: normalized forcing data, with the shape of (n_band, 7).
        :param attrs_norm: band attributes, with the shape of (n_band, 10).
        :param swe_band_t0: initial snow water equivalent, with the shape of (n_band, n_mul_comp).
        :param area_bsn_t0: initial basin area, with the shape of (n_basin, n_mul_comp).
        :param gve_band_t0: initial band glacier water equivalent, with the shape of (n_band, n_mul_comp).
        :param area_band_t0: initial band area, with the shape of (n_band, n_mul_comp).
        :param ts: time stamps.
        """
        forc = forc.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_band, n_mul_comp, 7]
        forc_norm = forc_norm.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_band, n_mul_comp, 7]
        attrs_norm = attrs_norm.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_band, n_mul_comp, 10]

        # calculate snow module
        snow_band_sim = self.cal_snow(swe=swe_band_t0, forc=forc, forc_norm=forc_norm, attrs_norm=attrs_norm,
                                      area_band=area_band_t0, param_ts=params_dict['param_ts_glac'],
                                      param_tm=params_dict['param_tm_glac'], param_ds=params_dict['param_ds_glac'],
                                      param_ds6=params_dict['param_ds6_glac'],
                                      param_ds12=params_dict['param_ds12_glac'],
                                      param_snow2ice=params_dict['param_snow2ice_glac'],
                                      param_Asp=params_dict['param_Asp_glac'],
                                      param_beta=params_dict['param_beta_glac'])

        # calculate glacier module
        if self.gla_config['update_freq'] == 'd':  # daily update
            update_flag = True
        elif self.gla_config['update_freq'] == 'sm':  # semi-monthly update
            update_flag = ts.day == 1 or ts.day == 15
        elif self.gla_config['update_freq'] == 'm':
            update_flag = ts.day == 1
        elif self.gla_config['update_freq'] == 'y':
            update_flag = ts.month == 1 and ts.day == 1
        else:
            update_flag = False

        # calculate glacier module
        glacier_band_sim = self.cal_glacier(swe=swe_band_t0, gwe=gve_band_t0, snow2ice=snow_band_sim['s_snow2ice'],
                                            forc=forc, attrs_norm=attrs_norm, area_band=area_band_t0,
                                            param_tg=params_dict['param_tg_glac'],
                                            param_dg=params_dict['param_dg_glac'],
                                            param_dg6=params_dict['param_dg6_glac'],
                                            param_dg12=params_dict['param_dg12_glac'],
                                            param_rf=params_dict['param_rf_glac'], cal_shift_flag=update_flag)
        # update glacier area
        area_band_tt, area_bsn_tt, gvol_band_new, gwe_band_new = self.update_area(gvol_band_tt=glacier_band_sim['g_vol'],
                                                                                  area_band_t0=area_band_t0,
                                                                                  area_bsn_t0=area_bsn_t0,
                                                                                  param_m=params_dict['param_m_glac'],  # type: ignore
                                                                                  param_n=params_dict['param_n_glac'],  # type: ignore
                                                                                  update_flag=update_flag)
        # store the output variables
        sim_band = {**snow_band_sim, **glacier_band_sim}
        # If the update was performed, overwrite the simulation results with the new, redistributed mass and area values.
        if update_flag:
            sim_band['g_vol'] = gvol_band_new
            sim_band['g_we'] = gwe_band_new

        sim_band['g_area'] = area_band_tt
        sim_bsn = {'g_area': area_bsn_tt}

        return sim_band, sim_bsn


class MlpModules(nn.Module):
    """
    MLP networks to substitute for the empirical snow and glacier modules.
    """

    def __init__(self, in_features: int, hidden_size: Union[int, List[int]], out_features: int, dropout: float = 0.5,
                 unnorm_var_num: int = 1):
        super(MlpModules, self).__init__()
        self.unnorm_var_num = unnorm_var_num  # number of un-normalized variables
        layers = []
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        assert isinstance(hidden_size, list), 'hidden_size must be an integer or a list of integers'
        for i in range(len(hidden_size)):
            out = hidden_size[i]
            layers.append(nn.Linear(in_features, out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = out
        layers.append(nn.Linear(in_features, out_features))
        self.net = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(unnorm_var_num)  # Batch normalization for the un-normalized variable

    def forward(self, x):
        """
        :param x: consists of forcing and static attributes with a shape of (N, n_mul_comp, F).
        """
        if self.unnorm_var_num > 0:
            x_unnorm = x[:, :, 0:self.unnorm_var_num]  # Extract the un-normalized variable and add a dimension
            x_unnorm = self.bn(x_unnorm.permute(0, 2, 1)).permute(0, 2,
                                                                  1)  # Apply batch normalization and remove the added dimension
            x = torch.cat((x_unnorm, x[:, :, self.unnorm_var_num:]), dim=-1)  # Concatenate the normalized variable back
        out = self.net(x)
        return out


class LstmCellModules(nn.Module):
    def __init__(self, in_lstm: int, hid_lstm: int, out_lstm: int, dropout: float = 0.5, unnorm_var_num: int = 1,
                 device: Union[str, torch.device] = 'cpu'):
        super(LstmCellModules, self).__init__()
        self.device = device
        self.hid_lstm = hid_lstm
        self.unnorm_var_num = unnorm_var_num

        self.bn = nn.BatchNorm1d(unnorm_var_num)
        self.fc_in = nn.Sequential(
            nn.Linear(in_lstm, hid_lstm),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.LSTMCell = nn.LSTMCell(hid_lstm, hid_lstm, device=device)
        self.fc_Out = nn.Linear(hid_lstm, out_lstm)
        self.hidden = None

    def forward(self, x, hidden=None):
        """
        :param x: consists of forcing, reservoir storages at last timestep t-1, and attributes
                  shape can be [N, F] or [N, M, F]
        """
        if x.dim() == 3:
            N, M, F = x.shape
            x = x.reshape(N * M, F)
            reshape_back = True
        else:
            reshape_back = False

        if self.unnorm_var_num > 0:
            x_unnorm = x[:, :self.unnorm_var_num]
            x_unnorm = self.bn(x_unnorm)
            x_rest = x[:, self.unnorm_var_num:]
            x = torch.cat([x_unnorm, x_rest], dim=-1)

        x = self.fc_in(x)
        h_1, c_1 = self.LSTMCell(x, hidden)
        out = self.fc_Out(h_1)

        if reshape_back:
            out = out.view(N, M, -1)

        return out, (h_1, c_1)


class MlpParams(nn.Module):
    """  Only use FC layers to learn the static parameters from static attributes. """

    def __init__(self, in_size: int, hidden_size: Union[int, List[int]], out_size: int, dropout: float = 0.5):
        super(MlpParams, self).__init__()
        layers = []
        in_features = in_size
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        for i in range(len(hidden_size)):
            out_features = hidden_size[i]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = out_features
        layers.append(nn.Linear(in_features, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x, attrs_idx=8):
        """
        :param attrs_idx: index of static attributes in the input features.
        :param x: consists of forcing and static attributes, with a shape of (N, F) or (N, L, F),
        where attributes keep the same along dimension L.
        """
        if x.dim() == 2:
            x = x[:, attrs_idx:]
        elif x.dim() == 3:
            x = x[:, -1, attrs_idx:]
        else:
            raise ValueError('Invalid input dimension, expected 2 or 3, got {}'.format(x.dim()))
        out = self.net(x)
        return out


class ConvMlpParams(nn.Module):
    def __init__(self, n_attrs: int, n_forc: int, n_params: int, in_length: int, hidden_size: int,
                 n_conv_kernel: List[int], conv_kernel_size: List[int], stride: Union[List[int], None] = None,
                 pool_kernel_size: Union[List[int], None] = None, dropout: float = 0.5):
        """
        :param n_attrs: number of static attributes.
        :param n_forc: number of forcing variables.
        :param n_params: number of parameters to be learned.
        :param in_length: length of input sequence for convolutional layer.
        :param hidden_size: hidden size of the fully connected layer.
        :param n_conv_kernel: number of kernels for each convolutional layer.
        :param conv_kernel_size: kernel size for each convolutional layer.
        :param stride: stride for each convolutional layer.
        :param pool_kernel_size: kernel size for each pooling layer.
        :param dropout: dropout rate.
        """
        super(ConvMlpParams, self).__init__()
        n_layer = len(n_conv_kernel)
        self.conv_layer = nn.Sequential()
        in_channel = n_forc  # need to modify the hardcode: 4 for smap and 1 for FDC
        out_len = in_length
        for i in range(n_layer):
            conv_layer = CNN1dKernel(in_channel=in_channel, n_kernel=n_conv_kernel[i], kernel_size=conv_kernel_size[i],
                                     stride=stride[i])
            self.conv_layer.add_module('CnnLayer%d' % (i + 1), conv_layer)
            self.conv_layer.add_module('Relu%d' % (i + 1), nn.ReLU())
            self.conv_layer.add_module('dropout%d' % (i + 1), nn.Dropout(p=dropout))
            in_channel = n_conv_kernel[i]
            out_len = cal_conv_size(lin=out_len, kernel=conv_kernel_size[i], stride=stride[i])
            if pool_kernel_size[i] is not None:
                self.conv_layer.add_module('Pooling%d' % (i + 1), nn.MaxPool1d(pool_kernel_size[i]))
                out_len = cal_pool_size(in_length=out_len, kernel_size=pool_kernel_size[i])
        self.n_out = int(out_len * n_conv_kernel[-1])  # total CNN feature number after convolution
        in_fc = self.n_out + n_attrs
        self.fc = nn.Sequential(
            nn.Linear(in_fc, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_params)
        )

    def forward(self, x, attrs_idx=8):
        forcing, attrs = x[:, :, :attrs_idx - 1], x[:, :, attrs_idx:]  # -1 means excluding the doy
        encoded = self.conv_layer(forcing.permute(0, 2, 1))
        out = self.fc(torch.cat((encoded.squeeze(1), attrs[:, -1, :]), dim=-1))
        return out


class LstmMlpParams(nn.Module):
    """
    Use LSTM to extract additional features from historical forcing, and then feed them into MLP layers after
    concat with static attributes.
    """

    def __init__(self, in_lstm: int, hid_lstm: int, out_lstm: int, in_fc: int, hid_fc: Union[int, List[int]],
                 out_fc: int, dropout: float = 0.5, device: Union[str, torch.device] = 'cpu'):
        """
        :param in_lstm: input size of x fed into the embedding layer before LSTM
        :param hid_lstm: hidden size of LSTM
        :param out_lstm: output size of the FC layer following LSTM
        :param in_fc: input size of MLPs which equals to the sum of out_LSTM and number of static attributes
        :param hid_fc: hidden size of MLPs
        :param out_fc: the final output size of MLPs
        :param dropout: the dropout of fc layers
        """
        super(LstmMlpParams, self).__init__()
        self.device = device
        self.hid_lstm = hid_lstm
        if isinstance(hid_fc, int):
            self.hid_fc = [hid_fc]
        else:
            self.hid_fc = hid_fc

        self.fc_in = nn.Sequential(
            nn.Linear(in_lstm, hid_lstm),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.LSTM = LSTM(hid_lstm, hid_lstm, device=device)
        self.fc_out = nn.Sequential(
            nn.Linear(hid_lstm, out_lstm),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        mlp_layers = []
        in_features = in_fc
        for i in range(len(self.hid_fc)):
            out_features = self.hid_fc[i]
            mlp_layers.append(nn.Linear(in_features, out_features))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_features = out_features
        mlp_layers.append(nn.Linear(in_features, out_fc))
        self.MLPs = nn.Sequential(*mlp_layers)

    def forward(self, x, attrs_idx=8, hidden=None):
        """
        :param x:  forcing and attributes with the shape of [N_bands, seq_Len, F] where F = n_forc + n_attrs
        :param attrs_idx: the index of static attributes in the input features
        :param hidden: hidden state of LSTM
        """
        # feed forcing into lstm to extract features
        forcing, attrs = x[:, :, :attrs_idx - 1], x[:, :, attrs_idx:]  # -1 means excluding the doy
        x0 = self.fc_in(forcing)
        hidden = self.LSTM.init_hidden(x.shape[0]) if hidden is None else hidden
        x0, _ = self.LSTM(x0, hidden)
        x0 = self.fc_out(x0)
        # concat the extracted features with static attributes and feed them into MLPs
        x1 = torch.cat((x0[:, -1, :], attrs[:, -1, :]), dim=-1)
        out = self.MLPs(x1)
        return out


class LstmCellParams(nn.Module):
    def __init__(self, in_lstm: int, hid_lstm: int, out_lstm: int, dropout: float = 0.5, unnorm_var_num: int = 1,
                 device: Union[str, torch.device] = 'cpu'):
        super(LstmCellParams, self).__init__()
        self.device = device
        self.hid_lstm = hid_lstm
        self.unnorm_var_num = unnorm_var_num

        self.bn = nn.BatchNorm1d(unnorm_var_num)
        self.fc_in = nn.Sequential(
            nn.Linear(in_lstm, hid_lstm),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.LSTMCell = nn.LSTMCell(hid_lstm, hid_lstm, device=device)
        self.fc_Out = nn.Linear(hid_lstm, out_lstm)

    def forward(self, x, hidden=None):
        """
        :param x: consists of forcing, reservoir storages at last timestep t-1, and attributes with a shape of [N, F]
        """
        # x: [N, F], F = unnorm_var_num + num_norm_var
        if self.unnorm_var_num > 0:
            x_unnorm = x[:, :self.unnorm_var_num]
            x_unnorm = self.bn(x_unnorm)
            x_rest = x[:, self.unnorm_var_num:]
            x = torch.cat([x_unnorm, x_rest], dim=-1)
        x = self.fc_in(x)
        h_1, c_1 = self.LSTMCell(x, hidden)
        out = self.fc_Out(h_1)
        return out, (h_1, c_1)


class LstmParams(nn.Module):
    """
    Use LSTM to learn the static parameters from forcing and attributes.
    """

    def __init__(self, in_lstm: int, hid_lstm: int, out_lstm: int, dropout: float = 0.5,
                 device: Union[str, torch.device] = 'cpu'):
        super(LstmParams, self).__init__()
        self.device = device
        self.hid_lstm = hid_lstm
        self.fc_in = nn.Sequential(
            nn.Linear(in_lstm, hid_lstm),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.LSTM = LSTM(hid_lstm, hid_lstm, device=device)
        self.fc_out = nn.Sequential(
            nn.Linear(hid_lstm, out_lstm)
        )

    def forward(self, x, attrs_idx=8, hidden=None, mode='train', block_len=5000):
        """
        :param x: Tensor of shape [N, L, F]
        :param attrs_idx: Index to split attributes and remove DOY
        :param hidden: Optional hidden state
        :param mode: 'train' or 'eval'
        :param block_len: Number of timesteps per block in eval mode (e.g. 3650 for 10 years) to avoid memory issues
        """
        x = torch.cat((x[:, :, :attrs_idx - 1], x[:, :, attrs_idx:]), dim=-1)  # [N, L, F']
        N, L, _ = x.shape
        hidden = self.LSTM.init_hidden(N) if hidden is None else hidden

        if mode == 'eval':
            out_chunks = []
            h = hidden
            for start in range(0, L, block_len):
                end = min(start + block_len, L)
                x_block = x[:, start:end]  # [N, B, F']
                x_block = self.fc_in(x_block)
                out_block, h = self.LSTM(x_block, h)
                out_block = self.fc_out(out_block)
                out_chunks.append(out_block.detach())  # Detach to prevent autograd memory tracking

            out = torch.cat(out_chunks, dim=1)  # [N, L, D_out]

        else:
            x = self.fc_in(x)
            x, _ = self.LSTM(x, hidden)
            out = self.fc_out(x)

        return out


class CNN1dKernel(torch.nn.Module):
    def __init__(self, *, in_channel=1, n_kernel=3, kernel_size=3, stride=1, padding=0):
        super(CNN1dKernel, self).__init__()
        self.cnn1d = nn.Conv1d(in_channels=in_channel, out_channels=n_kernel, kernel_size=kernel_size, padding=padding,
                               stride=stride)

    def forward(self, x):
        output = F.relu(self.cnn1d(x))
        return output


def cal_conv_size(lin, kernel, stride, padding=0, dilation=1):
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def cal_pool_size(in_length, kernel_size, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernel_size
    out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(out_length)


class LSTM(nn.Module):
    def __init__(self, in_lstm: int, hid_lstm: int, device: Union[str, torch.device] = 'cpu'):
        super(LSTM, self).__init__()
        self.hidLSTM = hid_lstm
        self.device = device
        self.lstm = nn.LSTM(in_lstm, hid_lstm, device=device, batch_first=True)

    def forward(self, x, hidden=None):
        hidden = self.init_hidden(x.shape[0]) if hidden is None else hidden
        out = self.lstm(x, hidden)
        return out

    def init_hidden(self, bsz):
        # LSTM h and c
        h = torch.zeros((1, bsz, self.hidLSTM), dtype=torch.float32).to(self.device)
        c = torch.zeros((1, bsz, self.hidLSTM), dtype=torch.float32).to(self.device)
        return h, c
