from collections import defaultdict
from typing import Union, List
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_scatter import scatter_sum, scatter_mean
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False


class DPLGlacierModel(nn.Module):
    def __init__(self, bsn_band_ids_dict: dict, n_attrs: int, n_forc: int, snow_config: dict, gla_config: dict,
                 mul_comp_config: dict, nn_params: dict, glac_area_band_t0: torch.Tensor,
                 snow_depth_t0: torch.Tensor, dropout: float = 0.5, device: Union[str, torch.device] = 'cpu'):
        super(DPLGlacierModel, self).__init__()
        # define multiple components
        self.n_mul_comp = mul_comp_config['n_mul_comp']
        self.mul_comp_weights_method = mul_comp_config['weights_method']
        # define the area and delta year for each band to initialize the glacier area
        self.g_area_band_t0 = glac_area_band_t0.unsqueeze(1).expand(-1, self.n_mul_comp)

        # define the snow and glacier
        params_range = nn_params['params_range']
        self.glacier_model = GlacierDynCell(bsn_band_ids_dict=bsn_band_ids_dict, n_attrs=n_attrs,
                                            snow_config=snow_config, snow_depth_t0=snow_depth_t0,
                                            gla_config=gla_config, params_range=params_range,
                                            n_mul_comp=mul_comp_config['n_mul_comp'],
                                            dropout=dropout, device=device)
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
                spin_up_len: int, initial_state: dict = None, mode: str = 'train'):
        # store the output variables
        if mode == 'train':
            out_vars = {'band': ['s_we', 'g_we', 'g_area'],
                        'basin': ['s_depth', 's_cf', 's_vol', 'g_vol', 'g_area']}
        else:
            out_vars = {'band': ['s_we', 's_param_slm', 'g_we', 'g_area', 'g_param_slm', 'param'],
                        'basin': ['s_pr', 's_melt', 's_slm', 's_depth', 's_cf', 's_vol', 'g_melt', 'g_slm', 'g_vol',
                                  'g_area', 'param']}

        # learn static parameters using neural network
        x_params_nn = torch.cat((forc_norm, attrs_norm.unsqueeze(1).expand(-1, forc_norm.size(1), -1)), dim=-1)
        params = self.param_nn(x_params_nn[:, :spin_up_len, :])
        # reshape the parameters into (n_bands, n_mul_comp, n_params)
        params_tmp = params.reshape(params.size(0), self.n_mul_comp, -1)
        if self.mul_comp_weights_method == 'dPL' and self.n_mul_comp > 1:
            mul_comp_weights = F.softmax(params_tmp[:, :, -1], dim=1)  # different weights determined by the model
        else:
            mul_comp_weights = torch.full_like(params_tmp[:, :, -1], 1 / self.n_mul_comp)  # equal weights
        # store other physical parameters in a dictionary and rescale the parameters to the physical range
        param_dict = {param_name: torch.sigmoid(params_tmp[:, :, i]) for i, param_name in
                      enumerate(self.glacier_model.non_none_params)}
        param_dict = self.glacier_model.rescale_param_range(param_dict)
        # unpack parameters
        for i, param in enumerate(self.glacier_model.non_none_params):
            self.glacier_model.params[param] = param_dict[param]
        # transform the band parameters to the basin parameters
        for param, scale in self.glacier_model.params_scale.items():
            if scale == 'basin':
                self.glacier_model.params[param] = self.glacier_model.cal_bsn_params(self.glacier_model.params[param])

        # Initialize variables from the state dictionary
        initial_state = initial_state or {}
        g_area_band_t0 = initial_state.get('area_band')
        g_area_bsn_t0 = initial_state.get('area_bsn')
        gwe_band_t0 = initial_state.get('gwe_band')
        swe_band_t0 = initial_state.get('swe_band')

        # Expand 1D tensors to 2D if necessary (e.g., for multiple components)
        tensors_to_process = [g_area_band_t0, g_area_bsn_t0, gwe_band_t0, swe_band_t0]
        g_area_band_t0, g_area_bsn_t0, gwe_band_t0, swe_band_t0 = [
            t.unsqueeze(1).expand(-1, self.n_mul_comp) if t is not None and t.dim() == 1 else t
            for t in tensors_to_process
        ]

        # Set default values for core area variables if they were not provided
        if g_area_band_t0 is None or g_area_bsn_t0 is None:
            g_area_band_t0 = self.g_area_band_t0 if g_area_band_t0 is None else g_area_band_t0
            g_area_bsn_t0 = self.glacier_model.trans_var_band2bsn(var_name='g_area', var_band=g_area_band_t0)
        # Initialize optional states (swe, gwe) if they are None
        params = self.glacier_model.params
        if swe_band_t0 is None:
            swe_band_t0, _ = self.glacier_model.init_swe(g_area_band_t0=g_area_band_t0, swe_band_t0=None,
                                                         param_Asp=params['param_Asp'], param_beta=params['param_beta']) # type: ignore
        if gwe_band_t0 is None:
            gwe_band_t0 = self.glacier_model.init_gwe(area_bsn=g_area_bsn_t0, area_band=g_area_band_t0,
                                                      param_m=params['param_m'], param_n=params['param_n']) # type: ignore

        # for time loop to simulate the glacier dynamics
        out_band, out_bsn = defaultdict(list), defaultdict(torch.Tensor)
        for t_step in range(x_params_nn.size(1)):
            out_band_t, out_bsn_t = self.glacier_model(forc=forc[:, t_step, :],
                                                       forc_norm=forc_norm[:, t_step, :],
                                                       attrs_norm=attrs_norm,
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
            for k, v in self.glacier_model.params.items():
                if v is not None:
                    if self.glacier_model.params_scale[k] == 'band':
                        out_band[k] = v  # type: ignore
                    else:
                        out_bsn[k] = v  # type: ignore

        return out_band, out_bsn


class GlacierDynCell(nn.Module):
    def __init__(self, bsn_band_ids_dict: dict, n_attrs: int, snow_config: dict, gla_config: dict, n_mul_comp: int,
                 snow_depth_t0: torch.Tensor, params_range: dict, dropout: float = 0.5, device: Union[str, torch.device] = 'cpu'):
        super(GlacierDynCell, self).__init__()
        self.eps = 1e-6  # a small value to avoid division by zero
        self.device = device
        self.bsn_band_ids_dict = bsn_band_ids_dict  # dictionary of basin-band ids
        self.band_slope = gla_config['slope']

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

        self.snow_config = snow_config  # configuration for snow module
        self.gla_config = gla_config  # configuration for glacier module
        self.rho_ice = 0.85  # density of ice, g/cm3, Huggonnet et al., 2021
        self.rho_w = 1  # density of water, g/cm3
        self.n_mul_comp = n_mul_comp  # number of multiple components
        self.snow_depth_t0 = snow_depth_t0  # initial snow depth
        if self.snow_depth_t0 is not None:
            if self.snow_depth_t0.dim() == 1:
                self.snow_depth_t0 = self.snow_depth_t0.unsqueeze(1).expand(-1, self.n_mul_comp)
        # all parameters
        self.params_range = params_range
        param_names = params_range.keys()
        self.params = {param: None for param in param_names}
        self.params_scale = {param: 'band' if param not in ['param_m', 'param_n'] else 'basin' for param in param_names}
        self.non_none_params = self.get_not_none_params()

        # initialize neural networks
        if self.snow_config['swe2sd_nn']:
            self.swe2sd_nn = MlpModules(in_features=7 + n_attrs, hidden_size=self.snow_config['swe2sd_nn_hidden'],
                                        out_features=1, dropout=dropout, unnorm_var_num=1)

        if self.gla_config['glacier_shift']:
            self.shift_out_idx, self.shift_in_idx, self.lowest_band_mask = self.init_glac_shift_idx()
            if self.gla_config['glacier_shift_nn']:
                self.gla_shift_nn = MlpModules(in_features=2 + n_attrs,
                                               hidden_size=self.gla_config['glacier_shift_nn_hidden'],
                                               out_features=1, dropout=dropout, unnorm_var_num=2)

        self.upper_neighbor_map = self._init_upper_neighbor_map()


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
        Gs = 0.2 * nRad # subsurface heat flux, MJ/m2/day, 20% of net radiation following GLEAM4
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
            slm = torch.clamp(pet, max=swe-melt)
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

    def cal_glacier(self, swe, gwe, snow2ice, forc, forc_norm, attrs_norm, area_band, param_tg, param_dg=None,
                    param_dg6=None, param_dg12=None, param_rf=None, cal_shift_flag: bool = True):
        """
        :param area_band: glacier area of the elevation band, km2.
        :param swe: snow water equivalent, mm.
        :param gwe: initial glacier water equivalent, mm.
        :param snow2ice: snow transferring to ice, mm/d.
        :param forc: forcing data including prec, tas, pet, rhu, wind, nRad, and doy.
        :param forc_norm: normalized forcing data, with the shape of (n_band, n_mul_comp, 7).
        :param attrs_norm: glacier attributes.
        :param param_tg: temperature threshold for glacier melt, °C.
        :param param_dg: day-degree factor for glacier melt, mm/°C/day.
        :param param_dg6: day-degree factor for glacier melt on 21 June, mm/°C/day.
        :param param_dg12: day-degree factor for glacier melt on 21 December, mm/°C/day.
        :param param_rf: glacier flow rate factor, Pa-3 s-1
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
        Gs = 0.2 * nRad # subsurface heat flux, MJ/m2/day, 20% of net radiation following GLEAM4
        lambda_fusion = 0.334  # latent heat of fusion, MJ/kg
        melt_pot = torch.clamp((nRad - Gs) / self.rho_w / lambda_fusion, min=0)  # potential melt, mm/day
        melt[mask] = torch.clamp((ddf * (tas - param_tg))[mask], min=torch.zeros_like(gwe)[mask],
                                 max=torch.min(melt_pot[mask], gwe[mask]))

        # calculate glacier sublimation
        slm = torch.zeros_like(gwe)
        if self.gla_config['glacier_slm']:
            # calculate potential evapotranspiration using Penman formular
            pet = self.cal_Penman_ET(tas=tas, prs=prs, nRad=nRad, rhu=rhu, wins=wins, melt=melt)
            tmp_slm = torch.clamp(pet, max=gwe-melt)
            slm[mask] = tmp_slm[mask]  # only consider glacier sublimation when there is no snow

        # calculate net glacier shift
        if self.gla_config['glacier_shift'] and cal_shift_flag:
            H_glac = gwe * self.rho_w / self.rho_ice / 1000 # glacier thickness, m
            # Calculate the shift-out amount from each band as before
            if self.gla_config['glacier_shift_nn']:
                # Input: [n_band, n_mul_comp, 2 + n_attrs
                x = torch.cat((H_glac.unsqueeze(-1), area_band, attrs_norm), dim=-1)
                out = torch.sigmoid(self.gla_shift_nn(x)).squeeze(-1)
                frac = self.rescale_param_range({'param_flow_frac': out})['param_flow_frac']
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
        params_partition = ['param_ts']
        params_snowmelt = ['param_tm', 'param_ds6', 'param_ds12'] if self.snow_config['sinusoidal_ddf'] else [
            'param_tm', 'param_ds']
        params_snow2ice = ['param_snow2ice'] if self.snow_config['snow2ice'] else []
        if (self.snow_config['swe2sd'] and self.snow_config['swe2sd_nn'] is False) or self.snow_depth_t0 is not None:
            params_swe2sd = ['param_Asp', 'param_beta']
        else:
            params_swe2sd = []
        params_snow = params_partition + params_snowmelt + params_snow2ice + params_swe2sd

        # parameters for glacier module
        params_gla_melt = ['param_tg', 'param_dg6', 'param_dg12'] if self.gla_config['sinusoidal_ddf'] else ['param_tg',
                                                                                                             'param_dg']
        params_vol_area = ['param_m', 'param_n']

        params_gla_shift = ['param_rf'] if self.gla_config['glacier_shift'] and (
                    self.gla_config['glacier_shift_nn'] is False) else ['param_flow_frac']
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
        gvol_bsn = self.vol_area_curve(param_m=param_m, param_n=param_n, area=area_bsn, cal_vol=True)
        gvol_bsn_at_bands = gvol_bsn[self.band_to_basin_map]
        area_bsn_at_bands = area_bsn[self.band_to_basin_map]
        # Distribute the basin glacier volume to each band based on the area ratio
        gvol_band = gvol_bsn_at_bands * area_band / (area_bsn_at_bands + self.eps)
        gwe_band = gvol_band * self.rho_ice / self.rho_w / (area_band + self.eps) * 1e6

        return gwe_band

    def init_swe(self, g_area_band_t0: torch.Tensor, swe_band_t0: Union[torch.Tensor, None],
                 param_Asp: torch.Tensor, param_beta: torch.Tensor):
        """
        Initialize snow water equivalent based on the snow depth.
        """
        if swe_band_t0 is None:
            if self.snow_depth_t0 is not None:
                # swe = param_a * sd ** param_beta / rho_w, in which the units of swe and sd are cm.
                s_dep_band_t0 = self.snow_depth_t0
                swe_band_t0 = (s_dep_band_t0 / 10 + self.eps) ** param_beta * param_Asp * 10
                # swe_band = self.snow_depth_t0 * 0.3 # convert snow depth to snow water equivalent
            else:
                swe_band_t0 = torch.zeros_like(g_area_band_t0)   # initialize the snow water equivalent
                s_dep_band_t0 = (swe_band_t0 / 10 / param_Asp  + self.eps) ** (1 / param_beta) * 10
        else:
            s_dep_band_t0 = (swe_band_t0 / 10 / param_Asp + self.eps) ** (1 / param_beta) * 10
        svol_band_t0 = s_dep_band_t0 * g_area_band_t0 * 1e-6

        return swe_band_t0, svol_band_t0

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
                gve_band_t0: torch.Tensor, area_bsn_t0: torch.Tensor, area_band_t0: torch.Tensor, ts: pd.Timestamp):
        """
        :param forc: includes prec, tas, pet, rhu, wind, nRad, and doy, with the shape of (n_band, 8).
        :param forc_norm: normalized forcing data, with the shape of (n_band, 8).
        :param attrs_norm: band attributes, with the shape of (n_band, 10).
        :param swe_band_t0: initial snow water equivalent, with the shape of (n_band, n_mul_comp).
        :param area_bsn_t0: initial basin area, with the shape of (n_basin, n_mul_comp).
        :param gve_band_t0: initial band glacier water equivalent, with the shape of (n_band, n_mul_comp).
        :param area_band_t0: initial band area, with the shape of (n_band, n_mul_comp).
        :param ts: time stamps.
        """
        forc = forc.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_band, n_mul_comp, 8]
        forc_norm = forc_norm.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_band, n_mul_comp, 8]
        attrs_norm = attrs_norm.unsqueeze(1).expand(-1, self.n_mul_comp, -1)  # [n_band, n_mul_comp, 10]

        # calculate snow module
        snow_band_sim = self.cal_snow(swe=swe_band_t0, forc=forc, forc_norm=forc_norm, attrs_norm=attrs_norm,
                                      area_band=area_band_t0, param_ts=self.params['param_ts'],
                                      param_tm=self.params['param_tm'], param_ds=self.params['param_ds'],
                                      param_ds6=self.params['param_ds6'], param_ds12=self.params['param_ds12'],
                                      param_snow2ice=self.params['param_snow2ice'], param_Asp=self.params['param_Asp'],
                                      param_beta=self.params['param_beta'])

        # update glacier area
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
                                            forc=forc, forc_norm=forc_norm, attrs_norm=attrs_norm,
                                            area_band=area_band_t0, param_tg=self.params['param_tg'],
                                            param_dg=self.params['param_dg'], param_dg6=self.params['param_dg6'],
                                            param_dg12=self.params['param_dg12'], param_rf=self.params['param_rf'],
                                            cal_shift_flag=update_flag)
        # Call the updated area/mass redistribution function
        area_band_tt, area_bsn_tt, gvol_band_new, gwe_band_new = self.update_area(gvol_band_tt=glacier_band_sim['g_vol'],
                                                                                  area_band_t0=area_band_t0,
                                                                                  area_bsn_t0=area_bsn_t0,
                                                                                  param_m=self.params['param_m'],  # type: ignore
                                                                                  param_n=self.params['param_n'],  # type: ignore
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

    def __init__(self, in_features: int, hidden_size: Union[int, List[int]], out_features: int, dropout: float = 0.5, unnorm_var_num: int = 1):
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
        x_unnorm = x[:, :, 0:self.unnorm_var_num]  # Extract the un-normalized variable and add a dimension
        x_unnorm = self.bn(x_unnorm.permute(0, 2, 1)).permute(0, 2, 1)  # Apply batch normalization and remove the added dimension
        x = torch.cat((x_unnorm, x[:, :, self.unnorm_var_num:]), dim=-1)  # Concatenate the normalized variable back
        out = self.net(x)
        return out


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
        :param x: consists of forcing and static attributes, where attributes keep the same along dimension L,
            with a shape of (N, L, F).
        """
        x = x[:, -1, attrs_idx:]
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
        print()
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