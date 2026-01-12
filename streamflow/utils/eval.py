import torch
import logging
from collections import defaultdict
import pickle
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
from functools import reduce


class SaveEval:
    def __init__(self, loader_all, glac_model, rr_model, config, glac_loss_config, rr_loss_config, eval_vars):
        periods = config['data']['periods']
        for k, v in periods.items():
            if isinstance(v, list) and all(isinstance(item, str) for item in v):  # one period
                periods[k] = [pd.date_range(start=v[0], end=v[1], freq='D')]
            elif isinstance(v, list) and all(isinstance(item, list) for item in v):  # multiple periods
                periods[k] = [pd.date_range(start=period[0], end=period[1], freq='D') for period in v]
        self.train = reduce(pd.Index.union, periods['train'])
        self.valid = reduce(pd.Index.union, periods['valid'])
        self.test = reduce(pd.Index.union, periods['test'])
        self.glac_snow_scale, self.rr_snow_scale = glac_loss_config['snow_scale'], rr_loss_config['snow_scale']
        # run the model to get the simulation results
        self.sim = self.get_sim(loader=loader_all, glac_model=glac_model, rr_model=rr_model, config=config)
        # load the observed data
        self.obs = self.get_obs(glac_loss_config=glac_loss_config, rr_loss_config=rr_loss_config, eval_vars=eval_vars)
        # calculate metrics
        self.evaluate(eval_vars=eval_vars, glac_bsn_codes=glac_loss_config['bsn_code'],
                      rr_bsn_codes=rr_loss_config['bsn_code'], period='valid', only_gs_LAI=rr_loss_config['only_gs_LAI'])
        self.evaluate(eval_vars=eval_vars, glac_bsn_codes=glac_loss_config['bsn_code'],
                      rr_bsn_codes=rr_loss_config['bsn_code'], period='test', only_gs_LAI=rr_loss_config['only_gs_LAI'])

    def get_sim(self, loader, glac_model, rr_model, config):
        logging.info(f'Getting simulation results.')
        if os.path.exists(os.path.join(config['out'], 'sim.pkl')):
            with open(os.path.join(config['out'], 'sim.pkl'), 'rb') as f:
                sim = pickle.load(f)
        else:
            # load the model weights and set the model to evaluation mode
            glac_model.load_state_dict(torch.load(os.path.join(config['out'], 'model_glac.pt'), weights_only=True))
            glac_model.eval()
            rr_model.load_state_dict(torch.load(os.path.join(config['out'], 'model_rr.pt'), weights_only=True))
            rr_model.eval()
            # initialize the simulation results
            dyn_params = config['model']['rr_model']['nn_params']['dynamic']['params']
            glac_band, glac_bsn, rr_bsn, sim_time = defaultdict(list), defaultdict(list), defaultdict(list), []
            # run the model to get the simulation results
            rr_hidden_state = None
            with torch.no_grad():
                for i, (glac_inputs, bsn_inputs, time) in enumerate(loader):
                    logging.info(f'Running the {i+1}/{len(loader)} batch of data.')
                    # glacier model
                    spin_up_len = config['data']['spin_up_len'] if i == 0 else 0
                    ts = pd.to_datetime(time.squeeze(0))
                    glac_forc, glac_forc_norm, glac_attrs_norm, glac_area_t0 = glac_inputs
                    glac_forc, glac_forc_norm, glac_attrs_norm = glac_forc.squeeze(0), glac_forc_norm.squeeze(0), glac_attrs_norm.squeeze(0)
                    glac_band_tmp, glac_bsn_tmp = glac_model(forc=glac_forc, forc_norm=glac_forc_norm,
                                                             attrs_norm=glac_attrs_norm, glac_area_t0=glac_area_t0,
                                                             ts=ts, spin_up_len=spin_up_len, mode='eval')
                    # rainfall-runoff model
                    bsn_forc, bsn_forc_norm, bsn_attrs_norm, riv_attrs_norm = map(lambda x: x.squeeze(0), bsn_inputs)
                    rr_bsn_tmp = rr_model(forc=bsn_forc, forc_norm=bsn_forc_norm, bsn_attrs_norm=bsn_attrs_norm,
                                          riv_attrs_norm=riv_attrs_norm, glac_sim_bsn=glac_bsn_tmp,
                                          spin_up_len=spin_up_len, mode='eval', hidden_state=rr_hidden_state)
                    rr_hidden_state = rr_bsn_tmp['hidden_state'] if 'hidden_state' in rr_bsn_tmp else None
                    # skip the spin-up period
                    for sim_tmp, save_dict in zip([glac_band_tmp, glac_bsn_tmp, rr_bsn_tmp], [glac_band, glac_bsn, rr_bsn]):
                        for k, v in sim_tmp.items():
                            if k.startswith('param') and k not in dyn_params:
                                save_dict[k].append(v.detach().cpu().numpy())
                            elif k != 'hidden_state':
                                save_dict[k].append(v.detach().cpu().numpy()[:, spin_up_len:])
                    sim_time.append(ts[spin_up_len:])

            # stack the simulation results
            for save_dict in [glac_band, glac_bsn, rr_bsn]:
                for k, v in save_dict.items():
                    if k.startswith('param') and k not in dyn_params:
                        save_dict[k] = v
                    else:
                        save_dict[k] = np.concatenate(v, axis=1) # type: ignore
            sim_time = pd.to_datetime(np.concatenate(sim_time))
            sim = {'glac_band': glac_band, 'glac_bsn': glac_bsn, 'rr_bsn': rr_bsn, 'time': sim_time}
            # save the simulation results
            with open(os.path.join(config['out'], 'sim.pkl'), 'wb') as f:
                pickle.dump(sim, f)  # type: ignore
        return sim

    @staticmethod
    def get_glac_sim(loader, glac_model, config):
        glac_model.load_state_dict(torch.load(os.path.join(config['out'], 'model_glac.pt'), weights_only=True))
        spin_up_len = config['data']['spin_up_len']
        glac_model.eval()
        with torch.no_grad():
            for i, (glac_inputs, bsn_inputs, time) in enumerate(loader):
                # glacier model
                ts = pd.to_datetime(time.squeeze(0))
                glac_forc, glac_forc_norm, glac_attrs_norm, glac_area_t0 = glac_inputs
                del glac_inputs, time
                glac_forc, glac_forc_norm, glac_attrs_norm = (
                    glac_forc.squeeze(0), glac_forc_norm.squeeze(0), glac_attrs_norm.squeeze(0)
                )
                glac_band, glac_bsn = glac_model(forc=glac_forc, forc_norm=glac_forc_norm,
                                                 attrs_norm=glac_attrs_norm, glac_area_t0=glac_area_t0,
                                                 ts=ts, spin_up_len=spin_up_len, mode='eval')
                glac_band = {k: v.detach().cpu().numpy() for k, v in glac_band.items()}
                glac_bsn = {k: v.detach().cpu().numpy() for k, v in glac_bsn.items()}
                gc.collect()
                logging.info('Finished glacier model running')
        return glac_band, glac_bsn, ts

    @staticmethod
    def get_rr_sim(loader, rr_model, config, glac_bsn):
        glac_sim_bsn = {k: torch.tensor(v) for k, v in glac_bsn.items() if k in
                        ['g_area', 's_melt', 'g_melt', 's_pr', 's_slm', 'g_slm']}
        rr_model.load_state_dict(torch.load(os.path.join(config['out'], 'model_rr.pt'), weights_only=True))
        spin_up_len = config['data']['spin_up_len']
        rr_model.eval()
        with torch.no_grad():
            for i, (glac_inputs, bsn_inputs, time) in enumerate(loader):
                # glacier model
                # rainfall-runoff model
                bsn_forc, bsn_forc_norm, bsn_attrs_norm, riv_attrs_norm = map(lambda x: x.squeeze(0), bsn_inputs)
                rr_bsn = rr_model(forc=bsn_forc, forc_norm=bsn_forc_norm, bsn_attrs_norm=bsn_attrs_norm,
                                  riv_attrs_norm=riv_attrs_norm, glac_sim_bsn=glac_sim_bsn, spin_up_len=spin_up_len,
                                  mode='eval')
                rr_bsn = {k: v.detach().cpu().numpy() for k, v in rr_bsn.items()}
                logging.info('Finished rainfall runoff model running')
        return rr_bsn

    @staticmethod
    def get_obs(glac_loss_config, rr_loss_config, eval_vars):
        obs = dict()
        if 'glac_area' in eval_vars:
            glac_area_obs = pd.read_csv(glac_loss_config['glac_area_path'], dtype={'basin_id': str}, sep=r'\s+')
            glac_area_obs['date'] = pd.to_datetime(glac_area_obs['date'])
            glac_area_obs['basin_id'] = glac_area_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            # reorder the observed glacier area based on the given basin codes
            glac_area_obs['basin_id'] = pd.Categorical(glac_area_obs['basin_id'],
                                                       categories=glac_loss_config['bsn_code'], ordered=True)
            glac_area_obs = glac_area_obs.sort_values(by=['basin_id'])
            glac_area_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_area'] = glac_area_obs

        if 'glac_tvol' in eval_vars:
            # load the observed glacier volume during 2017-2018
            glac_vol_obs = pd.read_csv(glac_loss_config['glac_vol_path'], dtype={'basin_id': str}, sep=r'\s+')
            glac_vol_obs['basin_id'] = glac_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            glac_vol_obs['basin_id'] = pd.Categorical(glac_vol_obs['basin_id'],
                                                      categories=glac_loss_config['bsn_code'], ordered=True)
            glac_vol_obs = glac_vol_obs.sort_values(by=['basin_id'])
            glac_vol_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_vol'] = glac_vol_obs
            # set the date as 2018-1-1
            glac_vol_obs = glac_vol_obs['vol'].values

            # load the observed glacier volume change during 2000-2019
            glac_dvol_obs = pd.read_csv(glac_loss_config['glac_dvol_path'], index_col=0, sep=r'\s+')
            glac_dvol_obs.columns = [str(x).zfill(12) for x in glac_dvol_obs.columns]
            glac_dvol_obs = glac_dvol_obs.reindex(columns=glac_loss_config['bsn_code']) * 10 ** (-9) # convert m^3 to km^3
            obs['glac_dvol'] = glac_dvol_obs

            # calculate the glacier volume change during 2000-2019 based on the glac_dvol_obs and glac_vol_obs
            if 'Hugonnet' in glac_loss_config['glac_dvol_path']:
                dates = pd.date_range(start='2000-01-01', end='2018-12-31', freq='YS')
            else:
                dates = pd.date_range(start='1957-01-01', end='2018-12-31', freq='YS')
            dates = dates.append(pd.to_datetime(['2018-12-31', '2019-12-31']))
            # Create a DataFrame to store the gvol for each date
            glac_tvol_obs = pd.DataFrame(index=dates, columns=glac_dvol_obs.columns)
            # Set the gvol for 2018-1-1
            glac_tvol_obs.loc['2018-01-01'] = glac_vol_obs
            # Calculate the gvol for the specified dates
            for date in dates:
                if date < pd.to_datetime('2018-01-01'):
                    glac_tvol_obs.loc[date] = glac_tvol_obs.loc['2018-01-01'] - glac_dvol_obs.loc[date.year:2017].sum()
                elif date > pd.to_datetime('2018-01-01'):
                    glac_tvol_obs.loc[date] = glac_tvol_obs.loc['2018-01-01'] + glac_dvol_obs.loc['2018':date.year].sum()
            glac_tvol_obs[glac_tvol_obs < 0] = 0
            glac_tvol_obs[glac_dvol_obs.columns[glac_dvol_obs.isna().any()]] = np.nan
            obs['glac_tvol'] = glac_tvol_obs


        if 'glac_vol' in eval_vars and 'glac_vol' not in obs.keys():
            glac_vol_obs = pd.read_csv(glac_loss_config['glac_vol_path'], dtype={'basin_id': str}, sep=r'\s+')
            glac_vol_obs['basin_id'] = glac_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            glac_vol_obs['basin_id'] = pd.Categorical(glac_vol_obs['basin_id'],
                                                      categories=glac_loss_config['bsn_code'], ordered=True)
            glac_vol_obs = glac_vol_obs.sort_values(by=['basin_id'])
            glac_vol_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_vol'] = glac_vol_obs

        if 'glac_dvol' in eval_vars and 'glac_dvol' not in obs.keys():
            glac_dvol_obs = pd.read_csv(glac_loss_config['glac_dvol_path'], index_col=0, sep=r'\s+')
            glac_dvol_obs.columns = [str(x).zfill(12) for x in glac_dvol_obs.columns]
            glac_dvol_obs = glac_dvol_obs.reindex(columns=glac_loss_config['bsn_code']) * 10 ** (-9) # convert m^3 to km^3
            obs['glac_dvol'] = glac_dvol_obs

        # load the observed snow depth
        if 'glac_sdep' in eval_vars:
            glac_sdep_obs = pd.read_csv(glac_loss_config['glac_sdep_path'], index_col=0, parse_dates=True, sep=r'\s+')
            glac_sdep_obs.columns = [str(x).zfill(12) for x in glac_sdep_obs.columns]
            glac_sdep_obs = glac_sdep_obs.reindex(columns=glac_loss_config['bsn_code'])
            obs['glac_sdep'] = glac_sdep_obs

        if 'bsn_Q' in eval_vars:
            # read the observed streamflow data
            bsn_q_obs_daily = pd.read_excel(rr_loss_config['bsn_Q_path'], index_col=0, parse_dates=True, sheet_name='daily')
            # bsn_q_obs_daily.drop(columns=rr_loss_config['held_out_gauges'], inplace=True)
            bsn_q_obs_monthly = pd.read_excel(rr_loss_config['bsn_Q_path'], index_col=0, parse_dates=True, sheet_name='monthly')
            bsn_q_obs_monthly.index = bsn_q_obs_monthly.index.to_period('M')
            df_gauges = pd.read_excel(rr_loss_config['bsn_Q_path'], sheet_name='gauges')
            # filter stations and determine the idx of bsn for each station
            bsn_code = rr_loss_config['bsn_code']
            bsn_idx_daily, station_daily, bsn_idx_monthly, station_monthly = [], [], [], []
            for station in bsn_q_obs_daily.columns:
                basin_ids = df_gauges[df_gauges['Station'] == station]['BasinIds'].values[0].split(',')
                basin_idx = [bsn_code.index(basin_id) for basin_id in basin_ids if basin_id in bsn_code]
                if len(basin_idx) > 0:
                    bsn_idx_daily.append(basin_idx)
                    station_daily.append(station)
            for station in bsn_q_obs_monthly.columns:
                basin_ids = df_gauges[df_gauges['Station'] == station]['BasinIds'].values[0].split(',')
                basin_idx = [bsn_code.index(basin_id) for basin_id in basin_ids if basin_id in bsn_code]
                if len(basin_idx) > 0:
                    bsn_idx_monthly.append(basin_idx)
                    station_monthly.append(station)
            bsn_q_obs_daily, bsn_q_obs_monthly = bsn_q_obs_daily.loc[:, station_daily], bsn_q_obs_monthly.loc[:, station_monthly]

            obs['bsn_Q'] = {'daily': {'Q': bsn_q_obs_daily, 'idx': bsn_idx_daily},
                            'monthly': {'Q': bsn_q_obs_monthly, 'idx': bsn_idx_monthly}}

        if 'bsn_LAI' in eval_vars:
            bsn_LAI_obs = pd.read_csv(rr_loss_config['bsn_LAI_path'], index_col=0, parse_dates=True, sep=r'\s+')
            bsn_LAI_obs.columns = [str(x).zfill(12) for x in bsn_LAI_obs.columns]
            bsn_LAI_obs = bsn_LAI_obs.reindex(columns=rr_loss_config['bsn_code'])
            obs['bsn_LAI'] = bsn_LAI_obs

        if 'bsn_sdep' in eval_vars:
            bsn_sdep_obs = pd.read_csv(rr_loss_config['bsn_sdep_path'], index_col=0, parse_dates=True, sep=r'\s+')
            bsn_sdep_obs.columns = [str(x).zfill(12) for x in bsn_sdep_obs.columns]
            bsn_sdep_obs = bsn_sdep_obs.reindex(columns=rr_loss_config['bsn_code'])
            obs['bsn_sdep'] = bsn_sdep_obs

        return obs

    def evaluate(self, eval_vars, glac_bsn_codes, rr_bsn_codes, only_gs_LAI=False, period='test'):
        if period == 'train':
            t_range = self.train
        elif period == 'valid':
            t_range = self.valid
        else:
            t_range = self.test
        idx_time = self.sim['time'].get_indexer(t_range) # type: ignore

        if 'glac_area' in eval_vars and 'g_area' in self.sim['glac_bsn'].keys():
            glac_area_sim_df = pd.DataFrame(self.sim['glac_bsn']['g_area'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            obs_dates = self.obs['glac_area']['date']
            glac_area_sim = np.array([glac_area_sim_df.loc[date, glac_bsn_codes[i]] if date in glac_area_sim_df.index
                                      else np.nan for i, date in enumerate(obs_dates)])
            glac_area_obs = self.obs['glac_area'].loc[:, 'area'].values
            glac_area_obs, glac_area_sim = glac_area_obs[~np.isnan(glac_area_sim)], glac_area_sim[~np.isnan(glac_area_sim)]
            # calculate the metrics
            if len(glac_area_obs) == 0:
                logging.info(f'No observed glacier area data in {period} period.')
            else:
                r, nse, rmse, kge, pbias = self.eval_fn(true=glac_area_obs, pred=glac_area_sim, cal_dim=0)
                logging.info(f'For glacier area in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'glac_tvol' in eval_vars and 'g_vol' in self.sim['glac_bsn'].keys():
            glac_tvol_obs_df = self.obs['glac_tvol']
            glac_tvol_sim_df = pd.DataFrame(self.sim['glac_bsn']['g_vol'][:, idx_time].T,
                                       index=t_range, columns=glac_bsn_codes)
            glac_tvol_sim_df = glac_tvol_sim_df.loc[glac_tvol_sim_df.index.isin(glac_tvol_obs_df.index)]
            glac_tvol_obs_df = glac_tvol_obs_df.loc[glac_tvol_obs_df.index.isin(glac_tvol_sim_df.index)]
            # gvol_obs, gvol_sim = gvol_obs_df.values.T.astype(float), gvol_sim_df.values.T
            glac_tvol_obs = glac_tvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            glac_tvol_sim = glac_tvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=glac_tvol_obs, pred=glac_tvol_sim, cal_dim=0)
            logging.info(f'For glacier volume in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'glac_dvol' in eval_vars and 'g_vol' in self.sim['glac_bsn'].keys():
            glac_dvol_obs_df = self.obs['glac_dvol']
            glac_dvol_obs_df = glac_dvol_obs_df.loc[glac_dvol_obs_df.index.isin([t.year for t in t_range])]
            glac_tvol_sim_df = pd.DataFrame(self.sim['glac_bsn']['g_vol'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            glac_dvol_sim_df = pd.DataFrame(index=glac_dvol_obs_df.index, columns=glac_bsn_codes)
            for year in glac_dvol_sim_df.index: # type: ignore
                glac_dvol_sim_df.loc[year] = (glac_tvol_sim_df.loc[pd.to_datetime(f'{year}-12-31')] -
                                              glac_tvol_sim_df.loc[pd.to_datetime(f'{year}-1-1')])
            # dgvol_obs, dgvol_sim = dgvol_obs_df.values.T, glac_dvol_sim_df.values.T
            dgvol_obs = glac_dvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            dgvol_sim = glac_dvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=dgvol_obs, pred=dgvol_sim, cal_dim=0)
            logging.info(f'For glacier volume change in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'glac_vol' in eval_vars and 2017 in range(t_range.year) and 'g_vol' in self.sim['glac_bsn'].keys(): # type: ignore
            glac_vol_obs_df = self.obs['glac_vol']
            glac_vol_obs = glac_vol_obs_df.loc[:, 'vol'].values
            glac_tvol_sim_df = pd.DataFrame(self.sim['glac_bsn']['g_vol'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            glac_vol_sim = glac_tvol_sim_df[glac_tvol_sim_df.index.year == 2017].values.mean(axis=0).T
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=glac_vol_obs, pred=glac_vol_sim, cal_dim=0)
            logging.info(f'For 2017 glacier volume in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'glac_sdep' in eval_vars and 's_depth' in self.sim['glac_bsn'].keys():
            s_depth_obs_df = self.obs['glac_sdep'].loc[self.obs['glac_sdep'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(self.sim['glac_bsn']['s_depth'][:, idx_time].T,
                                          index=t_range, columns=glac_bsn_codes)
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            # aggregate the observed and simulated snow depth to monthly scale
            if self.glac_snow_scale == 'monthly':
                s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
                s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
            else:
                s_depth_obs = s_depth_obs_df.values.T
                s_depth_sim = s_depth_sim_df.values.T
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1)
            logging.info(f'For glacier snow depth in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'bsn_Q' in eval_vars and 'Qriver' in self.sim['rr_bsn'].keys():
            q_obs_daily = self.obs['bsn_Q']['daily']['Q'].loc[self.obs['bsn_Q']['daily']['Q'].index.isin(t_range)] # type: ignore
            non_nan_count = np.sum(~np.isnan(q_obs_daily.values), axis=0)  # type: ignore
            valid_bsn_idx = np.nonzero(non_nan_count > 365)[0]
            if len(valid_bsn_idx) > 0:
                q_obs_daily = q_obs_daily.values[:, valid_bsn_idx].T
                q_sim = self.sim['rr_bsn']['Qriver'][:, idx_time]
                idx_bsn = [self.obs['bsn_Q']['daily']['idx'][idx] for idx in valid_bsn_idx]
                q_sim_daily = np.stack([q_sim[idx, :].sum(axis=0) for idx in idx_bsn])
                r, nse, rmse, kge, pbias = self.eval_fn(true=q_obs_daily, pred=q_sim_daily, cal_dim=1)
                logging.info(f'For daily streamflow in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

            t_range_monthly = t_range.to_period('M') # type: ignore
            q_obs_monthly = self.obs['bsn_Q']['monthly']['Q'].loc[t_range_monthly.unique()]
            ts_mask = [(t_range.month == index.month) & (t_range.year == index.year) for index in q_obs_monthly.index] # type: ignore
            non_nan_count = np.sum(~np.isnan(q_obs_monthly.values), axis=0)
            valid_bsn_idx = np.nonzero(non_nan_count > 24)[0]
            if len(valid_bsn_idx) > 0:
                q_obs_monthly = q_obs_monthly.values[:, valid_bsn_idx].T
                q_sim = self.sim['rr_bsn']['Qriver'][:, idx_time]
                idx_bsn = [self.obs['bsn_Q']['monthly']['idx'][idx] for idx in valid_bsn_idx]
                q_sim = np.stack([q_sim[idx, :].sum(axis=0) for idx in idx_bsn]) * 86400 # convert m3/s to m3/d
                q_sim_monthly = np.stack([q_sim[:, mask].sum(axis=1) for mask in ts_mask], axis=1) / 10**8
                r, nse, rmse, kge, pbias = self.eval_fn(true=q_obs_monthly, pred=q_sim_monthly, cal_dim=1)
                logging.info(f'For monthly streamflow in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'bsn_LAI' in eval_vars and 'LAI' in self.sim['rr_bsn'].keys():
            lai_obs_df = self.obs['bsn_LAI'].loc[self.obs['bsn_LAI'].index.isin(t_range)]
            lai_sim = self.sim['rr_bsn']['LAI'][:, idx_time]
            idx_lst = np.where(t_range.isin(lai_obs_df.index))[0]
            lai_sim_resample = np.stack([np.mean(lai_sim[:, start:end], axis=1) for start, end
                                   in zip(idx_lst[:-1], idx_lst[1:])], axis=1)
            lai_obs_resample = lai_obs_df.values.T[:, :-1]
            nan_mask = np.isnan(lai_obs_resample).any(axis=1)
            lai_sim_resample, lai_obs_resample = lai_sim_resample[~nan_mask], lai_obs_resample[~nan_mask]
            if only_gs_LAI:
                idx_lst_gs = np.where((t_range.isin(lai_obs_df.index)) & (t_range.month >= 5) & (t_range.month <= 10))[0] # type: ignore
                mask = np.where(np.isin(idx_lst, idx_lst_gs))[0]
                lai_sim_cal, lai_obs_cal = lai_sim_resample[:, mask], lai_obs_resample[:, mask]
            else:
                lai_sim_cal, lai_obs_cal = lai_sim_resample, lai_obs_resample
            r, nse, rmse, kge, pbias = self.eval_fn(true=lai_obs_cal, pred=lai_sim_cal, cal_dim=1)
            logging.info(f'For LAI in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'bsn_sdep' in eval_vars and 'sdep' in self.sim['rr_bsn'].keys():
            s_depth_obs_df = self.obs['bsn_sdep'].loc[self.obs['bsn_sdep'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(self.sim['rr_bsn']['sdep'][:, idx_time].T,
                                          index=t_range, columns=rr_bsn_codes)
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            # aggregate the observed and simulated snow depth to monthly scale
            if self.rr_snow_scale == 'monthly':
                s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
                s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
            else:
                s_depth_obs = s_depth_obs_df.values.T
                s_depth_sim = s_depth_sim_df.values.T
            # calculate the metrics
            no_snow = (s_depth_obs < 1) | np.isnan(s_depth_obs)  # [n, m]
            valid_mask = (no_snow.mean(axis=1) < 0.8) & (np.nanmean(s_depth_obs, axis=1) > 1) & (
                    np.nanmax(s_depth_obs, axis=1) > 10)
            r, nse, rmse, kge, pbias = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1, valid_mask=valid_mask)
            logging.info(f'For basin snow depth in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')


    @staticmethod
    def eval_fn(true: np.ndarray, pred: np.ndarray, cal_dim: int = 0, cal_flv: bool = False, valid_mask: np.ndarray = None):
        # check the dimensions of true and pred
        if len(true.shape) == 1 and len(pred.shape) == 1:
            true = true.reshape(1, -1)
            pred = pred.reshape(1, -1)
            cal_dim = 1
        assert true.ndim == 2 and pred.ndim == 2, 'The dimensions of true and pred should be 1 or 2.'
        # make sure the dtype is float
        all_nan_mask = np.isnan(true).all(axis=cal_dim, keepdims=True)
        true, pred = true.astype(float), pred.astype(float)
        pred[np.isnan(true)] = np.nan
        # Calculate mean along the specified dimension
        true_mean = np.nanmean(true, axis=cal_dim, keepdims=True)
        pred_mean = np.nanmean(pred, axis=cal_dim, keepdims=True)

        # Calculate r
        r_num = np.nansum((pred - pred_mean) * (true - true_mean), axis=cal_dim, keepdims=True)
        r_den = np.sqrt(np.nansum((pred - pred_mean) ** 2, axis=cal_dim, keepdims=True) *
                        np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True))
        r_den[r_den == 0] = 1e-5 # avoid division by zero
        r = r_num / r_den
        r = np.where(all_nan_mask, np.nan, r)  # set r to nan where all values are nan

        # Calculate NSE
        nse_num = np.nansum((pred - true) ** 2, axis=cal_dim, keepdims=True)
        nse_den = np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True)
        nse_den[nse_den == 0] = 1e-5 # avoid division by zero
        nse = 1 - nse_num / nse_den
        nse = np.where(all_nan_mask, np.nan, nse)  # set nse to nan where all values are nan

        # Calculate RMSE
        rmse = np.sqrt(np.nanmean((pred - true) ** 2, axis=cal_dim, keepdims=True))

        # Calculate KGE components
        true_mean[true_mean == 0] = 1e-5
        alpha = pred_mean / true_mean
        beta_num = np.nanstd(pred, axis=cal_dim, keepdims=True)
        beta_den = np.nanstd(true, axis=cal_dim, keepdims=True)
        beta_den[beta_den == 0] = 1e-5 # avoid division by zero
        beta = beta_num / beta_den
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # calculate the percent bias
        pbias_num = np.nanmean(pred - true, axis=cal_dim, keepdims=True)
        pbias_den = np.nanmean(true, axis=cal_dim, keepdims=True)
        pbias_den[pbias_den == 0] = 1e-5 # avoid division by zero
        pbias = pbias_num / pbias_den * 100

        if valid_mask is not None:
            r, nse, rmse, kge, pbias = r[valid_mask], nse[valid_mask], rmse[valid_mask], kge[valid_mask], pbias[valid_mask]
        r_mean = np.nanmean(r)
        nse_mean = np.nanmean(nse)
        rmse_mean = np.nanmean(rmse)
        kge_mean = np.nanmean(kge)
        pbias_mean = np.nanmean(pbias)

        return r_mean, nse_mean, rmse_mean, kge_mean, pbias_mean

class EvalSingle:  # for single model evaluation
    def __init__(self, folder: str, eval_vars: list[str], q_scale='daily'):
        self.q_scale = q_scale
        # load the simulation results
        with open(f'{folder}/sim.pkl', 'rb') as f:
            self.sim = pickle.load(f)
        # get the training, validation and test periods
        with open(f'{folder}/config.json', 'rb') as f:
            config = json.load(f)
        periods = config['data']['periods']
        for k, v in periods.items():
            if isinstance(v, list) and all(isinstance(item, str) for item in v):  # one period
                periods[k] = [pd.date_range(start=v[0], end=v[1], freq='D')]
            elif isinstance(v, list) and all(isinstance(item, list) for item in v):  # multiple periods
                periods[k] = [pd.date_range(start=period[0], end=period[1], freq='D') for period in v]
        self.train = reduce(pd.Index.union, periods['train'])
        self.valid = reduce(pd.Index.union, periods['valid'])
        self.test = reduce(pd.Index.union, periods['test'])
        self.held_out_gauges = config['train']['rr_loss']['held_out_gauges'] if 'held_out_gauges' in config['train'][
            'rr_loss'] else []

        # get band codes and bsn codes
        self.glac_band_codes, self.glac_bsn_codes, self.rr_bsn_codes = self.get_basin_codes(config)

        # get the observed data
        glac_loss_config = config['train']['glac_loss']
        glac_loss_config['bsn_code'] = self.glac_bsn_codes
        glac_loss_config['band_code'] = self.glac_band_codes
        rr_loss_config = config['train']['rr_loss']
        rr_loss_config['bsn_code'] = self.rr_bsn_codes
        self.glac_snow_scale, self.rr_snow_scale = glac_loss_config['snow_scale'], rr_loss_config['snow_scale']
        self.obs = self.get_obs(glac_loss_config=glac_loss_config, rr_loss_config=rr_loss_config, eval_vars=eval_vars)

        # get the metrics and dataset
        metrics, ds = dict(), dict()
        for period in ['train', 'valid', 'test']:
            ds[period], metrics[period] = self.evaluate(eval_vars=eval_vars,
                                                        glac_bsn_codes=glac_loss_config['bsn_code'],
                                                        rr_bsn_codes=rr_loss_config['bsn_code'],
                                                        only_gs_LAI=rr_loss_config['only_gs_LAI'],
                                                        period=period)
        self.metrics, self.ds = metrics, ds

    def get_basin_codes(self, config):
        sim_bsn_head = config['data']['sim_bsn_head']

        # get the glacier band codes and basin codes
        path = self.cal_abs_path(os.path.join(config['data']['glac_forc_dir'], 'forc_band.pkl'))
        forcing = pickle.load(open(path, 'rb'))
        if sim_bsn_head != 'all':
            glac_band_codes = [k for k in forcing.keys() if any(k.startswith(prefix) for prefix in sim_bsn_head)]
        else:
            glac_band_codes = list(forcing.keys())
        glac_basin_codes = sorted(list(set([band_code.split('_')[0] for band_code in glac_band_codes])))

        # get the basin codes for rainfall-runoff model
        path = self.cal_abs_path(os.path.join(config['data']['bsn_forc_dir'], 'forc_basins.pkl'))
        forcing = pickle.load(open(path, 'rb'))
        if sim_bsn_head != 'all':
            rr_bsn_codes = [k for k in forcing.keys() if any(k.startswith(prefix) for prefix in sim_bsn_head)]
        else:
            rr_bsn_codes = list(forcing.keys())

        return glac_band_codes, glac_basin_codes, rr_bsn_codes

    def cal_abs_path(self, relative_path):
        def find_project_root(current_path, marker_file):
            while not os.path.isfile(os.path.join(current_path, marker_file)):
                parent_path = os.path.dirname(current_path)
                if parent_path == current_path:
                    return None
                current_path = parent_path
            return current_path
        cwd = os.getcwd() # current working directory
        project_root = find_project_root(cwd, 'main.py') # project directory

        # calculate the relative path to cwd
        relative_path = Path(relative_path)
        absolute_path = (project_root / relative_path).resolve()

        return absolute_path

    def get_obs(self, glac_loss_config, rr_loss_config, eval_vars):
        obs = dict()
        if 'glac_area' in eval_vars:
            glac_area_path = self.cal_abs_path(glac_loss_config['glac_area_path'],)
            glac_area_obs = pd.read_csv(glac_area_path, dtype={'basin_id': str}, sep=r'\s+')
            glac_area_obs['date'] = pd.to_datetime(glac_area_obs['date'])
            glac_area_obs['basin_id'] = glac_area_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            # reorder the observed glacier area based on the given basin codes
            glac_area_obs['basin_id'] = pd.Categorical(glac_area_obs['basin_id'],
                                                       categories=glac_loss_config['bsn_code'], ordered=True)
            glac_area_obs = glac_area_obs.sort_values(by=['basin_id'])
            glac_area_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_area'] = glac_area_obs

        if 'glac_tvol' in eval_vars:
            # load the observed glacier volume during 2017-2018
            glac_vol_path = self.cal_abs_path(glac_loss_config['glac_vol_path'])
            glac_vol_obs = pd.read_csv(glac_vol_path, dtype={'basin_id': str}, sep=r'\s+')
            glac_vol_obs['basin_id'] = glac_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            glac_vol_obs['basin_id'] = pd.Categorical(glac_vol_obs['basin_id'],
                                                      categories=glac_loss_config['bsn_code'], ordered=True)
            glac_vol_obs = glac_vol_obs.sort_values(by=['basin_id'])
            glac_vol_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_vol'] = glac_vol_obs
            # set the date as 2018-1-1
            glac_vol_obs = glac_vol_obs['vol'].values

            # load the observed glacier volume change during 2000-2019
            glac_dvol_path = self.cal_abs_path(glac_loss_config['glac_dvol_path'])
            glac_dvol_obs = pd.read_csv(glac_dvol_path, index_col=0, sep=r'\s+')
            glac_dvol_obs.columns = [str(x).zfill(12) for x in glac_dvol_obs.columns]
            glac_dvol_obs = glac_dvol_obs.reindex(columns=glac_loss_config['bsn_code']) * 10 ** (-9) # convert m^3 to km^3
            obs['glac_dvol'] = glac_dvol_obs

            # calculate the glacier volume change during 2000-2019 based on the glac_dvol_obs and glac_vol_obs
            if 'Hugonnet' in glac_loss_config['glac_dvol_path']:
                dates = pd.date_range(start='2000-01-01', end='2018-12-31', freq='YS')
            else:
                dates = pd.date_range(start='1957-01-01', end='2018-12-31', freq='YS')
            dates = dates.append(pd.to_datetime(['2018-12-31', '2019-12-31']))
            # Create a DataFrame to store the gvol for each date
            glac_tvol_obs = pd.DataFrame(index=dates, columns=glac_dvol_obs.columns)
            # Set the gvol for 2018-1-1
            glac_tvol_obs.loc['2018-01-01'] = glac_vol_obs
            # Calculate the gvol for the specified dates
            for date in dates:
                if date < pd.to_datetime('2018-01-01'):
                    glac_tvol_obs.loc[date] = glac_tvol_obs.loc['2018-01-01'] - glac_dvol_obs.loc[date.year:2017].sum()
                elif date > pd.to_datetime('2018-01-01'):
                    glac_tvol_obs.loc[date] = glac_tvol_obs.loc['2018-01-01'] + glac_dvol_obs.loc['2018':date.year].sum()
            glac_tvol_obs[glac_tvol_obs < 0] = 0
            glac_tvol_obs[glac_dvol_obs.columns[glac_dvol_obs.isna().any()]] = np.nan
            obs['glac_tvol'] = glac_tvol_obs


        if 'glac_vol' in eval_vars and 'glac_vol' not in obs.keys():
            glac_vol_path = self.cal_abs_path(glac_loss_config['glac_vol_path'])
            glac_vol_obs = pd.read_csv(glac_vol_path, dtype={'basin_id': str}, sep=r'\s+')
            glac_vol_obs['basin_id'] = glac_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            glac_vol_obs['basin_id'] = pd.Categorical(glac_vol_obs['basin_id'],
                                                      categories=glac_loss_config['bsn_code'], ordered=True)
            glac_vol_obs = glac_vol_obs.sort_values(by=['basin_id'])
            glac_vol_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_vol'] = glac_vol_obs

        if 'glac_dvol' in eval_vars and 'glac_dvol' not in obs.keys():
            glac_dvol_path = self.cal_abs_path(glac_loss_config['glac_dvol_path'])
            glac_dvol_obs = pd.read_csv(glac_dvol_path, index_col=0, sep=r'\s+')
            glac_dvol_obs.columns = [str(x).zfill(12) for x in glac_dvol_obs.columns]
            glac_dvol_obs = glac_dvol_obs.reindex(columns=glac_loss_config['bsn_code']) * 10 ** (-9) # convert m^3 to km^3
            obs['glac_dvol'] = glac_dvol_obs

        # load the observed snow depth
        if 'glac_sdep' in eval_vars:
            glac_sdep_path = self.cal_abs_path(glac_loss_config['glac_sdep_path'])
            glac_sdep_obs = pd.read_csv(glac_sdep_path, index_col=0, parse_dates=True, sep=r'\s+')
            glac_sdep_obs.columns = [str(x).zfill(12) for x in glac_sdep_obs.columns]
            glac_sdep_obs = glac_sdep_obs.reindex(columns=glac_loss_config['bsn_code'])
            obs['glac_sdep'] = glac_sdep_obs


        if 'bsn_Q' in eval_vars:
            # read the observed streamflow data
            bsn_Q_path = self.cal_abs_path(rr_loss_config['bsn_Q_path'])
            bsn_q_obs_daily = pd.read_excel(bsn_Q_path, index_col=0, parse_dates=True, sheet_name='daily')
            bsn_q_obs_monthly = pd.read_excel(bsn_Q_path, index_col=0, parse_dates=True, sheet_name='monthly')
            bsn_q_obs_monthly.index = bsn_q_obs_monthly.index.to_period('M')
            df_gauges = pd.read_excel(bsn_Q_path, sheet_name='gauges')
            # filter stations and determine the idx of bsn for each station
            bsn_code = rr_loss_config['bsn_code']
            bsn_idx_daily, station_daily, bsn_idx_monthly, station_monthly = [], [], [], []
            for station in bsn_q_obs_daily.columns:
                basin_ids = df_gauges[df_gauges['Station'] == station]['BasinIds'].values[0].split(',')
                basin_idx = [bsn_code.index(basin_id) for basin_id in basin_ids if basin_id in bsn_code]
                if len(basin_idx) > 0:
                    bsn_idx_daily.append(basin_idx)
                    station_daily.append(station)
            for station in bsn_q_obs_monthly.columns:
                basin_ids = df_gauges[df_gauges['Station'] == station]['BasinIds'].values[0].split(',')
                basin_idx = [bsn_code.index(basin_id) for basin_id in basin_ids if basin_id in bsn_code]
                if len(basin_idx) > 0:
                    bsn_idx_monthly.append(basin_idx)
                    station_monthly.append(station)
            bsn_q_obs_daily, bsn_q_obs_monthly = bsn_q_obs_daily.loc[:, station_daily], bsn_q_obs_monthly.loc[:, station_monthly]

            obs['bsn_Q'] = {'daily': {'Q': bsn_q_obs_daily, 'idx': bsn_idx_daily},
                            'monthly': {'Q': bsn_q_obs_monthly, 'idx': bsn_idx_monthly}}

        if 'bsn_LAI' in eval_vars:
            bsn_LAI_path = self.cal_abs_path(rr_loss_config['bsn_LAI_path'])
            bsn_LAI_obs = pd.read_csv(bsn_LAI_path, index_col=0, parse_dates=True, sep=r'\s+')
            bsn_LAI_obs.columns = [str(x).zfill(12) for x in bsn_LAI_obs.columns]
            bsn_LAI_obs = bsn_LAI_obs.reindex(columns=rr_loss_config['bsn_code'])
            obs['bsn_LAI'] = bsn_LAI_obs

        if 'bsn_sdep' in eval_vars:
            bsn_sdep_path = self.cal_abs_path(rr_loss_config['bsn_sdep_path'])
            bsn_sdep_obs = pd.read_csv(bsn_sdep_path, index_col=0, parse_dates=True, sep=r'\s+')
            bsn_sdep_obs.columns = [str(x).zfill(12) for x in bsn_sdep_obs.columns]
            bsn_sdep_obs = bsn_sdep_obs.reindex(columns=rr_loss_config['bsn_code'])
            # df = bsn_sdep_obs.copy()  # df: [time, basin]
            # no_snow = (df < 1) | df.isna()  # [time, basin]
            # valid_mask = (no_snow.mean(axis=0) < 0.8) & (df.mean(axis=0, skipna=True) > 1)
            # obs['bsn_sdep'] = bsn_sdep_obs.loc[:, valid_mask]
            obs['bsn_sdep'] = bsn_sdep_obs


        return obs

    def evaluate(self, eval_vars, glac_bsn_codes, rr_bsn_codes, only_gs_LAI=False, period='test'):
        if period == 'train':
            t_range = self.train
        elif period == 'valid':
            t_range = self.valid
        else:
            t_range = self.test
        idx_time = self.sim['time'].get_indexer(t_range)

        metrics, ds = dict(), dict()
        if 'glac_area' in eval_vars and 'g_area' in self.sim['glac_bsn'].keys():
            glac_area_sim_df = pd.DataFrame(self.sim['glac_bsn']['g_area'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            obs_dates = self.obs['glac_area']['date']
            glac_area_sim = np.array([glac_area_sim_df.loc[date, glac_bsn_codes[i]] if date in glac_area_sim_df.index
                                      else np.nan for i, date in enumerate(obs_dates)])
            glac_area_obs = self.obs['glac_area'].loc[:, 'area'].values
            obs_dates = obs_dates[~np.isnan(glac_area_sim)]
            basin_ids = self.obs['glac_area'].loc[:, 'basin_id'].values[~np.isnan(glac_area_sim)]
            glac_area_obs, glac_area_sim = glac_area_obs[~np.isnan(glac_area_sim)], glac_area_sim[~np.isnan(glac_area_sim)]
            ds['glac_area'] = pd.DataFrame({'basin_id': basin_ids, 'obs': glac_area_obs.astype(float), 'sim': glac_area_sim.astype(float)},
                                           index=obs_dates)
            # calculate the metrics
            if len(glac_area_obs) == 0:
                logging.info(f'No observed glacier area data in {period} period.')
            else:
                r, nse, rmse, kge, pbias = self.eval_fn(true=glac_area_obs, pred=glac_area_sim, cal_dim=0)
                metrics['glac_area'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'glac_tvol' in eval_vars and 'g_vol' in self.sim['glac_bsn'].keys():
            glac_tvol_obs_df = self.obs['glac_tvol']
            glac_tvol_sim_df = pd.DataFrame(self.sim['glac_bsn']['g_vol'][:, idx_time].T,
                                       index=t_range, columns=glac_bsn_codes)
            glac_tvol_sim_df = glac_tvol_sim_df.loc[glac_tvol_sim_df.index.isin(glac_tvol_obs_df.index)]
            glac_tvol_obs_df = glac_tvol_obs_df.loc[glac_tvol_obs_df.index.isin(glac_tvol_sim_df.index)]
            # gvol_obs, gvol_sim = gvol_obs_df.values.T.astype(float), gvol_sim_df.values.T
            glac_tvol_obs = glac_tvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            glac_tvol_sim = glac_tvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            ds['glac_tvol'] = {'sim': glac_tvol_sim_df, 'obs': glac_tvol_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=glac_tvol_obs, pred=glac_tvol_sim, cal_dim=0)
            metrics['glac_tvol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'glac_dvol' in eval_vars and 'g_vol' in self.sim['glac_bsn'].keys():
            glac_dvol_obs_df = self.obs['glac_dvol']
            glac_dvol_obs_df = glac_dvol_obs_df.loc[glac_dvol_obs_df.index.isin([t.year for t in t_range])]
            glac_tvol_sim_df = pd.DataFrame(self.sim['glac_bsn']['g_vol'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            glac_dvol_sim_df = pd.DataFrame(index=glac_dvol_obs_df.index, columns=glac_bsn_codes)
            for year in glac_dvol_sim_df.index: # type: ignore
                glac_dvol_sim_df.loc[year] = (glac_tvol_sim_df.loc[pd.to_datetime(f'{year}-12-31')] -
                                              glac_tvol_sim_df.loc[pd.to_datetime(f'{year}-1-1')])
            # dgvol_obs, dgvol_sim = dgvol_obs_df.values.T, glac_dvol_sim_df.values.T
            dgvol_obs = glac_dvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            dgvol_sim = glac_dvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            ds['glac_dvol'] = {'sim': glac_dvol_sim_df, 'obs': glac_dvol_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=dgvol_obs, pred=dgvol_sim, cal_dim=0)
            metrics['glac_dvol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'glac_vol' in eval_vars and 2017 in t_range.year and 'g_vol' in self.sim['glac_bsn'].keys(): # type: ignore
            glac_vol_obs_df = self.obs['glac_vol']
            glac_vol_obs = glac_vol_obs_df.loc[:, 'vol'].values
            glac_tvol_sim_df = pd.DataFrame(self.sim['glac_bsn']['g_vol'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            glac_vol_sim = glac_tvol_sim_df[glac_tvol_sim_df.index.year == 2017].values.mean(axis=0).T
            ds['glac_vol'] = pd.DataFrame({'obs': glac_vol_obs, 'sim': glac_vol_sim}, index=glac_bsn_codes)
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=glac_vol_obs, pred=glac_vol_sim, cal_dim=0)
            metrics['glac_vol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'glac_sdep' in eval_vars and 's_depth' in self.sim['glac_bsn'].keys():
            s_depth_obs_df = self.obs['glac_sdep'].loc[self.obs['glac_sdep'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(self.sim['glac_bsn']['s_depth'][:, idx_time].T,
                                          index=t_range, columns=glac_bsn_codes)
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            # aggregate the observed and simulated snow depth to monthly scale
            if self.glac_snow_scale == 'monthly':
                s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
                s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
                ds['glac_sdep'] = {'sim': s_depth_sim_df.resample('ME').mean(), 'obs': s_depth_obs_df.resample('ME').mean()}
            else:
                s_depth_obs = s_depth_obs_df.values.T
                s_depth_sim = s_depth_sim_df.values.T
                ds['glac_sdep'] = {'sim': s_depth_sim_df, 'obs': s_depth_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1)
            metrics['glac_sdep'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}


        if 'bsn_Q' in eval_vars and 'Qriver' in self.sim['rr_bsn'].keys():
            if self.q_scale == 'daily':
                q_obs_daily_df = self.obs['bsn_Q']['daily']['Q'].loc[self.obs['bsn_Q']['daily']['Q'].index.isin(t_range)] # type: ignore
                non_nan_count = np.sum(~np.isnan(q_obs_daily_df.values), axis=0)  # type: ignore
                valid_bsn_idx = np.nonzero(non_nan_count > 365)[0]
                if len(valid_bsn_idx) > 0:
                    q_obs_daily_df = q_obs_daily_df.loc[:, q_obs_daily_df.columns[valid_bsn_idx]]
                    q_obs_daily = q_obs_daily_df.values.T
                    q_sim = self.sim['rr_bsn']['Qriver'][:, idx_time]
                    idx_bsn = [self.obs['bsn_Q']['daily']['idx'][idx] for idx in valid_bsn_idx]
                    q_sim_daily = np.stack([q_sim[idx, :].sum(axis=0) for idx in idx_bsn])
                    ds['bsn_Q_d'] = {'sim': pd.DataFrame(q_sim_daily.T, index=q_obs_daily_df.index, columns=q_obs_daily_df.columns),
                                     'obs': q_obs_daily_df}
                    r, nse, rmse, kge, pbias, flv, fhv = self.eval_fn(true=q_obs_daily, pred=q_sim_daily, cal_dim=1, cal_flv=True)
                    metrics['bsn_Q_d'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias, 'flv': flv, 'fhv': fhv}
            else:
                # 1. Retrieve daily observations within the target time range
                q_obs_daily_subset = self.obs['bsn_Q']['daily']['Q'].loc[self.obs['bsn_Q']['daily']['Q'].index.isin(t_range)]  # type: ignore

                # 2. Check data validity
                # Filter basins that have enough valid DAILY data points (e.g., > 365 days)
                # to ensure the derived monthly means are statistically meaningful.
                non_nan_count = np.sum(~np.isnan(q_obs_daily_subset.values), axis=0)  # type: ignore
                valid_bsn_idx = np.nonzero(non_nan_count > 365)[0]

                if len(valid_bsn_idx) > 0:
                    # 3. Filter valid basins for daily observations
                    q_obs_daily_subset = q_obs_daily_subset.loc[:, q_obs_daily_subset.columns[valid_bsn_idx]]
                    # 4. Retrieve and process simulated data (Daily Scale first)
                    # Get the corresponding basin indices
                    idx_bsn = [self.obs['bsn_Q']['daily']['idx'][idx] for idx in valid_bsn_idx]
                    # Retrieve raw simulated daily streamflow (all time steps)
                    q_sim_raw = self.sim['rr_bsn']['Qriver']  # shape: [n_basins, n_total_days]
                    # Spatial Aggregation: Sum sub-basins to the gauge location
                    q_sim_daily_aggr = np.stack([q_sim_raw[idx, :].sum(axis=0) for idx in idx_bsn])
                    # Create a DataFrame using the full simulation time index
                    df_sim_daily_all = pd.DataFrame(q_sim_daily_aggr.T, index=self.sim['time'],
                                                    columns=q_obs_daily_subset.columns)
                    # 5. Align Simulation with Observation (Time & NaN Consistency)
                    # First, extract simulation data for the exact dates present in the observation subset
                    df_sim_daily_matched = df_sim_daily_all.reindex(q_obs_daily_subset.index)
                    df_sim_daily_matched = df_sim_daily_matched.where(q_obs_daily_subset.notna())

                    # 6. Resample to Monthly Scale
                    # Calculate monthly means (ignoring NaNs). 'ME' = Month End frequency.
                    q_obs_mon_df = q_obs_daily_subset.resample('ME').mean()
                    q_sim_mon_df = df_sim_daily_matched.resample('ME').mean()

                    # Align indices strictly (in case resampling created extra months at edges)
                    q_sim_mon_df = q_sim_mon_df.reindex(q_obs_mon_df.index)

                    # 7. Prepare for Evaluation
                    q_obs_mon = q_obs_mon_df.values.T
                    q_sim_mon = q_sim_mon_df.values.T

                    # Store the monthly DataFrames for reference
                    ds['bsn_Q_m'] = {'sim': q_sim_mon_df, 'obs': q_obs_mon_df}

                    # 8. Calculate Metrics
                    # Note: self.eval_fn should be able to handle NaNs (or ignore them) in the input arrays.
                    r, nse, rmse, kge, pbias, flv, fhv = self.eval_fn(true=q_obs_mon, pred=q_sim_mon, cal_dim=1,
                                                                      cal_flv=True)
                    metrics['bsn_Q_m'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias, 'flv': flv,
                                          'fhv': fhv}

            t_range_monthly = t_range.to_period('M') # type: ignore
            q_obs_monthly_df = self.obs['bsn_Q']['monthly']['Q'].loc[t_range_monthly.unique()]
            ts_mask = [(t_range.month == index.month) & (t_range.year == index.year) for index in q_obs_monthly_df.index] # type: ignore
            non_nan_count = np.sum(~np.isnan(q_obs_monthly_df.values), axis=0)
            valid_bsn_idx = np.nonzero(non_nan_count > 24)[0]
            if len(valid_bsn_idx) > 0:
                q_obs_monthly_df = q_obs_monthly_df.loc[:, q_obs_monthly_df.columns[valid_bsn_idx]]
                q_obs_monthly = q_obs_monthly_df.values.T
                q_sim = self.sim['rr_bsn']['Qriver'][:, idx_time]
                idx_bsn = [self.obs['bsn_Q']['monthly']['idx'][idx] for idx in valid_bsn_idx]
                q_sim = np.stack([q_sim[idx, :].sum(axis=0) for idx in idx_bsn]) * 86400 # convert m3/s to m3/d
                q_sim_monthly = np.stack([q_sim[:, mask].sum(axis=1) for mask in ts_mask], axis=1) / 10**8
                ds['bsn_Q_m'] = {'sim': pd.DataFrame(q_sim_monthly.T, index=q_obs_monthly_df.index, columns=q_obs_monthly_df.columns),
                                 'obs': q_obs_monthly_df}
                r, nse, rmse, kge, pbias, flv, fhv = self.eval_fn(true=q_obs_monthly, pred=q_sim_monthly, cal_dim=1, cal_flv=True)
                metrics['bsn_Q_m'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias, 'flv': flv, 'fhv': fhv}

        if 'bsn_LAI' in eval_vars and 'LAI' in self.sim['rr_bsn'].keys():
            lai_obs_df = self.obs['bsn_LAI'].loc[self.obs['bsn_LAI'].index.isin(t_range)]
            lai_sim = self.sim['rr_bsn']['LAI'][:, idx_time]
            idx_lst = np.where(t_range.isin(lai_obs_df.index))[0]
            lai_sim_resample = np.stack([np.mean(lai_sim[:, start:end], axis=1) for start, end
                                   in zip(idx_lst[:-1], idx_lst[1:])], axis=1)
            lai_obs_resample = lai_obs_df.values.T[:, :-1]
            nan_mask = np.isnan(lai_obs_resample).any(axis=1)
            lai_sim_resample, lai_obs_resample = lai_sim_resample[~nan_mask], lai_obs_resample[~nan_mask]
            ds['bsn_LAI'] = {'sim': pd.DataFrame(lai_sim_resample.T, index=lai_obs_df.index[:-1],
                                                 columns=lai_obs_df.columns[~nan_mask]),
                             'obs': pd.DataFrame(lai_obs_resample.T, index=lai_obs_df.index[:-1],
                                                 columns=lai_obs_df.columns[~nan_mask])}
            if only_gs_LAI:
                idx_lst_gs = np.where((t_range.isin(lai_obs_df.index)) & (t_range.month >= 5) & (t_range.month <= 10))[0] # type: ignore
                mask = np.where(np.isin(idx_lst, idx_lst_gs))[0]
                lai_sim_cal, lai_obs_cal = lai_sim_resample[:, mask], lai_obs_resample[:, mask]
            else:
                lai_sim_cal, lai_obs_cal = lai_sim_resample, lai_obs_resample

            r, nse, rmse, kge, pbias = self.eval_fn(true=lai_obs_cal, pred=lai_sim_cal, cal_dim=1)
            metrics['bsn_LAI'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'bsn_sdep' in eval_vars and 'sdep' in self.sim['rr_bsn'].keys():
            s_depth_obs_df = self.obs['bsn_sdep'].loc[self.obs['bsn_sdep'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(self.sim['rr_bsn']['sdep'][:, idx_time].T,
                                          index=t_range, columns=rr_bsn_codes)
            s_depth_sim_df = s_depth_sim_df[s_depth_obs_df.columns]
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            # aggregate the observed and simulated snow depth to monthly scale
            if self.rr_snow_scale == 'monthly':
                s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
                s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
                ds['bsn_sdep'] = {'sim': s_depth_sim_df.resample('ME').mean(), 'obs': s_depth_obs_df.resample('ME').mean()}
            else:
                s_depth_obs = s_depth_obs_df.values.T
                s_depth_sim = s_depth_sim_df.values.T
                ds['bsn_sdep'] = {'sim': s_depth_sim_df, 'obs': s_depth_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1, var='snow')
            metrics['bsn_sdep'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}


        return ds, metrics

    @ staticmethod
    def eval_fn(true: np.ndarray, pred: np.ndarray, cal_dim: int = 0, cal_flv: bool = False, var=None):
        # check the dimensions of true and pred
        if len(true.shape) == 1 and len(pred.shape) == 1:
            true = true.reshape(1, -1)
            pred = pred.reshape(1, -1)
            cal_dim = 1
        assert true.ndim == 2 and pred.ndim == 2, 'The dimensions of true and pred should be 1 or 2.'
        # make sure the dtype is float
        if var == 'snow':
            no_snow = (true < 1) | np.isnan(true)  # [n, m]
            valid_mask = (no_snow.mean(axis=1) < 0.8) & (np.nanmean(true, axis=1) > 1) & (
                    np.nanmax(true, axis=1) > 10)
            true[~valid_mask, :] = np.nan
        all_nan_mask = np.isnan(true).all(axis=cal_dim, keepdims=True)
        true, pred = true.astype(float), pred.astype(float)
        pred[np.isnan(true)] = np.nan
        # Calculate mean along the specified dimension
        true_mean = np.nanmean(true, axis=cal_dim, keepdims=True)
        pred_mean = np.nanmean(pred, axis=cal_dim, keepdims=True)

        # Calculate r
        r_num = np.nansum((pred - pred_mean) * (true - true_mean), axis=cal_dim, keepdims=True)
        r_den = np.sqrt(np.nansum((pred - pred_mean) ** 2, axis=cal_dim, keepdims=True) *
                        np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True))
        r_den[r_den == 0] = 1e-5 # avoid division by zero
        r = r_num / r_den
        r = np.where(all_nan_mask, np.nan, r)  # set r to nan where all values are nan

        # Calculate NSE
        nse_num = np.nansum((pred - true) ** 2, axis=cal_dim, keepdims=True)
        nse_den = np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True)
        nse_den[nse_den == 0] = 1e-5 # avoid division by zero
        nse = 1 - nse_num / nse_den
        nse = np.where(all_nan_mask, np.nan, nse)  # set nse to nan where all values are nan

        # Calculate RMSE
        rmse = np.sqrt(np.nanmean((pred - true) ** 2, axis=cal_dim, keepdims=True))

        # Calculate KGE components
        true_mean[true_mean == 0] = 1e-5
        alpha = pred_mean / true_mean
        beta_num = np.nanstd(pred, axis=cal_dim, keepdims=True)
        beta_den = np.nanstd(true, axis=cal_dim, keepdims=True)
        beta_den[beta_den == 0] = 1e-5 # avoid division by zero
        beta = beta_num / beta_den
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # calculate the percent bias
        pbias_num = np.nanmean(pred - true, axis=cal_dim, keepdims=True)
        pbias_den = np.nanmean(true, axis=cal_dim, keepdims=True)
        pbias_den[pbias_den == 0] = 1e-5 # avoid division by zero
        pbias = pbias_num / pbias_den * 100

        # FLV the low flows bias bottom 30%, FHV the high flows bias top 2%
        if cal_flv:
            flv_list = []
            fhv_list = []
            for i in range(pred.shape[0]):
                pred_clean = pred[i, ~np.isnan(pred[i])]
                true_clean = true[i, ~np.isnan(true[i])]
                if len(pred_clean) == 0 or len(true_clean) == 0:
                    flv_list.append(np.nan)
                    fhv_list.append(np.nan)
                    continue
                pred_sort = np.sort(pred_clean)
                target_sort = np.sort(true_clean)
                index_low = round(0.3 * len(pred_sort))
                low_pred = pred_sort[:index_low]
                low_target = target_sort[:index_low]
                flv = (np.sum(low_pred - low_target, keepdims=True) / np.sum(low_target, keepdims=True)) * 100
                flv_list.append(flv)
                index_high = round(0.98 * len(pred_sort))
                high_pred = pred_sort[index_high:]
                high_target = target_sort[index_high:]
                fhv = (np.sum(high_pred - high_target, keepdims=True) / np.sum(high_target, keepdims=True)) * 100
                fhv_list.append(fhv)
            flv = np.array(flv_list)
            fhv = np.array(fhv_list)
            return r, nse, rmse, kge, pbias, flv, fhv
        else:
            return r, nse, rmse, kge, pbias


class EvalEnsemble:  # for ensemble model evaluation
    def __init__(self, folder: str, eval_vars: list[str], q_scale='daily'):
        # load the simulation results
        model_dict = {f[f.find('seed')+len('seed')+1:f.find('seed')+len('seed')+3]: f for f in os.listdir(f'{folder}') if f.find('seed') != -1}
        self.models = list(model_dict.keys())
        self.train, self.valid, self.test = None, None, None  # the training, validation and test periods
        self.glac_band_codes, self.glac_bsn_codes, self.rr_bsn_codes = None, None, None  # the band codes and basin codes
        self.glac_snow_scale, self.rr_snow_scale  = None, None
        self.obs = None # the observed data
        self.held_out_gauges = None
        self.q_scale = q_scale
        # loop over the models to evaluate
        sim = defaultdict(lambda: defaultdict(dict))
        metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        ds = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for model, path in model_dict.items():
            sim_tmp, metrics_tmp, ds_tmp = self.eval_model(f'{folder}/{path}', eval_vars)
            sim[model], metrics[model], ds[model] = sim_tmp, metrics_tmp, ds_tmp
        # calculate the ensemble mean simulation
        for var in ['glac_band', 'glac_bsn', 'rr_bsn']:
            for key in list(sim.values())[0][var].keys():
                sim_mean = np.mean(np.concatenate([np.expand_dims(sim[model][var][key], 0) for
                                                   model in model_dict.keys()]), axis=0)
                sim['mean'][var][key] = sim_mean
        sim['mean']['time'] = sim[list(model_dict.keys())[0]]['time']
        for model, path in model_dict.items():
            sim[model] = np.nan
        # calculate the ensemble mean metrics
        for period in ['train', 'valid', 'test']:
            ds['mean'][period], metrics['mean'][period] = self.evaluate(sim=sim['mean'],
                                                                        eval_vars=eval_vars,
                                                                        glac_bsn_codes=self.glac_bsn_codes,
                                                                        rr_bsn_codes=self.rr_bsn_codes,
                                                                        period=period)
        self.sim, self.metrics, self.ds = sim, metrics, ds

    def eval_model(self, folder: str, eval_vars: list[str]):
        with open(f'{folder}/sim.pkl', 'rb') as f:
            sim = pickle.load(f)
        # get the training, validation and test periods
        with open(f'{folder}/config.json', 'rb') as f:
            config = json.load(f)
        if self.glac_bsn_codes is None:
            self.glac_band_codes, self.glac_bsn_codes, self.rr_bsn_codes = self.get_basin_codes(config)
        if self.train is None:
            periods = config['data']['periods']
            for k, v in periods.items():
                if isinstance(v, list) and all(isinstance(item, str) for item in v):  # one period
                    periods[k] = [pd.date_range(start=v[0], end=v[1], freq='D')]
                elif isinstance(v, list) and all(isinstance(item, list) for item in v):  # multiple periods
                    periods[k] = [pd.date_range(start=period[0], end=period[1], freq='D') for period in v]
            self.train = reduce(pd.Index.union, periods['train'])
            self.valid = reduce(pd.Index.union, periods['valid'])
            self.test = reduce(pd.Index.union, periods['test'])
        if self.held_out_gauges is None:
            self.held_out_gauges = config['train']['rr_loss']['held_out_gauges'] if 'held_out_gauges' in config['train']['rr_loss'] else []
        # get the observed data
        if self.obs is None:
            glac_loss_config = config['train']['glac_loss']
            glac_loss_config['bsn_code'] = self.glac_bsn_codes
            glac_loss_config['band_code'] = self.glac_band_codes
            rr_loss_config = config['train']['rr_loss']
            rr_loss_config['bsn_code'] = self.rr_bsn_codes
            if self.glac_snow_scale is None:
                self.glac_snow_scale, self.rr_snow_scale = glac_loss_config['snow_scale'], rr_loss_config['snow_scale']
            self.obs = self.get_obs(glac_loss_config=glac_loss_config, rr_loss_config=rr_loss_config,
                                    eval_vars=eval_vars)
        # get the metrics and dataset
        metrics, ds = dict(), dict()
        for period in ['valid', 'test']:
            ds[period], metrics[period] = self.evaluate(sim=sim, eval_vars=eval_vars,
                                                        glac_bsn_codes=self.glac_bsn_codes,
                                                        rr_bsn_codes=self.rr_bsn_codes,
                                                        period=period)
        metrics, ds = metrics, ds
        return sim, metrics, ds

    def cal_abs_path(self, relative_path):
        def find_project_root(current_path, marker_file):
            while not os.path.isfile(os.path.join(current_path, marker_file)):
                parent_path = os.path.dirname(current_path)
                if parent_path == current_path:
                    return None
                current_path = parent_path
            return current_path
        cwd = os.getcwd() # current working directory
        project_root = find_project_root(cwd, 'main.py') # project directory

        # calculate the relative path to cwd
        relative_path = Path(relative_path)
        absolute_path = (project_root / relative_path).resolve()

        return absolute_path

    def get_basin_codes(self, config):
        sim_bsn_head = config['data']['sim_bsn_head']

        # get the glacier band codes and basin codes
        path = self.cal_abs_path(os.path.join(config['data']['glac_forc_dir'], 'forc_band.pkl'))
        forcing = pickle.load(open(path, 'rb'))
        if sim_bsn_head != 'all':
            glac_band_codes = [k for k in forcing.keys() if any(k.startswith(prefix) for prefix in sim_bsn_head)]
        else:
            glac_band_codes = list(forcing.keys())
        glac_basin_codes = sorted(list(set([band_code.split('_')[0] for band_code in glac_band_codes])))

        # get the basin codes for rainfall-runoff model
        path = self.cal_abs_path(os.path.join(config['data']['bsn_forc_dir'], 'forc_basins.pkl'))
        forcing = pickle.load(open(path, 'rb'))
        if sim_bsn_head != 'all':
            rr_bsn_codes = [k for k in forcing.keys() if any(k.startswith(prefix) for prefix in sim_bsn_head)]
        else:
            rr_bsn_codes = list(forcing.keys())

        return glac_band_codes, glac_basin_codes, rr_bsn_codes

    def get_obs(self, glac_loss_config, rr_loss_config, eval_vars):
        obs = dict()
        if 'glac_area' in eval_vars:
            glac_area_path = self.cal_abs_path(glac_loss_config['glac_area_path'],)
            glac_area_obs = pd.read_csv(glac_area_path, dtype={'basin_id': str}, sep=r'\s+')
            glac_area_obs['date'] = pd.to_datetime(glac_area_obs['date'])
            glac_area_obs['basin_id'] = glac_area_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            # reorder the observed glacier area based on the given basin codes
            glac_area_obs['basin_id'] = pd.Categorical(glac_area_obs['basin_id'],
                                                       categories=glac_loss_config['bsn_code'], ordered=True)
            glac_area_obs = glac_area_obs.sort_values(by=['basin_id'])
            glac_area_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_area'] = glac_area_obs

        if 'glac_tvol' in eval_vars:
            # load the observed glacier volume during 2017-2018
            glac_vol_path = self.cal_abs_path(glac_loss_config['glac_vol_path'])
            glac_vol_obs = pd.read_csv(glac_vol_path, dtype={'basin_id': str}, sep=r'\s+')
            glac_vol_obs['basin_id'] = glac_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            glac_vol_obs['basin_id'] = pd.Categorical(glac_vol_obs['basin_id'],
                                                      categories=glac_loss_config['bsn_code'], ordered=True)
            glac_vol_obs = glac_vol_obs.sort_values(by=['basin_id'])
            glac_vol_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_vol'] = glac_vol_obs
            # set the date as 2018-1-1
            glac_vol_obs = glac_vol_obs['vol'].values

            # load the observed glacier volume change during 2000-2019
            glac_dvol_path = self.cal_abs_path(glac_loss_config['glac_dvol_path'])
            glac_dvol_obs = pd.read_csv(glac_dvol_path, index_col=0, sep=r'\s+')
            glac_dvol_obs.columns = [str(x).zfill(12) for x in glac_dvol_obs.columns]
            glac_dvol_obs = glac_dvol_obs.reindex(columns=glac_loss_config['bsn_code']) * 10 ** (-9) # convert m^3 to km^3
            obs['glac_dvol'] = glac_dvol_obs

            # calculate the glacier volume change during 2000-2019 based on the glac_dvol_obs and glac_vol_obs
            if 'Hugonnet' in glac_loss_config['glac_dvol_path']:
                dates = pd.date_range(start='2000-01-01', end='2018-12-31', freq='YS')
            else:
                dates = pd.date_range(start='1957-01-01', end='2018-12-31', freq='YS')
            dates = dates.append(pd.to_datetime(['2018-12-31', '2019-12-31']))
            # Create a DataFrame to store the gvol for each date
            glac_tvol_obs = pd.DataFrame(index=dates, columns=glac_dvol_obs.columns)
            # Set the gvol for 2018-1-1
            glac_tvol_obs.loc['2018-01-01'] = glac_vol_obs
            # Calculate the gvol for the specified dates
            for date in dates:
                if date < pd.to_datetime('2018-01-01'):
                    glac_tvol_obs.loc[date] = glac_tvol_obs.loc['2018-01-01'] - glac_dvol_obs.loc[date.year:2017].sum()
                elif date > pd.to_datetime('2018-01-01'):
                    glac_tvol_obs.loc[date] = glac_tvol_obs.loc['2018-01-01'] + glac_dvol_obs.loc['2018':date.year].sum()
            glac_tvol_obs[glac_tvol_obs < 0] = 0
            glac_tvol_obs[glac_dvol_obs.columns[glac_dvol_obs.isna().any()]] = np.nan
            obs['glac_tvol'] = glac_tvol_obs


        if 'glac_vol' in eval_vars and 'glac_vol' not in obs.keys():
            glac_vol_path = self.cal_abs_path(glac_loss_config['glac_vol_path'])
            glac_vol_obs = pd.read_csv(glac_vol_path, dtype={'basin_id': str}, sep=r'\s+')
            glac_vol_obs['basin_id'] = glac_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            glac_vol_obs['basin_id'] = pd.Categorical(glac_vol_obs['basin_id'],
                                                      categories=glac_loss_config['bsn_code'], ordered=True)
            glac_vol_obs = glac_vol_obs.sort_values(by=['basin_id'])
            glac_vol_obs.dropna(axis=0, how='any', inplace=True, subset=['basin_id'])
            obs['glac_vol'] = glac_vol_obs

        if 'glac_dvol' in eval_vars and 'glac_dvol' not in obs.keys():
            glac_dvol_path = self.cal_abs_path(glac_loss_config['glac_dvol_path'])
            glac_dvol_obs = pd.read_csv(glac_dvol_path, index_col=0, sep=r'\s+')
            glac_dvol_obs.columns = [str(x).zfill(12) for x in glac_dvol_obs.columns]
            glac_dvol_obs = glac_dvol_obs.reindex(columns=glac_loss_config['bsn_code']) * 10 ** (-9) # convert m^3 to km^3
            obs['glac_dvol'] = glac_dvol_obs

        # load the observed snow depth
        if 'glac_sdep' in eval_vars:
            glac_sdep_path = self.cal_abs_path(glac_loss_config['glac_sdep_path'])
            glac_sdep_obs = pd.read_csv(glac_sdep_path, index_col=0, parse_dates=True, sep=r'\s+')
            glac_sdep_obs.columns = [str(x).zfill(12) for x in glac_sdep_obs.columns]
            glac_sdep_obs = glac_sdep_obs.reindex(columns=glac_loss_config['bsn_code'])
            obs['glac_sdep'] = glac_sdep_obs


        if 'bsn_Q' in eval_vars:
            # read the observed streamflow data
            bsn_Q_path = self.cal_abs_path(rr_loss_config['bsn_Q_path'])
            bsn_q_obs_daily = pd.read_excel(bsn_Q_path, index_col=0, parse_dates=True, sheet_name='daily')
            bsn_q_obs_monthly = pd.read_excel(bsn_Q_path, index_col=0, parse_dates=True, sheet_name='monthly')
            bsn_q_obs_monthly.index = bsn_q_obs_monthly.index.to_period('M')
            df_gauges = pd.read_excel(bsn_Q_path, sheet_name='gauges')
            # filter stations and determine the idx of bsn for each station
            bsn_code = rr_loss_config['bsn_code']
            bsn_idx_daily, station_daily, bsn_idx_monthly, station_monthly = [], [], [], []
            for station in bsn_q_obs_daily.columns:
                basin_ids = df_gauges[df_gauges['Station'] == station]['BasinIds'].values[0].split(',')
                basin_idx = [bsn_code.index(basin_id) for basin_id in basin_ids if basin_id in bsn_code]
                if len(basin_idx) > 0:
                    bsn_idx_daily.append(basin_idx)
                    station_daily.append(station)
            for station in bsn_q_obs_monthly.columns:
                basin_ids = df_gauges[df_gauges['Station'] == station]['BasinIds'].values[0].split(',')
                basin_idx = [bsn_code.index(basin_id) for basin_id in basin_ids if basin_id in bsn_code]
                if len(basin_idx) > 0:
                    bsn_idx_monthly.append(basin_idx)
                    station_monthly.append(station)
            bsn_q_obs_daily, bsn_q_obs_monthly = bsn_q_obs_daily.loc[:, station_daily], bsn_q_obs_monthly.loc[:, station_monthly]

            obs['bsn_Q'] = {'daily': {'Q': bsn_q_obs_daily, 'idx': bsn_idx_daily},
                            'monthly': {'Q': bsn_q_obs_monthly, 'idx': bsn_idx_monthly}}

        if 'bsn_LAI' in eval_vars:
            bsn_LAI_path = self.cal_abs_path(rr_loss_config['bsn_LAI_path'])
            bsn_LAI_obs = pd.read_csv(bsn_LAI_path, index_col=0, parse_dates=True, sep=r'\s+')
            bsn_LAI_obs.columns = [str(x).zfill(12) for x in bsn_LAI_obs.columns]
            bsn_LAI_obs = bsn_LAI_obs.reindex(columns=rr_loss_config['bsn_code'])
            obs['bsn_LAI'] = bsn_LAI_obs

        if 'bsn_sdep' in eval_vars:
            bsn_sdep_path = self.cal_abs_path(rr_loss_config['bsn_sdep_path'])
            bsn_sdep_obs = pd.read_csv(bsn_sdep_path, index_col=0, parse_dates=True, sep=r'\s+')
            bsn_sdep_obs.columns = [str(x).zfill(12) for x in bsn_sdep_obs.columns]
            bsn_sdep_obs = bsn_sdep_obs.reindex(columns=rr_loss_config['bsn_code'])
            # df = bsn_sdep_obs.copy()  # df: [time, basin]
            # no_snow = (df < 1) | df.isna()  # [time, basin]
            # valid_mask = (no_snow.mean(axis=0) < 0.8) & (df.mean(axis=0, skipna=True) > 1)
            # obs['bsn_sdep'] = bsn_sdep_obs.loc[:, valid_mask]
            obs['bsn_sdep'] = bsn_sdep_obs

        return obs

    def evaluate(self, sim, eval_vars, glac_bsn_codes, rr_bsn_codes, only_gs_LAI=False, period='test'):
        if period == 'train':
            t_range = self.train
        elif period == 'valid':
            t_range = self.valid
        else:
            t_range = self.test
        idx_time = sim['time'].get_indexer(t_range)

        metrics, ds = dict(), dict()
        if 'glac_area' in eval_vars and 'g_area' in sim['glac_bsn'].keys():
            glac_area_sim_df = pd.DataFrame(sim['glac_bsn']['g_area'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            obs_dates = self.obs['glac_area']['date']
            glac_area_sim = np.array([glac_area_sim_df.loc[date, glac_bsn_codes[i]] if date in glac_area_sim_df.index
                                      else np.nan for i, date in enumerate(obs_dates)])
            glac_area_obs = self.obs['glac_area'].loc[:, 'area'].values
            obs_dates = obs_dates[~np.isnan(glac_area_sim)]
            basin_ids = self.obs['glac_area'].loc[:, 'basin_id'].values[~np.isnan(glac_area_sim)]
            glac_area_obs, glac_area_sim = glac_area_obs[~np.isnan(glac_area_sim)], glac_area_sim[~np.isnan(glac_area_sim)]
            ds['glac_area'] = pd.DataFrame({'basin_id': basin_ids, 'obs': glac_area_obs.astype(float), 'sim': glac_area_sim.astype(float)},
                                           index=obs_dates)
            # calculate the metrics
            if len(glac_area_obs) == 0:
                logging.info(f'No observed glacier area data in {period} period.')
            else:
                r, nse, rmse, kge, pbias = self.eval_fn(true=glac_area_obs, pred=glac_area_sim, cal_dim=0)
                metrics['glac_area'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'glac_tvol' in eval_vars and 'g_vol' in sim['glac_bsn'].keys():
            glac_tvol_obs_df = self.obs['glac_tvol']
            glac_tvol_sim_df = pd.DataFrame(sim['glac_bsn']['g_vol'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            glac_tvol_sim_df = glac_tvol_sim_df.loc[glac_tvol_sim_df.index.isin(glac_tvol_obs_df.index)]
            glac_tvol_obs_df = glac_tvol_obs_df.loc[glac_tvol_obs_df.index.isin(glac_tvol_sim_df.index)]
            # gvol_obs, gvol_sim = gvol_obs_df.values.T.astype(float), gvol_sim_df.values.T
            glac_tvol_obs = glac_tvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            glac_tvol_sim = glac_tvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            ds['glac_tvol'] = {'sim': glac_tvol_sim_df, 'obs': glac_tvol_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=glac_tvol_obs, pred=glac_tvol_sim, cal_dim=0)
            metrics['glac_tvol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'glac_dvol' in eval_vars and 'g_vol' in sim['glac_bsn'].keys():
            glac_dvol_obs_df = self.obs['glac_dvol']
            glac_dvol_obs_df = glac_dvol_obs_df.loc[glac_dvol_obs_df.index.isin([t.year for t in t_range])]
            glac_tvol_sim_df = pd.DataFrame(sim['glac_bsn']['g_vol'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            glac_dvol_sim_df = pd.DataFrame(index=glac_dvol_obs_df.index, columns=glac_bsn_codes)
            for year in glac_dvol_sim_df.index: # type: ignore
                glac_dvol_sim_df.loc[year] = (glac_tvol_sim_df.loc[pd.to_datetime(f'{year}-12-31')] -
                                              glac_tvol_sim_df.loc[pd.to_datetime(f'{year}-1-1')])
            # dgvol_obs, dgvol_sim = dgvol_obs_df.values.T, glac_dvol_sim_df.values.T
            dgvol_obs = glac_dvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            dgvol_sim = glac_dvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            ds['glac_dvol'] = {'sim': glac_dvol_sim_df, 'obs': glac_dvol_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=dgvol_obs, pred=dgvol_sim, cal_dim=0)
            metrics['glac_dvol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'glac_vol' in eval_vars and 2017 in t_range.year and 'g_vol' in sim['glac_bsn'].keys() :
            glac_vol_obs_df = self.obs['glac_vol']
            glac_vol_obs = glac_vol_obs_df.loc[:, 'vol'].values
            glac_tvol_sim_df = pd.DataFrame(sim['glac_bsn']['g_vol'][:, idx_time].T,
                                            index=t_range, columns=glac_bsn_codes)
            glac_vol_sim = glac_tvol_sim_df[glac_tvol_sim_df.index.year == 2017].values.mean(axis=0).T
            ds['glac_vol'] = pd.DataFrame({'obs': glac_vol_obs, 'sim': glac_vol_sim}, index=glac_bsn_codes)
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=glac_vol_obs, pred=glac_vol_sim, cal_dim=0)
            metrics['glac_vol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'glac_sdep' in eval_vars and 's_depth' in sim['glac_bsn'].keys():
            s_depth_obs_df = self.obs['glac_sdep'].loc[self.obs['glac_sdep'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(sim['glac_bsn']['s_depth'][:, idx_time].T,
                                          index=t_range, columns=glac_bsn_codes)
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            # aggregate the observed and simulated snow depth to monthly scale
            if self.glac_snow_scale == 'monthly':
                s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
                s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
                ds['glac_sdep'] = {'sim': s_depth_sim_df.resample('ME').mean(), 'obs': s_depth_obs_df.resample('ME').mean()}
            else:
                s_depth_obs, s_depth_sim = s_depth_obs_df.values.T, s_depth_sim_df.values.T
                ds['glac_sdep'] = {'sim': s_depth_sim_df, 'obs': s_depth_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1)
            metrics['glac_sdep'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}


        if 'bsn_Q' in eval_vars and 'Qriver' in sim['rr_bsn'].keys():
            if self.q_scale == 'daily':
                q_obs_daily_df = self.obs['bsn_Q']['daily']['Q'].loc[self.obs['bsn_Q']['daily']['Q'].index.isin(t_range)]  # type: ignore
                non_nan_count = np.sum(~np.isnan(q_obs_daily_df.values), axis=0)  # type: ignore
                valid_bsn_idx = np.nonzero(non_nan_count > 365)[0]
                if len(valid_bsn_idx) > 0:
                    q_obs_daily_df = q_obs_daily_df.loc[:, q_obs_daily_df.columns[valid_bsn_idx]]
                    q_obs_daily = q_obs_daily_df.values.T
                    q_sim = sim['rr_bsn']['Qriver'][:, idx_time]
                    idx_bsn = [self.obs['bsn_Q']['daily']['idx'][idx] for idx in valid_bsn_idx]
                    q_sim_daily = np.stack([q_sim[idx, :].sum(axis=0) for idx in idx_bsn])
                    ds['bsn_Q_d'] = {'sim': pd.DataFrame(q_sim_daily.T, index=q_obs_daily_df.index, columns=q_obs_daily_df.columns),
                        'obs': q_obs_daily_df}
                    r, nse, rmse, kge, pbias, flv, fhv = self.eval_fn(true=q_obs_daily, pred=q_sim_daily, cal_dim=1,
                                                                      cal_flv=True)
                    metrics['bsn_Q_d'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias, 'flv': flv,
                                          'fhv': fhv}
            else:
                # 1. Retrieve daily observations within the target time range
                q_obs_daily_subset = self.obs['bsn_Q']['daily']['Q'].loc[
                    self.obs['bsn_Q']['daily']['Q'].index.isin(t_range)]  # type: ignore

                # 2. Check data validity
                # Filter basins that have enough valid DAILY data points (e.g., > 365 days)
                # to ensure the derived monthly means are statistically meaningful.
                non_nan_count = np.sum(~np.isnan(q_obs_daily_subset.values), axis=0)  # type: ignore
                valid_bsn_idx = np.nonzero(non_nan_count > 365)[0]

                if len(valid_bsn_idx) > 0:
                    # 3. Filter valid basins for daily observations
                    q_obs_daily_subset = q_obs_daily_subset.loc[:, q_obs_daily_subset.columns[valid_bsn_idx]]
                    # 4. Retrieve and process simulated data (Daily Scale first)
                    # Get the corresponding basin indices
                    idx_bsn = [self.obs['bsn_Q']['daily']['idx'][idx] for idx in valid_bsn_idx]
                    # Retrieve raw simulated daily streamflow (all time steps)
                    q_sim_raw = sim['rr_bsn']['Qriver']  # shape: [n_basins, n_total_days]
                    # Spatial Aggregation: Sum sub-basins to the gauge location
                    q_sim_daily_aggr = np.stack([q_sim_raw[idx, :].sum(axis=0) for idx in idx_bsn])
                    # Create a DataFrame using the full simulation time index
                    df_sim_daily_all = pd.DataFrame(q_sim_daily_aggr.T, index=sim['time'],
                                                    columns=q_obs_daily_subset.columns)
                    # 5. Align Simulation with Observation (Time & NaN Consistency)
                    # First, extract simulation data for the exact dates present in the observation subset
                    df_sim_daily_matched = df_sim_daily_all.reindex(q_obs_daily_subset.index)
                    df_sim_daily_matched = df_sim_daily_matched.where(q_obs_daily_subset.notna())

                    # 6. Resample to Monthly Scale
                    # Calculate monthly means (ignoring NaNs). 'ME' = Month End frequency.
                    q_obs_mon_df = q_obs_daily_subset.resample('ME').mean()
                    q_sim_mon_df = df_sim_daily_matched.resample('ME').mean()

                    # Align indices strictly (in case resampling created extra months at edges)
                    q_sim_mon_df = q_sim_mon_df.reindex(q_obs_mon_df.index)

                    # 7. Prepare for Evaluation
                    q_obs_mon = q_obs_mon_df.values.T
                    q_sim_mon = q_sim_mon_df.values.T

                    # Store the monthly DataFrames for reference
                    ds['bsn_Q_m'] = {'sim': q_sim_mon_df, 'obs': q_obs_mon_df}

                    # 8. Calculate Metrics
                    # Note: self.eval_fn should be able to handle NaNs (or ignore them) in the input arrays.
                    r, nse, rmse, kge, pbias, flv, fhv = self.eval_fn(true=q_obs_mon, pred=q_sim_mon, cal_dim=1,
                                                                      cal_flv=True)
                    metrics['bsn_Q_m'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias, 'flv': flv,
                                          'fhv': fhv}

            t_range_monthly = t_range.to_period('M') # type: ignore
            q_obs_monthly_df = self.obs['bsn_Q']['monthly']['Q'].loc[t_range_monthly.unique()]
            ts_mask = [(t_range.month == index.month) & (t_range.year == index.year) for index in q_obs_monthly_df.index] # type: ignore
            non_nan_count = np.sum(~np.isnan(q_obs_monthly_df.values), axis=0)
            valid_bsn_idx = np.nonzero(non_nan_count > 24)[0]
            if len(valid_bsn_idx) > 0:
                q_obs_monthly_df = q_obs_monthly_df.loc[:, q_obs_monthly_df.columns[valid_bsn_idx]]
                q_obs_monthly = q_obs_monthly_df.values.T
                q_sim = sim['rr_bsn']['Qriver'][:, idx_time]
                idx_bsn = [self.obs['bsn_Q']['monthly']['idx'][idx] for idx in valid_bsn_idx]
                q_sim = np.stack([q_sim[idx, :].sum(axis=0) for idx in idx_bsn]) * 86400 # convert m3/s to m3/d
                q_sim_monthly = np.stack([q_sim[:, mask].sum(axis=1) for mask in ts_mask], axis=1) / 10**8
                ds['bsn_Q_m'] = {'sim': pd.DataFrame(q_sim_monthly.T, index=q_obs_monthly_df.index, columns=q_obs_monthly_df.columns),
                                 'obs': q_obs_monthly_df}
                r, nse, rmse, kge, pbias, flv, fhv = self.eval_fn(true=q_obs_monthly, pred=q_sim_monthly, cal_dim=1, cal_flv=True)
                metrics['bsn_Q_m'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias, 'flv': flv, 'fhv': fhv}


        if 'bsn_LAI' in eval_vars and 'LAI' in sim['rr_bsn'].keys():
            lai_obs_df = self.obs['bsn_LAI'].loc[self.obs['bsn_LAI'].index.isin(t_range)]
            lai_sim = sim['rr_bsn']['LAI'][:, idx_time]
            idx_lst = np.where(t_range.isin(lai_obs_df.index))[0]
            lai_sim_resample = np.stack([np.mean(lai_sim[:, start:end], axis=1) for start, end
                                         in zip(idx_lst[:-1], idx_lst[1:])], axis=1)
            lai_obs_resample = lai_obs_df.values.T[:, :-1]
            nan_mask = np.isnan(lai_obs_resample).any(axis=1)
            lai_sim_resample, lai_obs_resample = lai_sim_resample[~nan_mask], lai_obs_resample[~nan_mask]
            ds['bsn_LAI'] = {'sim': pd.DataFrame(lai_sim_resample.T, index=lai_obs_df.index[:-1],
                                                 columns=lai_obs_df.columns[~nan_mask]),
                             'obs': pd.DataFrame(lai_obs_resample.T, index=lai_obs_df.index[:-1],
                                                 columns=lai_obs_df.columns[~nan_mask])}
            if only_gs_LAI:
                idx_lst_gs = np.where((t_range.isin(lai_obs_df.index)) & (t_range.month >= 5) & (t_range.month <= 10))[0] # type: ignore
                mask = np.where(np.isin(idx_lst, idx_lst_gs))[0]
                lai_sim_cal, lai_obs_cal = lai_sim_resample[:, mask], lai_obs_resample[:, mask]
            else:
                lai_sim_cal, lai_obs_cal = lai_sim_resample, lai_obs_resample

            r, nse, rmse, kge, pbias = self.eval_fn(true=lai_obs_cal, pred=lai_sim_cal, cal_dim=1)
            metrics['bsn_LAI'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'bsn_sdep' in eval_vars and 'sdep' in sim['rr_bsn'].keys():
            s_depth_obs_df = self.obs['bsn_sdep'].loc[self.obs['bsn_sdep'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(sim['rr_bsn']['sdep'][:, idx_time].T,
                                          index=t_range, columns=rr_bsn_codes)
            s_depth_sim_df = s_depth_sim_df[s_depth_obs_df.columns]
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            # aggregate the observed and simulated snow depth to monthly scale
            if self.rr_snow_scale == 'monthly':
                s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
                s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
                ds['bsn_sdep'] = {'sim': s_depth_sim_df.resample('ME').mean(), 'obs': s_depth_obs_df.resample('ME').mean()}
            else:
                s_depth_obs, s_depth_sim = s_depth_obs_df.values.T, s_depth_sim_df.values.T
                ds['bsn_sdep'] = {'sim': s_depth_sim_df, 'obs': s_depth_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1, var='snow')
            metrics['bsn_sdep'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}


        return ds, metrics

    @ staticmethod
    def eval_fn(true: np.ndarray, pred: np.ndarray, cal_dim: int = 0, cal_flv: bool = False, var=None):
        # check the dimensions of true and pred
        if len(true.shape) == 1 and len(pred.shape) == 1:
            true = true.reshape(1, -1)
            pred = pred.reshape(1, -1)
            cal_dim = 1
        assert true.ndim == 2 and pred.ndim == 2, 'The dimensions of true and pred should be 1 or 2.'
        # make sure the dtype is float
        if var == 'snow':
            no_snow = (true < 1) | np.isnan(true)  # [n, m]
            valid_mask = (no_snow.mean(axis=1) < 0.8) & (np.nanmean(true, axis=1) > 1) & (
                    np.nanmax(true, axis=1) > 10)
            true[~valid_mask, :] = np.nan
        all_nan_mask = np.isnan(true).all(axis=cal_dim, keepdims=True)
        true, pred = true.astype(float), pred.astype(float)
        pred[np.isnan(true)] = np.nan
        # Calculate mean along the specified dimension
        true_mean = np.nanmean(true, axis=cal_dim, keepdims=True)
        pred_mean = np.nanmean(pred, axis=cal_dim, keepdims=True)

        # Calculate r
        r_num = np.nansum((pred - pred_mean) * (true - true_mean), axis=cal_dim, keepdims=True)
        r_den = np.sqrt(np.nansum((pred - pred_mean) ** 2, axis=cal_dim, keepdims=True) *
                        np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True))
        r_den[r_den == 0] = 1e-5  # avoid division by zero
        r = r_num / r_den
        r = np.where(all_nan_mask, np.nan, r)  # set r to nan where all values are nan

        # Calculate NSE
        nse_num = np.nansum((pred - true) ** 2, axis=cal_dim, keepdims=True)
        nse_den = np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True)
        nse_den[nse_den == 0] = 1e-5  # avoid division by zero
        nse = 1 - nse_num / nse_den
        nse = np.where(all_nan_mask, np.nan, nse)  # set nse to nan where all values are nan

        # Calculate RMSE
        rmse = np.sqrt(np.nanmean((pred - true) ** 2, axis=cal_dim, keepdims=True))

        # Calculate KGE components
        true_mean[true_mean == 0] = 1e-5
        alpha = pred_mean / true_mean
        beta_num = np.nanstd(pred, axis=cal_dim, keepdims=True)
        beta_den = np.nanstd(true, axis=cal_dim, keepdims=True)
        beta_den[beta_den == 0] = 1e-5  # avoid division by zero
        beta = beta_num / beta_den
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # calculate the percent bias
        pbias_num = np.nanmean(pred - true, axis=cal_dim, keepdims=True)
        pbias_den = np.nanmean(true, axis=cal_dim, keepdims=True)
        pbias_den[pbias_den == 0] = 1e-5  # avoid division by zero
        pbias = pbias_num / pbias_den * 100

        # FLV the low flows bias bottom 30%, FHV the high flows bias top 2%
        if cal_flv:
            flv_list = []
            fhv_list = []
            for i in range(pred.shape[0]):
                pred_clean = pred[i, ~np.isnan(pred[i])]
                true_clean = true[i, ~np.isnan(true[i])]
                if len(pred_clean) == 0 or len(true_clean) == 0:
                    flv_list.append(np.nan)
                    fhv_list.append(np.nan)
                    continue
                pred_sort = np.sort(pred_clean)
                target_sort = np.sort(true_clean)
                index_low = round(0.3 * len(pred_sort))
                low_pred = pred_sort[:index_low]
                low_target = target_sort[:index_low]
                flv = (np.sum(low_pred - low_target, keepdims=True) / np.sum(low_target, keepdims=True)) * 100
                flv_list.append(flv)
                index_high = round(0.98 * len(pred_sort))
                high_pred = pred_sort[index_high:]
                high_target = target_sort[index_high:]
                fhv = (np.sum(high_pred - high_target, keepdims=True) / np.sum(high_target, keepdims=True)) * 100
                fhv_list.append(fhv)
            flv = np.array(flv_list)
            fhv = np.array(fhv_list)
            return r, nse, rmse, kge, pbias, flv, fhv
        else:
            return r, nse, rmse, kge, pbias


if __name__ == '__main__':
    folder = '../checkpoints/ensemble'
    evaluator = EvalEnsemble(folder=folder,
                             eval_vars=['glac_vol', 'bsn_sdep', 'bsn_LAI', 'bsn_Q'],
                             q_scale='monthly')

