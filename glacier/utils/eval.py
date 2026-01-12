import torch
import logging
from collections import defaultdict
import pickle
import json
import pandas as pd
import numpy as np
from collections import OrderedDict
import os
from pathlib import Path

class SaveEval:
    def __init__(self, loader_all, model, config, basin_codes, eval_vars):
        logging.info(f'Get simulation results for validation and test datasets.')
        self.train = pd.date_range(start=config['data']['periods'][0][0], end=config['data']['periods'][0][1], freq='D')
        self.valid = pd.date_range(start=config['data']['periods'][1][0], end=config['data']['periods'][1][1], freq='D')
        self.test = pd.date_range(start=config['data']['periods'][2][0], end=config['data']['periods'][2][1], freq='D')

        # load the model with the best validation loss
        model.load_state_dict(torch.load(os.path.join(config['out'], 'model.pt'), weights_only=True))
        # run the model to get the simulation results
        self.sim = self.get_sim(loader=loader_all, model=model, config=config)
        # load the observed data
        self.obs = self.get_obs(config=config, basin_codes=basin_codes, eval_vars=eval_vars)
        # calculate metrics
        self.evaluate(eval_vars=eval_vars, basin_codes=basin_codes, period='valid')
        self.evaluate(eval_vars=eval_vars, basin_codes=basin_codes, period='test')

    def get_sim(self, loader, model, config):
        if os.path.exists(os.path.join(config['out'], 'sim.pkl')):
            with open(os.path.join(config['out'], 'sim.pkl'), 'rb') as f:
                sim = pickle.load(f)
        else:
            model.eval()
            spin_up_len = config['data']['spin_up_len']
            out_train, out_valid, out_tst = defaultdict(dict), defaultdict(dict), defaultdict(dict)
            with torch.no_grad():
                for i, (forc, forc_norm, attrs_norm, ts) in enumerate(loader):
                    forc, forc_norm, attrs_norm = forc.squeeze(0), forc_norm.squeeze(0), attrs_norm.squeeze(0)
                    ts = pd.to_datetime(ts.squeeze(0))
                    # run the training period to get the initial state
                    logging.info('Run the training period to get the initial state.')
                    output_band, output_bsn = model(forc=forc[:, :spin_up_len + len(self.train)],
                                                    forc_norm=forc_norm[:, :spin_up_len + len(self.train)],
                                                    attrs_norm=attrs_norm,
                                                    ts=ts[:spin_up_len + len(self.train)],
                                                    spin_up_len=spin_up_len,
                                                    mode='eval',
                                                    initial_state=None)
                    if config['train']['gwe_swe_t0']:
                        state_t0 = {
                            'area_band': output_band['g_area'].detach().clone()[:, -spin_up_len],
                            'area_bsn': output_bsn['g_area'].detach().clone()[:, -spin_up_len],
                            'gwe_band': output_band['g_we'].detach().clone()[:, -spin_up_len],
                            'swe_band': output_band['s_we'].detach().clone()[:, -spin_up_len]
                        }
                    else:
                        state_t0 = {
                            'area_band': output_band['g_area'].detach().clone()[:, -spin_up_len],
                            'area_bsn': output_bsn['g_area'].detach().clone()[:, -spin_up_len],
                        }
                    # save the training results
                    for scale, output in dict(zip(['band', 'basin'], [output_band, output_bsn])).items():
                        for k, v in output.items():
                            if k.startswith('param'):
                                out_train[scale][k] = v.detach().cpu().numpy()
                            else:
                                out_train[scale][k] = v.detach().cpu().numpy()[:, spin_up_len:]
                    del output_band, output_bsn # release the memory


                    # validation and test periods
                    logging.info('Run the validation and test periods.')
                    output_band, output_bsn = model(forc=forc[:, len(self.train):],
                                                    forc_norm=forc_norm[:, len(self.train):],
                                                    attrs_norm=attrs_norm,
                                                    ts=ts[len(self.train):],
                                                    spin_up_len=spin_up_len,
                                                    mode='eval',
                                                    initial_state=state_t0)

                    # save the validation and test results
                    for scale, output in dict(zip(['band', 'basin'], [output_band, output_bsn])).items():
                        for k, v in output.items():
                            if k.startswith('param'):
                                out_valid[scale][k] = v.detach().cpu().numpy()
                                out_tst[scale][k] = v.detach().cpu().numpy()
                            else:
                                # validation results
                                start_idx_valid = spin_up_len
                                end_idx_valid = start_idx_valid + len(self.valid)
                                out_valid[scale][k] = v.detach().cpu().numpy()[:, start_idx_valid:end_idx_valid]
                                # test results
                                start_idx_tst = end_idx_valid
                                end_idx_tst = start_idx_tst + len(self.test)
                                out_tst[scale][k] = v.detach().cpu().numpy()[:, start_idx_tst:end_idx_tst]
                    del output_band, output_bsn # release the memory

            sim = {'train': out_train,'valid': out_valid, 'test': out_tst}
            # save the simulation results
            with open(os.path.join(config['out'], 'sim.pkl'), 'wb') as f:
                pickle.dump(sim, f)  # type: ignore
        return sim

    @staticmethod
    def get_obs(config, basin_codes, eval_vars):
        obs = dict()
        # load the observed glacier area
        if 'g_area' in eval_vars:
            g_area_obs = pd.read_csv(config['data']['g_area_path'], sep=r'\s+', dtype={'basin_id': str})
            g_area_obs['date'] = pd.to_datetime(g_area_obs['date'])
            g_area_obs['basin_id'] = g_area_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            # reorder the observed glacier area based on the given basin codes
            g_area_obs['basin_id'] = pd.Categorical(g_area_obs['basin_id'], categories=basin_codes, ordered=True)
            g_area_obs = g_area_obs.sort_values(by=['basin_id'])
            obs['g_area'] = g_area_obs

        # load the observed glacier volume during 2000-2019
        if 't_gvol' in eval_vars:
            assert (config['data']['gvol_path'].split('_')[-1]).split('.')[0] == 'basin', 'The glacier volume should be at basin scale.'
            # load the observed glacier volume during 2017-2018
            g_vol_obs = pd.read_csv(config['data']['gvol_path'], sep=r'\s+', dtype={'basin_id': str})
            g_vol_obs['basin_id'] = g_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            g_vol_obs['basin_id'] = pd.Categorical(g_vol_obs['basin_id'], categories=basin_codes, ordered=True)
            g_vol_obs = g_vol_obs.sort_values(by=['basin_id'])
            obs['g_vol'] = g_vol_obs
            # set the date as 2018-1-1
            gvol_obs = g_vol_obs['vol'].values

            # load the observed glacier volume change during 2000-2019
            d_gvol_obs = pd.read_csv(config['data']['d_gvol_path'], sep=r'\s+', index_col=0)
            d_gvol_obs.columns = [str(x).zfill(12) for x in d_gvol_obs.columns]
            d_gvol_obs = d_gvol_obs.reindex(columns=basin_codes) * 10 ** (-9)  # convert m^3 to km^3
            obs['d_gvol'] = d_gvol_obs

            # calculate the glacier volume change during 2000-2019 based on the glac_dvol_obs and g_vol_obs
            if 'Hugonnet' in config['data']['d_gvol_path']:
                dates = pd.date_range(start='2000-01-01', end='2018-12-31', freq='YS')
            else:
                dates = pd.date_range(start='1957-01-01', end='2018-12-31', freq='YS')
            dates = dates.append(pd.to_datetime(['2018-12-31', '2019-12-31']))
            # Create a DataFrame to store the gvol for each date
            t_gvol_obs = pd.DataFrame(index=dates, columns=d_gvol_obs.columns)
            # Set the gvol for 2018-1-1
            t_gvol_obs.loc['2018-01-01'] = gvol_obs
            # Calculate the gvol for the specified dates
            for date in dates:
                if date < pd.to_datetime('2018-01-01'):
                    t_gvol_obs.loc[date] = t_gvol_obs.loc['2018-01-01'] - d_gvol_obs.loc[date.year:2017].sum()
                elif date > pd.to_datetime('2018-01-01'):
                    t_gvol_obs.loc[date] = t_gvol_obs.loc['2018-01-01'] + d_gvol_obs.loc['2018':date.year].sum()
            t_gvol_obs[t_gvol_obs < 0] = 0
            t_gvol_obs[d_gvol_obs.columns[d_gvol_obs.isna().any()]] = np.nan
            obs['t_gvol'] = t_gvol_obs

        # load the observed glacier volume during 2017-2018
        if 'g_vol' in eval_vars and 'g_vol' not in obs.keys():
            g_vol_obs = pd.read_csv(config['data']['gvol_path'], sep=r'\s+', dtype={'basin_id': str})
            g_vol_obs['basin_id'] = g_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            g_vol_obs['basin_id'] = pd.Categorical(g_vol_obs['basin_id'], categories=basin_codes, ordered=True)
            g_vol_obs = g_vol_obs.sort_values(by=['basin_id'])
            obs['g_vol'] = g_vol_obs

        # load the observed glacier volume change
        if 'd_gvol' in eval_vars and 'd_gvol' not in obs.keys():
            d_gvol_obs = pd.read_csv(config['data']['d_gvol_path'], sep=r'\s+', index_col=0)
            d_gvol_obs.columns = [str(x).zfill(12) for x in d_gvol_obs.columns]
            d_gvol_obs = d_gvol_obs.reindex(columns=basin_codes) * 10 ** (-9)  # convert m^3 to km^3
            obs['d_gvol'] = d_gvol_obs

        # load the observed snow depth
        if 's_depth' in eval_vars:
            s_depth_obs = pd.read_csv(config['data']['s_depth_path'], sep=r'\s+', index_col=0, parse_dates=True)
            s_depth_obs.columns = [str(x).zfill(12) for x in s_depth_obs.columns]
            s_depth_obs = s_depth_obs.reindex(columns=basin_codes)
            obs['s_depth'] = s_depth_obs


        return obs

    def evaluate(self, eval_vars, basin_codes, period='valid'):
        sim = self.sim[period]['basin']
        t_range = self.valid if period == 'valid' else self.test

        if 'g_area' in eval_vars:
            g_area_sim_df = pd.DataFrame(sim['g_area'].T, index=t_range, columns=basin_codes)
            obs_dates = self.obs['g_area']['date']
            g_area_sim = np.array([g_area_sim_df.loc[date, basin_codes[i]] if date in g_area_sim_df.index else np.nan
                                   for i, date in enumerate(obs_dates)])
            g_area_obs = self.obs['g_area']['area'].values
            g_area_obs, g_area_sim = g_area_obs[~np.isnan(g_area_sim)], g_area_sim[~np.isnan(g_area_sim)]
            # calculate the metrics
            if len(g_area_obs) == 0:
                logging.info(f'No observed glacier area data in {period} period.')
            else:
                r, nse, rmse, kge = self.eval_fn(true=g_area_obs, pred=g_area_sim, cal_dim=0)
                logging.info(f'For glacier area in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 't_gvol' in eval_vars:
            gvol_obs_df = self.obs['t_gvol']
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=basin_codes)
            gvol_sim_df = gvol_sim_df.loc[gvol_sim_df.index.isin(gvol_obs_df.index)]
            gvol_obs_df = gvol_obs_df.loc[gvol_obs_df.index.isin(gvol_sim_df.index)]
            # gvol_obs, gvol_sim = gvol_obs_df.values.T.astype(float), gvol_sim_df.values.T
            gvol_obs = gvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            gvol_sim = gvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            # calculate the metrics
            r, nse, rmse, kge = self.eval_fn(true=gvol_obs, pred=gvol_sim, cal_dim=0)
            logging.info(f'For glacier volume in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'd_gvol' in eval_vars:
            dgvol_obs_df = self.obs['d_gvol']
            dgvol_obs_df = dgvol_obs_df.loc[dgvol_obs_df.index.isin(range(t_range[0].year, t_range[-1].year + 1))]
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=basin_codes)
            dgvol_sim_df = pd.DataFrame( index=dgvol_obs_df.index, columns=basin_codes)
            for year in range(t_range[0].year, t_range[-1].year+1):
                dgvol_sim_df.loc[year] = gvol_sim_df.loc[pd.to_datetime(f'{year}-12-31')] - gvol_sim_df.loc[pd.to_datetime(f'{year}-1-1')]
            # dgvol_obs, dgvol_sim = dgvol_obs_df.values.T, dgvol_sim_df.values.T
            dgvol_obs = dgvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            dgvol_sim = dgvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            # calculate the metrics
            r, nse, rmse, kge = self.eval_fn(true=dgvol_obs, pred=dgvol_sim, cal_dim=0)
            logging.info(f'For glacier volume change in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 'g_vol' in eval_vars and 2017 in range(t_range[0].year, t_range[-1].year):
            gvol_obs = self.obs['g_vol']['vol'].values
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=basin_codes)
            gvol_sim = gvol_sim_df[gvol_sim_df.index.year == 2017].values.mean(axis=0).T
            # calculate the metrics
            r, nse, rmse, kge = self.eval_fn(true=gvol_obs, pred=gvol_sim, cal_dim=0)
            logging.info(f'For 2017 glacier volume in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

        if 's_depth' in eval_vars:
            s_depth_obs_df = self.obs['s_depth'].loc[self.obs['s_depth'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(sim['s_depth'].T, index=t_range, columns=basin_codes)
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            # aggregate the observed and simulated snow depth to monthly scale
            s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
            s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
            # calculate the metrics
            r, nse, rmse, kge = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1)
            logging.info(f'For snow depth in {period} period: R: {r:.3f}, NSE: {nse:.3f}, RMSE: {rmse:.3f}, KGE: {kge:.3f}')

    @staticmethod
    def eval_fn(true: np.ndarray, pred: np.ndarray, cal_dim: int = 0):
        # check the dimensions of true and pred
        if len(true.shape) == 1 and len(pred.shape) == 1:
            true = true.reshape(1, -1)
            pred = pred.reshape(1, -1)
            cal_dim = 1
        assert len(true.shape) == 2 and len(pred.shape) == 2, 'The dimensions of true and pred should be 1 or 2.'
        # make sure the dtype is float
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

        # Calculate NSE
        nse_num = np.nansum((pred - true) ** 2, axis=cal_dim, keepdims=True)
        nse_den = np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True)
        nse_den[nse_den == 0] = 1e-5 # avoid division by zero
        nse = 1 - nse_num / nse_den

        # Calculate RMSE
        rmse = np.sqrt(np.nanmean((pred - true) ** 2, axis=cal_dim, keepdims=True))

        # Calculate KGE components
        true_mean[true_mean == 0] = 1e-10
        alpha = pred_mean / true_mean
        beta_num = np.nanstd(pred, axis=cal_dim, keepdims=True)
        beta_den = np.nanstd(true, axis=cal_dim, keepdims=True)
        beta_den[beta_den == 0] = 1e-5 # avoid division by zero
        beta = beta_num / beta_den
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # Average the metrics over the other dimension
        r_mean = np.nanmean(r)
        nse_mean = np.nanmean(nse)
        rmse_mean = np.nanmean(rmse)
        kge_mean = np.nanmean(kge)

        return r_mean, nse_mean, rmse_mean, kge_mean


class EvalSingle:  # for single model evaluation
    def __init__(self, folder: str, eval_vars: list[str]):
        # load the simulation results
        with open(f'{folder}/sim.pkl', 'rb') as f:
            self.sim = pickle.load(f)
        # get the training, validation and test periods
        with open(f'{folder}/config.json', 'rb') as f:
            config = json.load(f)
        self.band_codes, self.basin_codes = self.get_basin_codes(config)
        self.train = pd.date_range(start=config['data']['periods'][0][0], end=config['data']['periods'][0][1], freq='D')
        self.valid = pd.date_range(start=config['data']['periods'][1][0], end=config['data']['periods'][1][1], freq='D')
        self.test = pd.date_range(start=config['data']['periods'][2][0], end=config['data']['periods'][2][1], freq='D')
        # get the observed data
        self.obs = self.get_obs(config, eval_vars)
        # get the metrics and dataset
        metrics, ds = dict(), dict()
        for period in ['valid', 'test']:
            ds[period], metrics[period] = self.evaluate(eval_vars, period)
        self.metrics, self.ds = metrics, ds

    def get_basin_codes(self, config):
        path = self.cal_abs_path(os.path.join(config['data']['forcing_dir'], 'forc_band.pkl'))
        forcing = pickle.load(open(path, 'rb'))
        forcing = OrderedDict(sorted(forcing.items()))
        band_codes = list(forcing.keys())
        basin_codes = sorted(list(set([band_code.split('_')[0] for band_code in band_codes])))
        return band_codes, basin_codes

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

    def get_obs(self, config, eval_vars):
        obs = dict()
        # load the observed glacier area
        if 'g_area' in eval_vars:
            g_area_path = self.cal_abs_path(config['data']['g_area_path'])
            g_area_obs = pd.read_csv(g_area_path, sep=r'\s+', dtype={'basin_id': str})
            g_area_obs['date'] = pd.to_datetime(g_area_obs['date'])
            g_area_obs['basin_id'] = g_area_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            # reorder the observed glacier area based on the given basin codes
            g_area_obs['basin_id'] = pd.Categorical(g_area_obs['basin_id'], categories=self.basin_codes, ordered=True)
            g_area_obs = g_area_obs.sort_values(by=['basin_id'])
            obs['g_area'] = g_area_obs

        # load the observed glacier volume during 2000-2019
        if 't_gvol' in eval_vars:
            assert (config['data']['gvol_path'].split('_')[-1]).split('.')[0] == 'basin', 'The glacier volume should be at basin scale.'
            # load the observed glacier volume during 2017-2018
            g_vol_path = self.cal_abs_path(config['data']['gvol_path'])
            g_vol_obs = pd.read_csv(g_vol_path, sep=r'\s+', dtype={'basin_id': str})
            g_vol_obs['basin_id'] = g_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            g_vol_obs['basin_id'] = pd.Categorical(g_vol_obs['basin_id'], categories=self.basin_codes, ordered=True)
            g_vol_obs = g_vol_obs.sort_values(by=['basin_id'])
            obs['g_vol'] = g_vol_obs
            # set the date as 2018-1-1
            gvol_obs = g_vol_obs['vol'].values

            # load the observed glacier volume change during 2000-2019
            d_gvol_path = self.cal_abs_path(config['data']['d_gvol_path'])
            d_gvol_obs = pd.read_csv(d_gvol_path, sep=r'\s+', index_col=0)
            d_gvol_obs.columns = [str(x).zfill(12) for x in d_gvol_obs.columns]
            d_gvol_obs = d_gvol_obs.reindex(columns=self.basin_codes) * 10 ** (-9)  # convert m^3 to km^3
            obs['d_gvol'] = d_gvol_obs

            # calculate the glacier volume change during 2000-2019 based on the glac_dvol_obs and g_vol_obs
            if 'Hugonnet' in config['data']['d_gvol_path']:
                dates = pd.date_range(start='2000-01-01', end='2018-12-31', freq='YS')
            else:
                dates = pd.date_range(start='1957-01-01', end='2018-12-31', freq='YS')
            dates = dates.append(pd.to_datetime(['2018-12-31', '2019-12-31']))
            # Create a DataFrame to store the gvol for each date
            t_gvol_obs = pd.DataFrame(index=dates, columns=d_gvol_obs.columns)
            # Set the gvol for 2018-1-1
            t_gvol_obs.loc['2018-01-01'] = gvol_obs
            # Calculate the gvol for the specified dates
            for date in dates:
                if date < pd.to_datetime('2018-01-01'):
                    t_gvol_obs.loc[date] = t_gvol_obs.loc['2018-01-01'] - d_gvol_obs.loc[date.year:2017].sum()
                elif date > pd.to_datetime('2018-01-01'):
                    t_gvol_obs.loc[date] = t_gvol_obs.loc['2018-01-01'] + d_gvol_obs.loc['2018':date.year].sum()
            t_gvol_obs[t_gvol_obs < 0] = 0
            t_gvol_obs[d_gvol_obs.columns[d_gvol_obs.isna().any()]] = np.nan
            obs['t_gvol'] = t_gvol_obs

        # load the observed glacier volume during 2017-2018
        if 'g_vol' in eval_vars and 'g_vol' not in obs.keys():
            gvol_path = self.cal_abs_path(config['data']['gvol_path'])
            g_vol_obs = pd.read_csv(gvol_path, sep=r'\s+', dtype={'basin_id': str})
            g_vol_obs['basin_id'] = g_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            g_vol_obs['basin_id'] = pd.Categorical(g_vol_obs['basin_id'], categories=self.basin_codes, ordered=True)
            g_vol_obs = g_vol_obs.sort_values(by=['basin_id'])
            obs['g_vol'] = g_vol_obs

        # load the observed glacier volume change
        if 'd_gvol' in eval_vars and 'd_gvol' not in obs.keys():
            d_gvol_path = self.cal_abs_path(config['data']['d_gvol_path'])
            d_gvol_obs = pd.read_csv(d_gvol_path, sep=r'\s+', index_col=0)
            d_gvol_obs.columns = [str(x).zfill(12) for x in d_gvol_obs.columns]
            d_gvol_obs = d_gvol_obs.reindex(columns=self.basin_codes) * 10 ** (-9)  # convert m^3 to km^3
            obs['d_gvol'] = d_gvol_obs

        # load the observed snow depth
        if 's_depth' in eval_vars:
            s_depth_path = self.cal_abs_path(config['data']['s_depth_path'])
            s_depth_obs = pd.read_csv(s_depth_path, sep=r'\s+', index_col=0, parse_dates=True)
            s_depth_obs.columns = [str(x).zfill(12) for x in s_depth_obs.columns]
            s_depth_obs = s_depth_obs.reindex(columns=self.basin_codes)
            obs['s_depth'] = s_depth_obs


        return obs

    def evaluate(self, eval_vars, period='valid'):
        metrics, ds = dict(), dict()
        sim = self.sim[period]['basin']
        t_range = self.valid if period == 'valid' else self.test
        if 'g_area' in eval_vars:
            g_area_sim_df = pd.DataFrame(sim['g_area'].T, index=t_range, columns=self.basin_codes)
            obs_dates = self.obs['g_area']['date']
            g_area_sim = np.array([g_area_sim_df.loc[date, self.basin_codes[i]] if date in g_area_sim_df.index else np.nan
                                   for i, date in enumerate(obs_dates)])
            g_area_obs = self.obs['g_area']['area'].values
            obs_dates = obs_dates[~np.isnan(g_area_sim)]
            g_area_obs, g_area_sim = g_area_obs[~np.isnan(g_area_sim)], g_area_sim[~np.isnan(g_area_sim)]
            ds['g_area'] = pd.DataFrame({'obs': g_area_obs.astype(float), 'sim': g_area_sim.astype(float)}, index=obs_dates)
            # calculate the metrics
            if len(g_area_obs) > 0:
                r, nse, rmse, kge, pbias = self.eval_fn(true=g_area_obs, pred=g_area_sim, cal_dim=0)
                metrics['g_area'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 't_gvol' in eval_vars:
            gvol_obs_df = self.obs['t_gvol']
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim_df = gvol_sim_df.loc[gvol_sim_df.index.isin(gvol_obs_df.index)]
            gvol_obs_df = gvol_obs_df.loc[gvol_obs_df.index.isin(gvol_sim_df.index)]
            # gvol_obs, gvol_sim = gvol_obs_df.values.T.astype(float), gvol_sim_df.values.T.astype(float)
            gvol_obs = gvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            gvol_sim = gvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            ds['t_gvol'] = {'sim': gvol_sim_df, 'obs': gvol_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=gvol_obs, pred=gvol_sim, cal_dim=0)
            metrics['t_gvol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'd_gvol' in eval_vars:
            dgvol_obs_df = self.obs['d_gvol']
            dgvol_obs_df = dgvol_obs_df.loc[dgvol_obs_df.index.isin(range(t_range[0].year, t_range[-1].year + 1))]
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=self.basin_codes)
            dgvol_sim_df = pd.DataFrame(index=dgvol_obs_df.index, columns=self.basin_codes)
            for year in range(t_range[0].year, t_range[-1].year+1):
                dgvol_sim_df.loc[year] = gvol_sim_df.loc[pd.to_datetime(f'{year}-12-31')] - gvol_sim_df.loc[pd.to_datetime(f'{year}-1-1')]
            # dgvol_obs, dgvol_sim = dgvol_obs_df.values.T.astype(float), dgvol_sim_df.values.T.astype(float)
            dgvol_obs = dgvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            dgvol_sim = dgvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            ds['d_gvol'] = {'sim': dgvol_sim_df.astype(float), 'obs': dgvol_obs_df.astype(float)}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=dgvol_obs, pred=dgvol_sim, cal_dim=0)
            metrics['d_gvol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'g_vol' in eval_vars and 2017 in range(t_range[0].year, t_range[-1].year):
            gvol_obs = self.obs['g_vol']['vol'].values
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim = gvol_sim_df[gvol_sim_df.index.year == 2017].values.mean(axis=0).T
            ds['g_vol'] = pd.DataFrame({'obs': gvol_obs, 'sim': gvol_sim}, index=self.basin_codes)
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=gvol_obs, pred=gvol_sim, cal_dim=0)
            metrics['g_vol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 's_depth' in eval_vars:
            s_depth_obs_df = self.obs['s_depth'].loc[self.obs['s_depth'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(sim['s_depth'].T, index=t_range, columns=self.basin_codes)
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            ds['s_depth'] = {'sim': s_depth_sim_df, 'obs': s_depth_obs_df}
            # aggregate the observed and simulated snow depth to monthly scale
            s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
            s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1)
            metrics['s_depth'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}


        return ds, metrics

    @ staticmethod
    def eval_fn(true: np.ndarray, pred: np.ndarray, cal_dim: int = 0):
        # check the dimensions of true and pred
        if len(true.shape) == 1 and len(pred.shape) == 1:
            true = true.reshape(1, -1)
            pred = pred.reshape(1, -1)
            cal_dim = 1
        assert true.ndim == 2 and pred.ndim == 2, 'The dimensions of true and pred should be 1 or 2.'
        # make sure the dtype is float
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

        # Calculate NSE
        nse_num = np.nansum((pred - true) ** 2, axis=cal_dim, keepdims=True)
        nse_den = np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True)
        nse_den[nse_den == 0] = 1e-5 # avoid division by zero
        nse = 1 - nse_num / nse_den

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

        return r, nse, rmse, kge, pbias


class EvalEnsemble:  # for ensemble model evaluation
    def __init__(self, folder: str, eval_vars: list[str]):
        # load the simulation results
        model_dict = {f[5:7]: f for f in os.listdir(f'{folder}') if f.startswith('seed')}
        self.models = list(model_dict.keys())
        self.train, self.valid, self.test = None, None, None  # the training, validation and test periods
        self.band_codes, self.basin_codes = None, None  # the band codes and basin codes
        self.obs = None # the observed data
        # loop over the models to evaluate
        sim = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        ds = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for model, path in model_dict.items():
            sim_tmp, metrics_tmp, ds_tmp = self.eval_model(f'{folder}/{path}', eval_vars)
            sim[model], metrics[model], ds[model] = sim_tmp, metrics_tmp, ds_tmp
        # calculate the ensemble mean simulation
        for period in ['valid', 'test']:
            for scale in ['band', 'basin']:
                for var in list(sim.values())[0][period][scale].keys():
                    sim_mean = np.mean(np.concatenate([np.expand_dims(sim[model][period][scale][var], 0) for
                                                       model in model_dict.keys()]), axis=0)
                    sim['mean'][period][scale][var] = sim_mean
        # calculate the ensemble mean metrics
        for period in ['valid', 'test']:
            ds['mean'][period], metrics['mean'][period] = self.evaluate(sim['mean'], eval_vars, period)

        self.sim, self.metrics, self.ds = sim, metrics, ds

    def eval_model(self, folder: str, eval_vars: list[str]):
        with open(f'{folder}/sim.pkl', 'rb') as f:
            sim = pickle.load(f)
        # get the training, validation and test periods
        with open(f'{folder}/config.json', 'rb') as f:
            config = json.load(f)
        if self.basin_codes is None:
            self.band_codes, self.basin_codes = self.get_basin_codes(config)
        if self.train is None:
            self.train = pd.date_range(start=config['data']['periods'][0][0], end=config['data']['periods'][0][1], freq='D')
        if self.valid is None:
            self.valid = pd.date_range(start=config['data']['periods'][1][0], end=config['data']['periods'][1][1], freq='D')
        if self.test is None:
            self.test = pd.date_range(start=config['data']['periods'][2][0], end=config['data']['periods'][2][1], freq='D')
        # get the observed data
        if self.obs is None:
            self.obs = self.get_obs(config, eval_vars)
        # get the metrics and dataset
        metrics, ds = dict(), dict()
        for period in ['valid', 'test']:
            ds[period], metrics[period] = self.evaluate(sim, eval_vars, period)
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
        path = self.cal_abs_path(os.path.join(config['data']['forcing_dir'], 'forcing_band.pkl'))
        forcing = pickle.load(open(path, 'rb'))
        forcing = OrderedDict(sorted(forcing.items()))
        band_codes = list(forcing.keys())
        basin_codes = sorted(list(set([band_code.split('_')[0] for band_code in band_codes])))
        return band_codes, basin_codes

    def get_obs(self, config, eval_vars):
        obs = dict()
        # load the observed glacier area
        if 'g_area' in eval_vars:
            g_area_path = self.cal_abs_path(config['data']['g_area_path'])
            g_area_obs = pd.read_csv(g_area_path, sep=r'\s+', dtype={'basin_id': str})
            g_area_obs['date'] = pd.to_datetime(g_area_obs['date'])
            g_area_obs['basin_id'] = g_area_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            # reorder the observed glacier area based on the given basin codes
            g_area_obs['basin_id'] = pd.Categorical(g_area_obs['basin_id'], categories=self.basin_codes, ordered=True)
            g_area_obs = g_area_obs.sort_values(by=['basin_id'])
            obs['g_area'] = g_area_obs

        # load the observed glacier volume during 2000-2019
        if 't_gvol' in eval_vars:
            assert (config['data']['gvol_path'].split('_')[-1]).split('.')[0] == 'basin', 'The glacier volume should be at basin scale.'
            # load the observed glacier volume during 2017-2018
            g_vol_path = self.cal_abs_path(config['data']['gvol_path'])
            g_vol_obs = pd.read_csv(g_vol_path, sep=r'\s+', dtype={'basin_id': str})
            g_vol_obs['basin_id'] = g_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            g_vol_obs['basin_id'] = pd.Categorical(g_vol_obs['basin_id'], categories=self.basin_codes, ordered=True)
            g_vol_obs = g_vol_obs.sort_values(by=['basin_id'])
            obs['g_vol'] = g_vol_obs
            # set the date as 2018-1-1
            gvol_obs = g_vol_obs['vol'].values

            # load the observed glacier volume change during 2000-2019
            d_gvol_path = self.cal_abs_path(config['data']['d_gvol_path'])
            d_gvol_obs = pd.read_csv(d_gvol_path, sep=r'\s+', index_col=0)
            d_gvol_obs.columns = [str(x).zfill(12) for x in d_gvol_obs.columns]
            d_gvol_obs = d_gvol_obs.reindex(columns=self.basin_codes) * 10 ** (-9)  # convert m^3 to km^3
            obs['d_gvol'] = d_gvol_obs

            # calculate the glacier volume change during 2000-2019 based on the glac_dvol_obs and g_vol_obs
            if 'Hugonnet' in config['data']['d_gvol_path']:
                dates = pd.date_range(start='2000-01-01', end='2018-12-31', freq='YS')
            else:
                dates = pd.date_range(start='1957-01-01', end='2018-12-31', freq='YS')
            dates = dates.append(pd.to_datetime(['2018-12-31', '2019-12-31']))
            # Create a DataFrame to store the gvol for each date
            t_gvol_obs = pd.DataFrame(index=dates, columns=d_gvol_obs.columns)
            # Set the gvol for 2018-1-1
            t_gvol_obs.loc['2018-01-01'] = gvol_obs
            # Calculate the gvol for the specified dates
            for date in dates:
                if date < pd.to_datetime('2018-01-01'):
                    t_gvol_obs.loc[date] = t_gvol_obs.loc['2018-01-01'] - d_gvol_obs.loc[date.year:2017].sum()
                elif date > pd.to_datetime('2018-01-01'):
                    t_gvol_obs.loc[date] = t_gvol_obs.loc['2018-01-01'] + d_gvol_obs.loc['2018':date.year].sum()
            t_gvol_obs[t_gvol_obs < 0] = 0
            t_gvol_obs[d_gvol_obs.columns[d_gvol_obs.isna().any()]] = np.nan
            obs['t_gvol'] = t_gvol_obs

        # load the observed glacier volume during 2017-2018
        if 'g_vol' in eval_vars and 'g_vol' not in obs.keys():
            gvol_path = self.cal_abs_path(config['data']['gvol_path'])
            g_vol_obs = pd.read_csv(gvol_path, sep=r'\s+', dtype={'basin_id': str})
            g_vol_obs['basin_id'] = g_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            g_vol_obs['basin_id'] = pd.Categorical(g_vol_obs['basin_id'], categories=self.basin_codes, ordered=True)
            g_vol_obs = g_vol_obs.sort_values(by=['basin_id'])
            obs['g_vol'] = g_vol_obs

        # load the observed glacier volume change
        if 'd_gvol' in eval_vars and 'd_gvol' not in obs.keys():
            d_gvol_path = self.cal_abs_path(config['data']['d_gvol_path'])
            d_gvol_obs = pd.read_csv(d_gvol_path, sep=r'\s+', index_col=0)
            d_gvol_obs.columns = [str(x).zfill(12) for x in d_gvol_obs.columns]
            d_gvol_obs = d_gvol_obs.reindex(columns=self.basin_codes) * 10 ** (-9)  # convert m^3 to km^3
            obs['d_gvol'] = d_gvol_obs

        # load the observed snow depth
        if 's_depth' in eval_vars:
            s_depth_path = self.cal_abs_path(config['data']['s_depth_path'])
            s_depth_obs = pd.read_csv(s_depth_path, sep=r'\s+', index_col=0, parse_dates=True)
            s_depth_obs.columns = [str(x).zfill(12) for x in s_depth_obs.columns]
            s_depth_obs = s_depth_obs.reindex(columns=self.basin_codes)
            obs['s_depth'] = s_depth_obs


        return obs

    def evaluate(self, sim, eval_vars, period='valid'):
        metrics, ds = dict(), dict()
        sim = sim[period]['basin']
        t_range = self.valid if period == 'valid' else self.test
        if 'g_area' in eval_vars:
            g_area_sim_df = pd.DataFrame(sim['g_area'].T, index=t_range, columns=self.basin_codes)
            obs_dates = self.obs['g_area']['date']
            g_area_sim = np.array([g_area_sim_df.loc[date, self.basin_codes[i]] if date in g_area_sim_df.index else np.nan
                                   for i, date in enumerate(obs_dates)])
            g_area_obs = self.obs['g_area']['area'].values
            obs_dates = obs_dates[~np.isnan(g_area_sim)]
            g_area_obs, g_area_sim = g_area_obs[~np.isnan(g_area_sim)], g_area_sim[~np.isnan(g_area_sim)]
            ds['g_area'] = pd.DataFrame({'obs': g_area_obs.astype(float), 'sim': g_area_sim.astype(float)}, index=obs_dates)
            # calculate the metrics
            if len(g_area_obs) > 0:
                r, nse, rmse, kge, pbias = self.eval_fn(true=g_area_obs, pred=g_area_sim, cal_dim=0)
                metrics['g_area'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 't_gvol' in eval_vars:
            gvol_obs_df = self.obs['t_gvol']
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim_df = gvol_sim_df.loc[gvol_sim_df.index.isin(gvol_obs_df.index)]
            gvol_obs_df = gvol_obs_df.loc[gvol_obs_df.index.isin(gvol_sim_df.index)]
            # gvol_obs, gvol_sim = gvol_obs_df.values.T.astype(float), gvol_sim_df.values.T.astype(float)
            gvol_obs = gvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            gvol_sim = gvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            ds['t_gvol'] = {'sim': gvol_sim_df, 'obs': gvol_obs_df}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=gvol_obs, pred=gvol_sim, cal_dim=0)
            metrics['t_gvol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'd_gvol' in eval_vars:
            dgvol_obs_df = self.obs['d_gvol']
            dgvol_obs_df = dgvol_obs_df.loc[dgvol_obs_df.index.isin(range(t_range[0].year, t_range[-1].year + 1))]
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=self.basin_codes)
            dgvol_sim_df = pd.DataFrame(index=dgvol_obs_df.index, columns=self.basin_codes)
            for year in range(t_range[0].year, t_range[-1].year+1):
                dgvol_sim_df.loc[year] = gvol_sim_df.loc[pd.to_datetime(f'{year}-12-31')] - gvol_sim_df.loc[pd.to_datetime(f'{year}-1-1')]
            # dgvol_obs, dgvol_sim = dgvol_obs_df.values.T.astype(float), dgvol_sim_df.values.T.astype(float)
            dgvol_obs = dgvol_obs_df.values.T.astype(float).mean(axis=1, keepdims=True)
            dgvol_sim = dgvol_sim_df.values.T.astype(float).mean(axis=1, keepdims=True)
            ds['d_gvol'] = {'sim': dgvol_sim_df.astype(float), 'obs': dgvol_obs_df.astype(float)}
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=dgvol_obs, pred=dgvol_sim, cal_dim=0)
            metrics['d_gvol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 'g_vol' in eval_vars and 2017 in range(t_range[0].year, t_range[-1].year):
            gvol_obs = self.obs['g_vol']['vol'].values
            # gvol_sim_df = pd.DataFrame(sim['g_vol'].T + sim['s_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim_df = pd.DataFrame(sim['g_vol'].T, index=t_range, columns=self.basin_codes)
            gvol_sim = gvol_sim_df[gvol_sim_df.index.year == 2017].values.mean(axis=0).T
            ds['g_vol'] = pd.DataFrame({'obs': gvol_obs, 'sim': gvol_sim}, index=self.basin_codes)
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=gvol_obs, pred=gvol_sim, cal_dim=0)
            metrics['g_vol'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        if 's_depth' in eval_vars:
            s_depth_obs_df = self.obs['s_depth'].loc[self.obs['s_depth'].index.isin(t_range)]
            s_depth_sim_df = pd.DataFrame(sim['s_depth'].T, index=t_range, columns=self.basin_codes)
            s_depth_sim_df = s_depth_sim_df.loc[s_depth_sim_df.index.isin(s_depth_obs_df.index)]
            ds['s_depth'] = {'sim': s_depth_sim_df, 'obs': s_depth_obs_df}
            # aggregate the observed and simulated snow depth to monthly scale
            s_depth_obs = s_depth_obs_df.resample('ME').mean().values.T
            s_depth_sim = s_depth_sim_df.resample('ME').mean().values.T
            # calculate the metrics
            r, nse, rmse, kge, pbias = self.eval_fn(true=s_depth_obs, pred=s_depth_sim, cal_dim=1)
            metrics['s_depth'] = {'r': r, 'nse': nse, 'rmse': rmse, 'kge': kge, 'pbias': pbias}

        return ds, metrics

    @ staticmethod
    def eval_fn(true: np.ndarray, pred: np.ndarray, cal_dim: int = 0):
        # check the dimensions of true and pred
        if len(true.shape) == 1 and len(pred.shape) == 1:
            true = true.reshape(1, -1)
            pred = pred.reshape(1, -1)
            cal_dim = 1
        assert true.ndim == 2 and pred.ndim == 2, 'The dimensions of true and pred should be 1 or 2.'
        # make sure the dtype is float
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

        # Calculate NSE
        nse_num = np.nansum((pred - true) ** 2, axis=cal_dim, keepdims=True)
        nse_den = np.nansum((true - true_mean) ** 2, axis=cal_dim, keepdims=True)
        nse_den[nse_den == 0] = 1e-5 # avoid division by zero
        nse = 1 - nse_num / nse_den

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

        return r, nse, rmse, kge, pbias


if __name__ == '__main__':
    folder = '../checkpoints/seed_19_seqL_1096_winSz_1096_freq_sm_loss_MSE_wLoss_1_0_1_1_0_0_sslmnn_False_gslmnn_False_t_0616-2325'
    evaluator = EvalSingle(folder=folder, eval_vars=['g_area', 't_gvol', 'd_gvol', 'g_vol', 's_depth'])

