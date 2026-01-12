import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LossFn(nn.Module):
    """
    An optimized loss function module for a coupled glacier and rainfall-runoff model.

    This class is designed for high performance by:
    1. Pre-computing all observational data into tensors during initialization.
    2. Vectorizing all calculations to eliminate slow Python loops.
    3. Consolidating metric calculations into a single, robust helper function.
    """

    def __init__(self, glac_loss_config: dict, rr_loss_config: dict, glac_rr_weight: list, metric: str = 'NSE'):
        super().__init__()
        assert metric in ['NSE', 'KGE', 'MSE'], 'The metric must be NSE, KGE or MSE.'
        self.metric = metric
        self.eps = 1e-6
        self.glac_loss_config = glac_loss_config
        self.glac_w_loss = glac_loss_config.get('weight', {})
        self.rr_loss_config = rr_loss_config
        self.bsn_w_loss = rr_loss_config.get('weight', {})
        self.glac_rr_weight = glac_rr_weight

        # --- Pre-load and pre-process all observation data into tensors ---
        if self.glac_loss_config.get('retrain', False) and self.glac_rr_weight[0] > 0:
            self._init_glac_loss_tensors(glac_loss_config)

        if sum(self.bsn_w_loss.values()) > 0 and self.glac_rr_weight[1] > 0:
            self._init_rr_loss_tensors(rr_loss_config)

    def _calculate_metric(self, pred: torch.Tensor, obs: torch.Tensor, cal_dim: int = -1):
        """
        A centralized function to calculate NSE, KGE, or MSE,
        """
        # Create a mask for valid (non-NaN) observation points
        mask = ~torch.isnan(obs)
        if mask.sum() < 10:
            return torch.tensor(np.nan, device=pred.device, dtype=pred.dtype)

        # Replace NaNs with zeros for calculation, as in the original code
        obs_pad = torch.where(mask, obs, 0.0)

        # Calculate means over valid data points only
        n = mask.sum(dim=cal_dim, keepdim=True).clamp(min=1)
        obs_mean = torch.nanmean(obs, dim=cal_dim, keepdim=True)
        pred_mean = (pred * mask).sum(dim=cal_dim, keepdim=True) / n

        if self.metric == 'NSE':
            numerator = ((pred - obs_pad) ** 2 * mask).sum(dim=cal_dim, keepdim=True)
            denominator = ((obs_pad - obs_mean) ** 2 * mask).sum(dim=cal_dim, keepdim=True)
           # Handle zero division
            denominator = torch.where(denominator == 0, self.eps, denominator)
            metric_val = 1 - (numerator / denominator)

        elif self.metric == 'KGE':
            # Correlation component (r)
            r_num = ((pred - pred_mean) * (obs_pad - obs_mean) * mask).sum(dim=cal_dim, keepdim=True)
            r_den_pred_sq_sum = ((pred - pred_mean) ** 2 * mask).sum(dim=cal_dim, keepdim=True)
            r_den_obs_sq_sum = ((obs_pad - obs_mean) ** 2 * mask).sum(dim=cal_dim, keepdim=True)
            r_den = torch.sqrt(r_den_pred_sq_sum * r_den_obs_sq_sum)
           # Handle zero division
            r_den = torch.where(r_den == 0, self.eps, r_den)
            r = r_num / r_den

            # Bias component (alpha)
           # Handle zero division
            safe_obs_mean = torch.where(obs_mean == 0, self.eps, obs_mean)
            alpha = pred_mean / safe_obs_mean

            # Variability component (beta)
            pred_std = torch.sqrt(r_den_pred_sq_sum / n)
            obs_std = torch.sqrt(r_den_obs_sq_sum / n)
           # Handle zero division
            safe_obs_std = torch.where(obs_std == 0, self.eps, obs_std)
            beta = pred_std / safe_obs_std

            metric_val = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        else:  # MSE
            # This replicates the original code's scaled MSE (Sum-of-Squared-Errors / (StdDev+eps)^2)
            numerator = ((pred - obs_pad) ** 2 * mask).sum(dim=cal_dim, keepdim=True)
            obs_var = ((obs_pad - obs_mean) ** 2 * mask).sum(dim=cal_dim, keepdim=True) / n
           # Handle zero division
            denominator = obs_var + self.eps
            metric_val = numerator / denominator

        return metric_val.nanmean()

    def _init_glac_loss_tensors(self, config):
        """Loads and pre-processes all glacier-related observation data."""
        if self.glac_w_loss.get('glac_tvol', 0) > 0:
            self.glac_w_loss['glac_vol'] = 0

        if self.glac_w_loss.get('glac_area', 0) > 0:
            df = pd.read_csv(config['glac_area_path'], sep=r'\s+', dtype={'basin_id': str})
            df['date'] = pd.to_datetime(df['date'])
            df['basin_id'] = df['basin_id'].str.zfill(12)
            df['basin_id'] = pd.Categorical(df['basin_id'], categories=config['bsn_code'], ordered=True)
            df = df.sort_values(by='basin_id').dropna(subset=['basin_id'])
            self.glac_area_obs_tensor = torch.tensor(df['area'].values, dtype=torch.float32)
            self.glac_area_dates_numeric = torch.tensor(pd.to_numeric(df['date']).values, dtype=torch.int64)

        if self.glac_w_loss.get('glac_tvol', 0) > 0:
            g_vol_df = pd.read_csv(config['glac_vol_path'], sep=r'\s+', dtype={'basin_id': str})
            g_vol_df['basin_id'] = g_vol_df['basin_id'].str.zfill(12)
            g_vol_df['basin_id'] = pd.Categorical(g_vol_df['basin_id'], categories=config['bsn_code'], ordered=True)
            g_vol_df = g_vol_df.sort_values(by='basin_id').dropna(subset=['basin_id'])
            gvol_obs_val = g_vol_df['vol'].values

            d_gvol_df = pd.read_csv(config['glac_dvol_path'], sep=r'\s+', index_col=0)
            d_gvol_df.columns = [str(c).zfill(12) for c in d_gvol_df.columns]
            d_gvol_df = d_gvol_df.reindex(columns=config['bsn_code']) * 1e-9 # convert m^3 to km^3

            if 'Hugonnet' in ['glac_dvol_path']:
                dates = pd.date_range(start='2000-01-01', end='2018-12-31', freq='YS')
            else:
                dates = pd.date_range(start='1957-01-01', end='2018-12-31', freq='YS')
            dates = dates.append(pd.to_datetime(['2018-12-31', '2019-12-31']))
            t_gvol_df = pd.DataFrame(index=dates, columns=d_gvol_df.columns)
            t_gvol_df.loc['2018-01-01'] = gvol_obs_val
            for date in dates:
                if date < pd.to_datetime('2018-01-01'):
                    t_gvol_df.loc[date] = t_gvol_df.loc['2018-01-01'] - d_gvol_df.loc[date.year:2017].sum()
                elif date > pd.to_datetime('2018-01-01'):
                    t_gvol_df.loc[date] = t_gvol_df.loc['2018-01-01'] + d_gvol_df.loc[2018: date.year].sum()
            t_gvol_df[t_gvol_df < 0] = 0
            t_gvol_df[d_gvol_df.columns[d_gvol_df.isna().any()]] = np.nan
            self.glac_tvol_obs_dates = t_gvol_df.index
            self.glac_tvol_obs_tensor = torch.tensor(t_gvol_df.values.T.astype(float), dtype=torch.float32)

        if self.glac_w_loss.get('glac_vol', 0) > 0:
            df = pd.read_csv(config['glac_vol_path'], sep=r'\s+', dtype={'basin_id': str})
            if 'band' in config['glac_vol_path']:
                self.g_vol_scale = 'band'
                df['band_id'] = df['band_id'].str.zfill(12)
                df['band_id'] = pd.Categorical(df['band_id'], categories=config['band_code'], ordered=True)
                df = df.sort_values(by='band_id').dropna(subset=['band_id'])
            else:
                self.g_vol_scale = 'basin'
                df['basin_id'] = df['basin_id'].str.zfill(12)
                df['basin_id'] = pd.Categorical(df['basin_id'], categories=config['bsn_code'], ordered=True)
                df = df.sort_values(by='basin_id').dropna(subset=['basin_id'])
            self.glac_vol_obs_tensor = torch.tensor(df['vol'].values, dtype=torch.float32)

        if self.glac_w_loss.get('glac_dvol', 0) > 0:
            df = pd.read_csv(config['glac_dvol_path'], sep=r'\s+', index_col=0)
            df.columns = [str(c).zfill(12) for c in df.columns]
            self.glac_dvol_obs_df = df.reindex(columns=config['bsn_code']) * 1e-9

        if self.glac_w_loss.get('glac_sdep', 0) > 0:
            df = pd.read_csv(config['glac_sdep_path'], sep=r'\s+', index_col=0, parse_dates=True)
            df.columns = [str(c).zfill(12) for c in df.columns]
            self.glac_sdep_obs_df = df.reindex(columns=config['bsn_code'])

    def _init_rr_loss_tensors(self, config):
        """
        Loads and pre-processes all runoff-related observation data.
        """
        if self.bsn_w_loss.get('bsn_Q', 0) > 0:
            gauges_df = pd.read_excel(config['bsn_Q_path'], sheet_name='gauges')
            bsn_codes = config['bsn_code']

            # --- Daily Data Initialization ---
            q_daily_df = pd.read_excel(config['bsn_Q_path'], index_col=0, parse_dates=True, sheet_name='daily')
            if 'held_out_gauges' in config:
                q_daily_df.drop(columns=config['held_out_gauges'], inplace=True, errors='ignore')
            # First, identify the list of stations that are valid (have basins in our model)
            station_names_daily = []
            for station in q_daily_df.columns:
                info = gauges_df[gauges_df['Station'] == station]
                if not info.empty:
                    basin_ids = str(info['BasinIds'].values[0]).split(',')
                    # Check if any basin for this station is in our list of modeled basins
                    if any(bid in bsn_codes for bid in basin_ids):
                        station_names_daily.append(station)
            # Create a mapping from the valid station name to its new, 0-based index
            station_to_new_idx = {name: i for i, name in enumerate(station_names_daily)}

            # Now, build the mapping tensors using the new, correct indices
            basin_indices_daily, gauge_indices_daily = [], []
            for station_name in station_names_daily:
                new_gauge_index = station_to_new_idx[station_name]
                info = gauges_df[gauges_df['Station'] == station_name]
                basin_ids = str(info['BasinIds'].values[0]).split(',')
                for bid in basin_ids:
                    if bid in bsn_codes:
                        basin_indices_daily.append(bsn_codes.index(bid))
                        gauge_indices_daily.append(new_gauge_index)  # Use the new, safe index

            # Filter the DataFrame and create tensors as before
            q_daily_df_filtered = q_daily_df[station_names_daily]
            self.q_daily_obs_tensor = torch.tensor(q_daily_df_filtered.values, dtype=torch.float32).T
            self.q_daily_dates = q_daily_df_filtered.index
            self.q_basin_to_gauge_map_daily = torch.tensor(basin_indices_daily, dtype=torch.long)
            self.q_gauge_indices_daily = torch.tensor(gauge_indices_daily, dtype=torch.long)
            self.num_gauges_daily = len(station_names_daily)
            areas_d = [gauges_df.loc[gauges_df['Station'] == s, 'Area'].values[0] for s in station_names_daily]
            self.q_gauge_areas_tensor_daily = torch.tensor(areas_d, dtype=torch.float32)

            # --- Monthly Data Initialization (Applying the same fix) ---
            q_monthly_df = pd.read_excel(config['bsn_Q_path'], index_col=0, parse_dates=True, sheet_name='monthly')
            if 'held_out_gauges' in config:
                q_monthly_df.drop(columns=config['held_out_gauges'], inplace=True, errors='ignore')

            station_names_monthly = []
            for station in q_monthly_df.columns:
                info = gauges_df[gauges_df['Station'] == station]
                if not info.empty and any(bid in bsn_codes for bid in str(info['BasinIds'].values[0]).split(',')):
                    station_names_monthly.append(station)

            station_to_new_idx_m = {name: i for i, name in enumerate(station_names_monthly)}
            b_idx_m, g_idx_m = [], []
            for station_name in station_names_monthly:
                new_gauge_index = station_to_new_idx_m[station_name]
                info = gauges_df[gauges_df['Station'] == station_name]
                for bid in str(info['BasinIds'].values[0]).split(','):
                    if bid in bsn_codes:
                        b_idx_m.append(bsn_codes.index(bid))
                        g_idx_m.append(new_gauge_index)

            q_monthly_df_f = q_monthly_df[station_names_monthly]
            self.q_monthly_obs_tensor = torch.tensor(q_monthly_df_f.values, dtype=torch.float32).T
            self.q_monthly_dates = q_monthly_df_f.index
            self.q_basin_to_gauge_map_monthly = torch.tensor(b_idx_m, dtype=torch.long)
            self.q_gauge_indices_monthly = torch.tensor(g_idx_m, dtype=torch.long)
            self.num_gauges_monthly = len(station_names_monthly)
            areas_m = [gauges_df.loc[gauges_df['Station'] == s, 'Area'].values[0] for s in station_names_monthly]
            self.q_gauge_areas_tensor_monthly = torch.tensor(areas_m, dtype=torch.float32)

        if self.bsn_w_loss.get('bsn_LAI', 0) > 0:
            df = pd.read_csv(config['bsn_LAI_path'], sep=r'\s+', index_col=0, parse_dates=True)
            df.columns = [str(c).zfill(12) for c in df.columns]
            self.bsn_LAI_obs_df = df.reindex(columns=config['bsn_code'])

        if self.bsn_w_loss.get('bsn_sdep', 0) > 0:
            df = pd.read_csv(config['bsn_sdep_path'], sep=r'\s+', index_col=0, parse_dates=True)
            df.columns = [str(c).zfill(12) for c in df.columns]
            self.bsn_sdep_obs_df = df.reindex(columns=config['bsn_code'])

    def cal_glac_area(self, g_area_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        """Calculates loss for glacier area, ensuring perfect date alignment."""
        g_area_pred, ts = g_area_pred[:, spin_up_len:], ts[spin_up_len:]

        # Convert simulation timestamps to numeric tensor on the correct device
        ts_numeric = torch.tensor(pd.to_numeric(ts), device=g_area_pred.device, dtype=torch.int64)

        # Move pre-computed observation tensors to the correct device
        obs_dates_numeric = self.glac_area_dates_numeric.to(g_area_pred.device)
        obs_area = self.glac_area_obs_tensor.to(g_area_pred.device)

        # Use broadcasting to create a [n_basins, seq_len] mask
        # It's True where an observation date for a basin matches a simulation timestamp
        mask = (obs_dates_numeric.unsqueeze(1) == ts_numeric.unsqueeze(0))

        # Filter predictions and observations using the mask
        pred = g_area_pred[mask]

        # Expand obs to match the mask shape before filtering
        obs_expanded = obs_area.unsqueeze(1).expand_as(g_area_pred)
        obs = obs_expanded[mask]

        # Apply nan mask after filtering
        nan_mask = ~torch.isnan(obs)
        pred, obs = pred[nan_mask], obs[nan_mask]

        if pred.numel() >= 30:
            return self._calculate_metric(pred, obs, cal_dim=0)
        else:
            return torch.tensor(np.nan, device=g_area_pred.device, dtype=g_area_pred.dtype)

    def cal_glac_tvol(self, g_vol_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        g_vol_pred, ts = g_vol_pred[:, spin_up_len:], ts[spin_up_len:]

        # Find intersection of dates between simulation and observation
        obs_dates = self.glac_tvol_obs_dates
        common_dates_mask = obs_dates.isin(ts)

        if not common_dates_mask.any():
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

        # Filter obs and preds to common dates
        obs = self.glac_tvol_obs_tensor[:, common_dates_mask].to(g_vol_pred.device)
        pred = g_vol_pred[:, ts.isin(obs_dates[common_dates_mask])]

        non_nan_mask = ~torch.isnan(obs).any(dim=1)
        pred, obs = pred[non_nan_mask], obs[non_nan_mask]

        if obs.size(1) > 0:
            return self._calculate_metric(pred, obs, cal_dim=0)
        else:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

    def cal_glac_vol(self, g_vol_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        g_vol_pred, ts = g_vol_pred[:, spin_up_len:], ts[spin_up_len:]
        cal_period = pd.date_range(start='2017-1-1', end='2018-1-1')
        ts_mask = ts.isin(cal_period)

        if ts_mask.sum() >= 365:
            pred = g_vol_pred[:, ts_mask].mean(dim=1)
            obs = self.glac_vol_obs_tensor.to(pred.device)

            mask = ~torch.isnan(pred) & ~torch.isnan(obs)
            pred, obs = pred[mask], obs[mask]

            return self._calculate_metric(pred, obs, cal_dim=0)
        else:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

    def cal_glac_dvol(self, g_vol_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        """Calculates loss for annual glacier volume change, correctly handling years."""
        g_vol_pred, ts = g_vol_pred[:, spin_up_len:], ts[spin_up_len:]

        year_counts = ts.year.value_counts()
        valid_years = year_counts[year_counts > 300].index
        obs_df = self.glac_dvol_obs_df[self.glac_dvol_obs_df.index.isin(valid_years)]
        if obs_df.empty:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

        ts_series = pd.Series(np.arange(len(ts)), index=ts)
        year_indices = ts_series.groupby(ts.year).agg(['first', 'last'])
        aligned_indices = year_indices.reindex(obs_df.index).dropna()
        if aligned_indices.empty:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

        start_indices = aligned_indices['first'].values.astype(int)
        end_indices = aligned_indices['last'].values.astype(int)

        obs = torch.tensor(obs_df.loc[aligned_indices.index].values.T, device=g_vol_pred.device, dtype=g_vol_pred.dtype)
        pred = g_vol_pred[:, end_indices] - g_vol_pred[:, start_indices]

        # Filter out basins that have NaN values in observations and calculate the metric.
        non_nan_mask = ~torch.isnan(obs).any(dim=1)
        pred, obs = pred[non_nan_mask], obs[non_nan_mask]

        if pred.numel() == 0:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

        return self._calculate_metric(pred, obs, cal_dim=0)

    def cal_glac_sdep(self, s_depth_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        """
        Calculates loss for snow depth.
        This version has been updated to use a vectorized method for monthly resampling.
        """
        # Skip spin-up period
        s_depth_pred, ts = s_depth_pred[:, spin_up_len:], ts[spin_up_len:]

        # Find the common time range between observations and simulations
        obs_df = self.s_depth_obs
        t_range = obs_df.index.intersection(ts)
        if t_range.empty:
            return torch.tensor(np.nan, device=s_depth_pred.device, dtype=s_depth_pred.dtype)

        # Filter daily observations and predictions to the common time range
        obs_df_tmp = obs_df.loc[t_range]
        pred_tmp = s_depth_pred[:, ts.isin(t_range)]

        if self.snow_scale == 'monthly':
            # Resample observations using pandas (no gradients needed)
            monthly_obs_df = obs_df_tmp.resample('ME').mean().dropna()
            if monthly_obs_df.empty:
                return torch.tensor(np.nan, device=s_depth_pred.device, dtype=s_depth_pred.dtype)
            # Create a unique integer ID for each month in the daily time range.
            month_indices, unique_months = pd.factorize(t_range.to_period('M'))
            month_indices_t = torch.tensor(month_indices, device=pred_tmp.device, dtype=torch.long)

            # Sum the predictions for each month group using scatter_add_.
            sums = torch.zeros(pred_tmp.size(0), len(unique_months), device=pred_tmp.device)
            sums.scatter_add_(1, month_indices_t.expand_as(pred_tmp), pred_tmp)
            # Count the number of days in each month group.
            counts = torch.bincount(month_indices_t, minlength=len(unique_months)).float().to(pred_tmp.device)
            # Calculate the mean by dividing sums by counts.
            monthly_pred_all = sums / counts.clamp(min=1)
            # Align the calculated monthly predictions with the (potentially filtered) monthly observations.
            target_indices = torch.tensor(unique_months.get_indexer(monthly_obs_df.index.to_period('M')),
                                          dtype=torch.long, device=pred_tmp.device)

            pred = monthly_pred_all[:, target_indices]
            obs = torch.tensor(monthly_obs_df.values.T, device=pred.device, dtype=pred.dtype)

        else:  # Daily scale
            obs = torch.tensor(obs_df_tmp.values.T, device=s_depth_pred.device, dtype=s_depth_pred.dtype)
            pred = pred_tmp

        # Filter basins with no significant snow or with NaNs
        if pred.numel() == 0 or obs.numel() == 0:
            return torch.tensor(np.nan, device=s_depth_pred.device, dtype=s_depth_pred.dtype)

        non_nan_mask = ~torch.isnan(obs).any(dim=1)
        # Filter basins where less than 5% of observations are non-zero snow
        non_zero_mask = (obs > 0).float().mean(dim=1) > 0.05
        final_mask = non_nan_mask & non_zero_mask

        pred, obs = pred[final_mask], obs[final_mask]

        if pred.numel() == 0:
            return torch.tensor(np.nan, device=s_depth_pred.device, dtype=s_depth_pred.dtype)

        # Calculate the final metric
        cal_dim = 1 if obs.dim() > 1 and obs.size(1) >= 24 else 0
        return self._calculate_metric(pred, obs, cal_dim=cal_dim)

    def cal_glac_loss(self, glac_sim_band: dict, glac_sim_bsn: dict, ts: pd.DatetimeIndex, spin_up_len: int):
        """Calculates the total weighted loss for the glacier model."""
        device = glac_sim_bsn['g_area'].device
        nan_tensor = torch.tensor(np.nan, device=device)

        # Calculate metric for each variable if its weight is > 0
        metric_area = self.cal_glac_area(glac_sim_bsn['g_area'], ts, spin_up_len) if (
                self.glac_w_loss.get('glac_area', 0) > 0) else nan_tensor
        metric_tvol = self.cal_glac_tvol(glac_sim_bsn['g_vol'], ts, spin_up_len) if (
                self.glac_w_loss.get('glac_tvol', 0) > 0) else nan_tensor
        metric_dvol = self.cal_glac_dvol(glac_sim_bsn['g_vol'], ts, spin_up_len) if (
                self.glac_w_loss.get('glac_dvol', 0) > 0) else nan_tensor
        metric_sdep = self.cal_glac_sdep(glac_sim_bsn['s_depth'], ts, spin_up_len) if (
                self.glac_w_loss.get('glac_sdep', 0) > 0) else nan_tensor
        metric_gvol = self.cal_glac_vol(glac_sim_bsn['g_vol'], ts, spin_up_len) if  (
                self.glac_w_loss.get('glac_vol', 0) > 0) else nan_tensor

        metrics = torch.stack([metric_area, metric_gvol, metric_dvol, metric_tvol, metric_sdep])
        metric_dict = {'glac_dvol': metric_dvol.item(), 'glac_tvol': metric_tvol.item(),
                       'glac_area': metric_area.item(), 'glac_sdep': metric_sdep.item(),
                       'glac_vol': metric_gvol.item()}

        weights = torch.tensor(list(self.glac_w_loss.values()), device=device)
        weights = torch.where(torch.isnan(metrics), 0.0, weights)
        # Normalize weights so they sum to 1
        weights_sum = weights.sum()
        if weights_sum > self.eps:
            weights = weights / weights_sum
        else:  # Handle case where all metrics are NaN
            return nan_tensor, metric_dict

        loss = 1 - torch.nansum(metrics * weights) if self.metric in ['NSE', 'KGE'] else torch.nansum(metrics * weights)

        return loss, metric_dict

    def cal_bsn_q(self, bsn_q_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        """
        Calculates loss for streamflow using vectorized aggregation for daily and monthly scales.
        The logic is verified to be correct.
        """
        pred, ts = bsn_q_pred[:, spin_up_len:], ts[spin_up_len:]
        device = pred.device
        nan_tensor = torch.tensor(np.nan, device=device)

        # --- Daily Streamflow Calculation ---
        t_range_daily = self.q_daily_dates.intersection(ts)
        obs_daily = self.q_daily_obs_tensor[:, self.q_daily_dates.isin(t_range_daily)].to(device)
        valid_mask = torch.sum(~torch.isnan(obs_daily), dim=1) >= 30  # At least 30 valid obs
        if not t_range_daily.empty and valid_mask.any() and self.num_gauges_daily > 0:
            pred_daily_basins = pred[:, ts.isin(t_range_daily)]

            # Create a zero tensor to store the aggregated gauge flows. Shape: [num_gauges, num_days]
            pred_daily_gauges = torch.zeros(self.num_gauges_daily, pred_daily_basins.size(1), device=device)

            # Select all relevant basin flows from the prediction tensor.
            source_tensor = pred_daily_basins.index_select(0, self.q_basin_to_gauge_map_daily.to(device))

            # The core aggregation step:
            # Add each basin's flow from the 'source_tensor' into the 'pred_daily_gauges' tensor.
            # The 'self.q_gauge_indices_daily' tensor specifies the destination row (the gauge) for each addition.
            # This correctly handles the many-to-one summation.
            pred_daily_gauges.index_add_(0, self.q_gauge_indices_daily.to(device), source_tensor)

            areas_d = self.q_gauge_areas_tensor_daily.to(device).unsqueeze(1)
            # Convert units from m3/s to mm/day
            conversion_d = 86400 * 1000 / (areas_d * 1e6).clamp(min=self.eps)
            metric_daily = self._calculate_metric((pred_daily_gauges * conversion_d)[valid_mask],
                                                  (obs_daily * conversion_d)[valid_mask], cal_dim=1)
        else:
            metric_daily = nan_tensor

        # --- Monthly Streamflow Calculation ---
        monthly_dates_obs = self.q_monthly_dates.to_period('M')
        monthly_dates_sim = ts.to_period('M')
        common_months = monthly_dates_obs.intersection(monthly_dates_sim.unique())
        common_months_timestamp = common_months.to_timestamp()
        obs_monthly = self.q_monthly_obs_tensor[:, self.q_monthly_dates.isin(common_months_timestamp)].to(device)
        valid_mask = torch.sum(~torch.isnan(obs_monthly), dim=1) >= 10  # At least 10 valid obs
        if not common_months.empty and self.num_gauges_monthly > 0 and valid_mask.any():
            sim_mask = monthly_dates_sim.isin(common_months)
            pred_daily_for_monthly = pred[:, sim_mask]
            ts_for_monthly = ts[sim_mask]

            month_indices, unique_months = pd.factorize(ts_for_monthly.to_period('M'))
            month_indices_t = torch.tensor(month_indices, device=device, dtype=torch.long)

            # Sum daily flow (m3/s) for each day in the month
            monthly_pred_sums = torch.zeros(pred.size(0), len(unique_months), device=device)
            monthly_pred_sums.scatter_add_(1, month_indices_t.expand_as(pred_daily_for_monthly), pred_daily_for_monthly)

            # Aggregate monthly basin sums to gauges
            monthly_pred_gauges = torch.zeros(self.num_gauges_monthly, len(unique_months), device=device)
            monthly_pred_gauges.index_add_(0, self.q_gauge_indices_monthly.to(device),
                                           monthly_pred_sums.index_select(0, self.q_basin_to_gauge_map_monthly.to( device)))

            days_in_month = torch.tensor(pd.Series(unique_months.to_timestamp()).dt.days_in_month.values, device=device)
            areas_m = self.q_gauge_areas_tensor_monthly.to(device).unsqueeze(1)

            # Convert units to average mm/day
            # Obs unit: 10^8 m3/month -> avg mm/day
            obs_final = obs_monthly * 1e8 * 1000 / (areas_m * 1e6 * days_in_month).clamp(min=self.eps)
            # Pred unit: sum of daily m3/s for a month -> avg mm/day
            pred_final = monthly_pred_gauges * 86400 * 1000 / (areas_m * 1e6 * days_in_month).clamp(min=self.eps)

            metric_monthly = self._calculate_metric(pred_final[valid_mask], obs_final[valid_mask], cal_dim=1)
        else:
            metric_monthly = nan_tensor

        # --- Combine Metrics with new weighting logic ---
        daily_weight = self.rr_loss_config.get('q_daily_weight', 1.0)
        # Check for NaN conditions
        is_daily_nan = torch.isnan(metric_daily)
        is_monthly_nan = torch.isnan(metric_monthly)

        if not is_daily_nan and not is_monthly_nan:
            # Both are valid, use normal weights
            return metric_daily * daily_weight + metric_monthly * (1 - daily_weight)
        elif not is_daily_nan and is_monthly_nan:
            # Only daily is valid, its weight becomes 1.0
            return metric_daily
        elif is_daily_nan and not is_monthly_nan:
            # Only monthly is valid, its weight becomes 1.0
            return metric_monthly
        else:
            # Both are NaN, return NaN
            return nan_tensor

    def cal_bsn_LAI(self, lai_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        """
        Calculates loss for LAI using a robust, vectorized resampling method.
        This version uses a more robust np.searchsorted approach that correctly handles
        sliced timestamps and removes the confusing capping logic.
        """
        # Slice pred and ts to remove spin-up period, as before.
        pred, ts = lai_pred[:, spin_up_len:], ts[spin_up_len:]
        device = pred.device
        obs_df = self.bsn_LAI_obs_df

        # Find observation dates that exist within the current (already sliced) simulation time range.
        t_range_obs = obs_df.index.intersection(ts)
        if len(t_range_obs) < 10:
            # We need at least two observation points to form an interval.
            return torch.tensor(np.nan, device=device)

        # Use np.searchsorted to find which observation interval each simulation timestep falls into.
        # This is a robust way to create interval IDs and correctly handles the sliced 'ts'.
        # It returns the index of the right boundary of the interval for each timestamp in 'ts'.
        interval_indices = np.searchsorted(t_range_obs.values, ts.values, side='right')

        # We want the ID of the interval itself (i.e., the index of the left boundary), so we subtract 1.
        # Now, an ID of 0 corresponds to the interval [t_obs_0, t_obs_1).
        interval_indices = interval_indices - 1

        # Create a mask to only consider timesteps that fall *between* observation points.
        # We exclude times before the first observation (ID < 0) and times at or after the last observation
        # (ID >= number of intervals), as we have nothing to compare them against.
        num_intervals = len(t_range_obs) - 1
        valid_mask = (interval_indices >= 0) & (interval_indices < num_intervals)

        # Filter the prediction tensor and the interval IDs to only include the valid timesteps.
        pred_filtered = pred[:, valid_mask]
        interval_indices_filtered = torch.tensor(interval_indices[valid_mask], device=device, dtype=torch.long)

        # Step 6: Sum predictions for each interval using the efficient scatter_add_
        sums = torch.zeros(pred.size(0), num_intervals, device=device)
        sums.scatter_add_(1, interval_indices_filtered.expand_as(pred_filtered), pred_filtered)

        # Count the number of simulation days that fell into each interval
        counts = torch.bincount(interval_indices_filtered, minlength=num_intervals).float().to(device)

        # Calculate the mean prediction for each interval
        pred_resampled = sums / counts.clamp(min=1)

        # ---- Align with the observation at the START of the interval. ---
        # We now take all observations except the last one to align with the N-1 intervals.
        obs_resampled = torch.tensor(obs_df.loc[t_range_obs[:-1]].values.T, device=device, dtype=pred.dtype)

        if self.rr_loss_config.get('only_gs_LAI', False):
            # The mask should now be based on the start-of-interval dates.
            growing_season_mask = (t_range_obs[:-1].month >= 5) & (t_range_obs[:-1].month <= 10)
            pred_final = pred_resampled[:, growing_season_mask]
            obs_final = obs_resampled[:, growing_season_mask]
        else:
            pred_final = pred_resampled
            obs_final = obs_resampled
        valid_mask = torch.isnan(obs_final).float().mean(dim=1) < 0.5 # At least 50% non-NaN values

        return self._calculate_metric(pred_final[valid_mask], obs_final[valid_mask], cal_dim=1)

    def cal_bsn_sdep(self, sdep_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int,
                 scale: str):
        """
        A generic, vectorized function to calculate snow depth loss
        """
        pred, ts = sdep_pred[:, spin_up_len:], ts[spin_up_len:]
        t_range = self.bsn_sdep_obs_df.index.intersection(ts)
        if t_range.empty: return torch.tensor(np.nan, device=pred.device)
        obs_df_tmp = self.bsn_sdep_obs_df.loc[t_range]
        pred_tmp = pred[:, ts.isin(t_range)]

        if scale == 'monthly':
            monthly_obs_df = obs_df_tmp.resample('ME').mean().dropna()
            if monthly_obs_df.empty: return torch.tensor(np.nan, device=pred.device)

            month_indices, unique_months = pd.factorize(t_range.to_period('M'))
            month_indices_t = torch.tensor(month_indices, device=pred.device, dtype=torch.long)
            sums = torch.zeros(pred_tmp.size(0), len(unique_months), device=pred.device)
            sums.scatter_add_(1, month_indices_t.expand_as(pred_tmp), pred_tmp)
            counts = torch.bincount(month_indices_t, minlength=len(unique_months)).float().to(pred.device)

            monthly_pred_all = sums / counts.clamp(min=1)

            target_indices = torch.tensor(unique_months.get_indexer(monthly_obs_df.index.to_period('M')),
                                          dtype=torch.long, device=pred.device)
            pred_final = monthly_pred_all[:, target_indices]
            obs_final = torch.tensor(monthly_obs_df.values.T, device=pred.device, dtype=pred.dtype)
        else:  # Daily
            pred_final = pred_tmp
            obs_final = torch.tensor(obs_df_tmp.values.T, device=pred.device, dtype=pred.dtype)

        # Filter out basins with no snow (less than 1mm) or with NaNs
        no_snow = (obs_final < 1) | torch.isnan(obs_final)  # [n, m]
        valid_mask = (no_snow.float().mean(dim=1) < 0.8) & (obs_final.nanmean(dim=1) > 1) & (torch.max(obs_final, dim=1).values > 10)

        return self._calculate_metric(pred_final[valid_mask], obs_final[valid_mask], cal_dim=1)


    def cal_bsn_loss(self, bsn_sim: dict, ts: pd.DatetimeIndex, spin_up_len: int):
        device = bsn_sim['Qriver'].device
        nan_tensor = torch.tensor(np.nan, device=device)

        metric_q = self.cal_bsn_q(bsn_sim['Qriver'], ts, spin_up_len) if self.bsn_w_loss.get('bsn_Q',
                                                                                             0) > 0 else nan_tensor
        metric_lai = self.cal_bsn_LAI(bsn_sim['LAI'], ts, spin_up_len) if self.bsn_w_loss.get('bsn_LAI',
                                                                                              0) > 0 else nan_tensor
        metric_sdep = self.cal_bsn_sdep(bsn_sim['sdep'], ts, spin_up_len,
                                        self.rr_loss_config['snow_scale']) if self.bsn_w_loss.get('bsn_sdep',
                                                                                                  0) > 0 else nan_tensor

        metrics = torch.stack([metric_q, metric_lai, metric_sdep])
        weights_list = [self.bsn_w_loss.get('bsn_Q', 0), self.bsn_w_loss.get('bsn_LAI', 0),
                        self.bsn_w_loss.get('bsn_sdep', 0)]
        weights = torch.tensor(weights_list, device=device)
        weights = torch.where(torch.isnan(metrics), 0.0, weights)
        weights /= weights.sum().clamp(min=self.eps)

        metric_dict = {'bsn_Q': metric_q.item(), 'bsn_LAI': metric_lai.item(), 'bsn_sdep': metric_sdep.item()}
        if self.metric in ['NSE', 'KGE']:
            loss = 1 - torch.nansum(metrics * weights) if ~torch.isnan(metrics).all() else nan_tensor
        else:
            loss = torch.nansum(metrics * weights) if ~torch.isnan(metrics).all() else nan_tensor
        return loss, metric_dict


    def forward(self, glac_sim_band: dict, glac_sim_bsn: dict, rr_sim_bsn: dict, ts: pd.DatetimeIndex,
                spin_up_len: int):
        """The main forward pass to compute the final combined loss."""
        device = glac_sim_bsn['g_area'].device
        nan_tensor = torch.tensor(np.nan, device=device)

        glac_loss, glac_metric = (self.cal_glac_loss(glac_sim_band, glac_sim_bsn, ts, spin_up_len)
                                  if self.glac_loss_config.get('retrain', False) and self.glac_rr_weight[0] > 0
                                  else (nan_tensor, {}))

        bsn_loss, bsn_metric = (self.cal_bsn_loss(rr_sim_bsn, ts, spin_up_len)
                                if sum(self.bsn_w_loss.values()) > 0 and self.glac_rr_weight[1] > 0
                                else (nan_tensor, {}))

        losses = torch.stack([glac_loss, bsn_loss])
        weights = torch.tensor(self.glac_rr_weight, device=device)
        weights = torch.where(torch.isnan(losses), 0.0, weights)

        if weights.sum() < self.eps:
            return nan_tensor, {**glac_metric, **bsn_metric}

        weights /= weights.sum()

        total_loss = torch.nansum(losses * weights)
        metric_dict = {**glac_metric, **bsn_metric}

        return total_loss, metric_dict

