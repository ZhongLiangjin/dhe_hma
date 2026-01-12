import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class LossFn(nn.Module):
    def __init__(self, g_area_path: str, gvol_path: str, d_gvol_path: str, s_depth_path: str,
                 w_loss: dict, basin_codes: list, band_codes: list, snow_scale: str = 'daily', metric: str = 'NSE'):
        """
        Initializes the LossFn module.
        All data loading and pre-processing are done here once to maximize performance.
        """
        super().__init__()
        self.eps = 1e-6  # A small epsilon to prevent division by zero
        # If weight for t_gvol is > 0, set g_vol weight to 0 to avoid double counting
        if w_loss.get('t_gvol', 0) > 0:
            w_loss['g_vol'] = 0
        self.w_loss = w_loss
        self.snow_scale = snow_scale
        assert metric in ['NSE', 'KGE', 'MSE'], 'The metric must be NSE, KGE, or MSE.'
        self.metric = metric

        # --- Pre-load and pre-process all observation data ---
        # This section converts pandas DataFrames to PyTorch tensors once at initialization.

        if self.w_loss.get('g_area', 0) > 0:
            g_area_obs_df = pd.read_csv(g_area_path, sep=r'\s+', dtype={'basin_id': str})
            g_area_obs_df['date'] = pd.to_datetime(g_area_obs_df['date'])
            g_area_obs_df['basin_id'] = g_area_obs_df['basin_id'].apply(lambda x: str(x).zfill(12))
            g_area_obs_df['basin_id'] = pd.Categorical(g_area_obs_df['basin_id'], categories=basin_codes, ordered=True)
            g_area_obs_df = g_area_obs_df.sort_values(by=['basin_id'])
            # Convert to PyTorch tensors on CPU. They will be moved to the correct device later.
            self.g_area_obs_tensor = torch.tensor(g_area_obs_df['area'].values, dtype=torch.float32)
            self.g_area_dates_numeric = torch.tensor(pd.to_numeric(g_area_obs_df['date'].values), dtype=torch.int64)

        if self.w_loss.get('t_gvol', 0) > 0:
            # This logic combines two datasets to create a time series of glacier volume
            g_vol_obs = pd.read_csv(gvol_path, sep=r'\s+', dtype={'basin_id': str})
            g_vol_obs['basin_id'] = g_vol_obs['basin_id'].apply(lambda x: str(x).zfill(12))
            g_vol_obs['basin_id'] = pd.Categorical(g_vol_obs['basin_id'], categories=basin_codes, ordered=True)
            g_vol_obs = g_vol_obs.sort_values(by=['basin_id'])
            gvol_obs_val = g_vol_obs['vol'].values

            d_gvol_obs = pd.read_csv(d_gvol_path, sep=r'\s+', index_col=0)
            d_gvol_obs.columns = [str(x).zfill(12) for x in d_gvol_obs.columns]
            d_gvol_obs = d_gvol_obs.reindex(columns=basin_codes) * 1e-9  # m^3 to km^3

            if 'Hugonnet' in d_gvol_path:
                dates = pd.date_range(start='2000-01-01', end='2018-12-31', freq='YS')
            else:
                dates = pd.date_range(start='1957-01-01', end='2018-12-31', freq='YS')
            dates = dates.append(pd.to_datetime(['2018-12-31', '2019-12-31']))
            t_gvol_obs_df = pd.DataFrame(index=dates, columns=d_gvol_obs.columns)
            t_gvol_obs_df.loc['2018-01-01'] = gvol_obs_val

            # Backfill and forward-fill volumes based on annual changes
            for date in dates:
                if date.year < 2018:
                    t_gvol_obs_df.loc[date] = t_gvol_obs_df.loc['2018-01-01'] - d_gvol_obs.loc[date.year:2017].sum()
                elif date.year > 2018:
                    t_gvol_obs_df.loc[date] = t_gvol_obs_df.loc['2018-01-01'] + d_gvol_obs.loc[2018:date.year - 1].sum()

            t_gvol_obs_df[t_gvol_obs_df < 0] = 0
            t_gvol_obs_df[d_gvol_obs.columns[d_gvol_obs.isna().any()]] = np.nan
            self.t_gvol_obs_dates = t_gvol_obs_df.index
            self.t_gvol_obs_tensor = torch.tensor(t_gvol_obs_df.values.T.astype(float), dtype=torch.float32)

        if self.w_loss.get('g_vol', 0) > 0:
            g_vol_obs_df = pd.read_csv(gvol_path, sep=r'\s+', dtype={'basin_id': str})
            if 'band' in gvol_path:
                self.g_vol_scale = 'band'
                g_vol_obs_df['band_id'] = g_vol_obs_df['band_id'].apply(lambda x: str(x).zfill(12))
                g_vol_obs_df['band_id'] = pd.Categorical(g_vol_obs_df['band_id'], categories=band_codes, ordered=True)
                g_vol_obs_df = g_vol_obs_df.sort_values(by=['band_id'])
            else:
                self.g_vol_scale = 'basin'
                g_vol_obs_df['basin_id'] = g_vol_obs_df['basin_id'].apply(lambda x: str(x).zfill(12))
                g_vol_obs_df['basin_id'] = pd.Categorical(g_vol_obs_df['basin_id'], categories=basin_codes,
                                                          ordered=True)
                g_vol_obs_df = g_vol_obs_df.sort_values(by=['basin_id'])
            self.g_vol_obs_tensor = torch.tensor(g_vol_obs_df['vol'].values, dtype=torch.float32)

        if self.w_loss.get('d_gvol', 0) > 0:
            d_gvol_obs_df = pd.read_csv(d_gvol_path, sep=r'\s+', index_col=0)
            d_gvol_obs_df.columns = [str(x).zfill(12) for x in d_gvol_obs_df.columns]
            d_gvol_obs_df = d_gvol_obs_df.reindex(columns=basin_codes) * 1e-9  # m^3 to km^3
            self.d_gvol_obs = d_gvol_obs_df  # Keep df for year indexing
            self.d_gvol_obs_tensor = torch.tensor(d_gvol_obs_df.values.T, dtype=torch.float32)

        if self.w_loss.get('s_depth', 0) > 0:
            s_depth_obs_df = pd.read_csv(s_depth_path, sep=r'\s+', index_col=0, parse_dates=True)
            s_depth_obs_df.columns = [str(x).zfill(12) for x in s_depth_obs_df.columns]
            self.s_depth_obs = s_depth_obs_df.reindex(columns=basin_codes)

    def _calculate_metric(self, pred: torch.Tensor, obs: torch.Tensor, cal_dim: int = -1):
        """
        A centralized function to calculate NSE, KGE, or MSE,
        """
        # Create a mask for valid (non-NaN) observation points
        mask = ~torch.isnan(obs)
        if mask.sum() < 2:
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
            # Use the user's preferred method for handling zero division
            denominator = torch.where(denominator == 0, self.eps, denominator)
            metric_val = 1 - (numerator / denominator)

        elif self.metric == 'KGE':
            # Correlation component (r)
            r_num = ((pred - pred_mean) * (obs_pad - obs_mean) * mask).sum(dim=cal_dim, keepdim=True)
            r_den_pred_sq_sum = ((pred - pred_mean) ** 2 * mask).sum(dim=cal_dim, keepdim=True)
            r_den_obs_sq_sum = ((obs_pad - obs_mean) ** 2 * mask).sum(dim=cal_dim, keepdim=True)
            r_den = torch.sqrt(r_den_pred_sq_sum * r_den_obs_sq_sum)
            # Use the user's preferred method for handling zero division
            r_den = torch.where(r_den == 0, self.eps, r_den)
            r = r_num / r_den

            # Bias component (alpha)
            # Use the user's preferred method for handling zero division
            safe_obs_mean = torch.where(obs_mean == 0, self.eps, obs_mean)
            alpha = pred_mean / safe_obs_mean

            # Variability component (beta)
            pred_std = torch.sqrt(r_den_pred_sq_sum / n)
            obs_std = torch.sqrt(r_den_obs_sq_sum / n)
            # Use the user's preferred method for handling zero division
            safe_obs_std = torch.where(obs_std == 0, self.eps, obs_std)
            beta = pred_std / safe_obs_std

            metric_val = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        else:  # MSE
            # This replicates the original code's scaled MSE (Sum-of-Squared-Errors / (StdDev+eps)^2)
            numerator = ((pred - obs_pad) ** 2 * mask).sum(dim=cal_dim, keepdim=True)
            obs_std = torch.sqrt(((obs_pad - obs_mean) ** 2 * mask).sum(dim=cal_dim, keepdim=True) / n)
            # Use the user's preferred method for handling zero division
            denominator = (obs_std + self.eps) ** 2
            metric_val = numerator / denominator

        return metric_val.nanmean()

    def cal_area(self, g_area_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        """Calculates loss for glacier area, ensuring perfect date alignment."""
        g_area_pred, ts = g_area_pred[:, spin_up_len:], ts[spin_up_len:]

        # Convert simulation timestamps to numeric tensor on the correct device
        ts_numeric = torch.tensor(pd.to_numeric(ts), device=g_area_pred.device, dtype=torch.int64)

        # Move pre-computed observation tensors to the correct device
        obs_dates_numeric = self.g_area_dates_numeric.to(g_area_pred.device)
        obs_area = self.g_area_obs_tensor.to(g_area_pred.device)

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

    def cal_dgvol(self, g_vol_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        """
        Calculates loss for annual glacier volume change.
        This version correctly finds the first and last available simulation day *within* each specific year.
        """
        g_vol_pred, ts = g_vol_pred[:, spin_up_len:], ts[spin_up_len:]

        # Filter years with sufficient observations (>300 days) to ensure robust calculation.
        year_counts = ts.year.value_counts()
        valid_years = year_counts[year_counts > 300].index
        obs_df_tmp = self.d_gvol_obs[self.d_gvol_obs.index.isin(valid_years)]
        if obs_df_tmp.empty:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

        # For the simulation timestamps `ts`, find the first and last index for each year.
        ts_series = pd.Series(np.arange(len(ts)), index=ts)
        year_indices = ts_series.groupby(ts.year).agg(['first', 'last'])

        # Align the found indices with the years from our observation data.
        aligned_indices = year_indices.reindex(obs_df_tmp.index).dropna()
        if aligned_indices.empty:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)
        start_indices = aligned_indices['first'].values.astype(int)
        end_indices = aligned_indices['last'].values.astype(int)

        # Filter the observation tensor to match only the years that are valid and available.
        valid_obs_years = aligned_indices.index
        obs = torch.tensor(obs_df_tmp.loc[valid_obs_years].values.T,
                           device=g_vol_pred.device, dtype=g_vol_pred.dtype)
        # Calculate predicted volume change using the correct start and end indices.
        pred = g_vol_pred[:, end_indices] - g_vol_pred[:, start_indices]

        # Filter out basins that have NaN values in observations and calculate the metric.
        non_nan_mask = ~torch.isnan(obs).any(dim=1)
        pred, obs = pred[non_nan_mask], obs[non_nan_mask]

        if pred.numel() == 0:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

        return self._calculate_metric(pred, obs, cal_dim=0)

    def cal_gvol(self, g_vol_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        g_vol_pred, ts = g_vol_pred[:, spin_up_len:], ts[spin_up_len:]
        cal_period = pd.date_range(start='2017-1-1', end='2018-1-1')
        ts_mask = ts.isin(cal_period)

        if ts_mask.sum() >= 365:
            pred = g_vol_pred[:, ts_mask].mean(dim=1)
            obs = self.g_vol_obs_tensor.to(pred.device)

            mask = ~torch.isnan(pred) & ~torch.isnan(obs)
            pred, obs = pred[mask], obs[mask]

            return self._calculate_metric(pred, obs, cal_dim=0)
        else:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

    def cal_tgvol(self, g_vol_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        g_vol_pred, ts = g_vol_pred[:, spin_up_len:], ts[spin_up_len:]

        # Find intersection of dates between simulation and observation
        obs_dates = self.t_gvol_obs_dates
        common_dates_mask = obs_dates.isin(ts)

        if not common_dates_mask.any():
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

        # Filter obs and preds to common dates
        obs = self.t_gvol_obs_tensor[:, common_dates_mask].to(g_vol_pred.device)
        pred = g_vol_pred[:, ts.isin(obs_dates[common_dates_mask])]

        non_nan_mask = ~torch.isnan(obs).any(dim=1)
        pred, obs = pred[non_nan_mask], obs[non_nan_mask]

        if obs.size(1) > 0:
            return self._calculate_metric(pred, obs, cal_dim=0)
        else:
            return torch.tensor(np.nan, device=g_vol_pred.device, dtype=g_vol_pred.dtype)

    def cal_sdep(self, s_depth_pred: torch.Tensor, ts: pd.DatetimeIndex, spin_up_len: int):
        """
        Calculates loss for snow depth.
        This version correctly performs monthly resampling using differentiable PyTorch operations.
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
            # Resample observations using pandas
            monthly_obs_df = obs_df_tmp.resample('ME').mean().dropna()
            if monthly_obs_df.empty:
                return torch.tensor(np.nan, device=s_depth_pred.device, dtype=s_depth_pred.dtype)

            # Perform resampling for predictions using pure PyTorch to maintain the computation graph
            monthly_pred_list = []
            for month_end_date in monthly_obs_df.index:
                # Create a boolean mask for each month
                mask = (t_range.year == month_end_date.year) & (t_range.month == month_end_date.month)
                # Use the mask to select daily predictions for the current month and calculate the mean
                # This operation is fully differentiable.
                monthly_mean = pred_tmp[:, mask].mean(dim=1)
                monthly_pred_list.append(monthly_mean)

            # Stack the monthly means into a single tensor
            pred = torch.stack(monthly_pred_list, dim=1)
            obs = torch.tensor(monthly_obs_df.values.T, device=pred.device, dtype=pred.dtype)
        else:  # Daily scale
            obs = torch.tensor(obs_df_tmp.values.T, device=s_depth_pred.device, dtype=s_depth_pred.dtype)
            pred = pred_tmp

        # Filter basins with no significant snow or with NaNs
        if pred.numel() == 0 or obs.numel() == 0:
            return torch.tensor(np.nan, device=s_depth_pred.device, dtype=s_depth_pred.dtype)

        non_nan_mask = ~torch.isnan(obs).any(dim=1)
        non_zero_mask = (obs > 0).float().mean(dim=1) > 0.05
        final_mask = non_nan_mask & non_zero_mask

        pred, obs = pred[final_mask], obs[final_mask]

        if pred.numel() == 0:
            return torch.tensor(np.nan, device=s_depth_pred.device, dtype=s_depth_pred.dtype)

        # Calculate the final metric
        cal_dim = 1 if obs.size(1) >= 24 else 0
        return self._calculate_metric(pred, obs, cal_dim=cal_dim)

    def forward(self, sim_band: dict, sim_bsn: dict, ts: pd.DatetimeIndex, spin_up_len: int):
        """
        The forward pass calculates the weighted loss based on the simulation outputs.
        """
        device, dtype = sim_bsn['g_area'].device, sim_bsn['g_area'].dtype
        nan_tensor = torch.tensor(np.nan, device=device, dtype=dtype)

        # Calculate metric for each variable if its weight is > 0
        metric_area = self.cal_area(sim_bsn['g_area'], ts, spin_up_len) if self.w_loss.get('g_area', 0) > 0 else nan_tensor
        metric_tgvol = self.cal_tgvol(sim_bsn['g_vol'], ts, spin_up_len) if self.w_loss.get('t_gvol', 0) > 0 else nan_tensor
        metric_dgvol = self.cal_dgvol(sim_bsn['g_vol'], ts, spin_up_len) if self.w_loss.get('d_gvol', 0) > 0 else nan_tensor
        metric_sdep = self.cal_sdep(sim_bsn['s_depth'], ts, spin_up_len) if self.w_loss.get('s_depth', 0) > 0 else nan_tensor
        if self.w_loss.get('g_vol', 0) > 0:
            g_vol_input = sim_band['g_vol'] if self.g_vol_scale == 'band' else sim_bsn['g_vol']
            metric_gvol = self.cal_gvol(g_vol_input, ts, spin_up_len)
        else:
            metric_gvol = nan_tensor

        # Combine metrics into a single tensor
        metrics = torch.stack([metric_area, metric_gvol, metric_dgvol, metric_tgvol, metric_sdep])
        metric_dict = {
            'g_area': metric_area.item(), 'g_vol': metric_gvol.item(),
            'd_gvol': metric_dgvol.item(), 't_gvol': metric_tgvol.item(), 's_depth': metric_sdep.item()
        }

        # Calculate weighted loss
        weights = torch.tensor(list(self.w_loss.values()), device=device, dtype=dtype)
        # Set weight to 0 for any metric that resulted in NaN
        weights = torch.where(torch.isnan(metrics), 0.0, weights)

        # Normalize weights so they sum to 1
        weights_sum = weights.sum()
        if weights_sum > self.eps:
            weights = weights / weights_sum
        else:  # Handle case where all metrics are NaN
            return nan_tensor, metric_dict

        # Use torch.nansum to safely compute the sum, ignoring any remaining NaNs
        if self.metric in ['NSE', 'KGE']:
            loss = 1 - torch.nansum(metrics * weights) if ~torch.isnan(metrics).all() else nan_tensor
        else:  # MSE
            loss = torch.nansum(metrics * weights) if ~torch.isnan(metrics).all() else nan_tensor

        return loss, metric_dict