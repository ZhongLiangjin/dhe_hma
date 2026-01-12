import json
import logging
import pickle
import pandas as pd
import numpy as np
import os
from typing import Union, List
from datetime import datetime
from collections import OrderedDict
import torch
from torch.utils.data import dataset, dataloader


class Loader:
    def __init__(self, forcing_dir: str, attr_dir: str, acr_path: str,  periods: List[List[str]],  seq_len: int,
                 spin_up_len: int, win_sz: int, device: Union[str, torch.device] = 'cpu', logger: bool = False):
        # define the time range for training, validation and testing
        self.train = pd.date_range(start=periods[0][0], end=periods[0][1], freq='D')
        self.valid = pd.date_range(start=periods[1][0], end=periods[1][1], freq='D')
        self.test = pd.date_range(start=periods[2][0], end=periods[2][1], freq='D')
        self.t_range = pd.date_range(start=self.train[0], end=self.test[-1], freq='D')
        self.logger = logger

        # load forcing data and glacier attributes
        band_forcing = self.load_forcing(forcing_dir=forcing_dir)
        self.band_codes = list(band_forcing.keys())
        self.band_attrs = self.load_attrs(attr_dir=attr_dir)

        # calculate the delta year between the observation date and the start of the time range
        self.g_area_band_t0 = self.cal_init_glacier_area(acr_path=acr_path, spin_up_len=spin_up_len, device=device)
        self.band_attrs.drop(columns=['date'], inplace=True)

        # link the elevation band id to the basin id
        self.bsn_band_ids_dict, self.bsn_codes = self.link_band_basin()
        # calculate the area of each basin
        self.area_bsn = dict()
        for bsn_id, band_ids in self.bsn_band_ids_dict.items():
            self.area_bsn[bsn_id] = np.sum(self.band_attrs['area'].values[band_ids])
        self.band_elev = self.band_attrs['elev'].values
        self.band_slope = self.band_attrs['slope'].values
        # get the dataset
        data_trn, data_val, data_tst = self.split_dataset(band_forc=band_forcing)

        # normalize the forcing data and attributes
        self.forc_vars = data_trn['forc_vars']
        self.attr_vars = data_trn['attr_vars']
        # calculate mean and std for normalization
        self.stat_dict = self.cal_mean_std(forc_trn=data_trn['forc'], forc_var_lst=data_trn['forc_vars'],
                                           attr=data_trn['attrs'], attr_var_lst=data_trn['attr_vars'])
        for ds in [data_trn, data_val, data_tst]:
            ds['forc_norm'] = self.trans_norm(x=ds['forc'], var_lst=ds['forc_vars'], to_norm=True)
            ds['attrs_norm'] = self.trans_norm(x=ds['attrs'], var_lst=ds['attr_vars'], to_norm=True)

        # generate sequences
        data_all, data_trn, data_val, data_tst = self.gen_seq(data_trn=data_trn, data_val=data_val, data_tst=data_tst,
                                                              seq_len=seq_len, win_sz=win_sz, spin_up_len=spin_up_len)

        # get dataset and dataloader
        ds_all = MyDataset(data_dict=data_all, device=device)
        ds_trn = MyDataset(data_dict=data_trn, device=device)
        ds_val = MyDataset(data_dict=data_val, device=device)
        ds_tst = MyDataset(data_dict=data_tst, device=device)
        self.loader_all = dataloader.DataLoader(dataset=ds_all, batch_size=1, shuffle=False)
        self.loader_trn = dataloader.DataLoader(dataset=ds_trn, batch_size=1, shuffle=False)
        self.loader_val = dataloader.DataLoader(dataset=ds_val, batch_size=1, shuffle=False)
        # self.loader_tst = dataloader.DataLoader(dataset=ds_tst, batch_size=1, shuffle=False)

        self.print(f'Loader initialized successfully.')

    def load_forcing(self, forcing_dir: str):
        self.print(f'Loading forcing data...')
        if os.path.exists(os.path.join(forcing_dir, 'forc_band.pkl')):
            forcing = pickle.load(open(os.path.join(forcing_dir, 'forc_band.pkl'), 'rb'))
            for k, v in forcing.items():
                forcing[k] = v[v.index.isin(self.t_range)]
        else:
            # initialize lists to store the elevation band name and corresponding forcing data
            forcing = dict()
            folders = os.listdir(forcing_dir)  # debris-covered and debris-free glacier
            for folder in folders:
                files = os.listdir(os.path.join(forcing_dir, folder))
                for i, file in enumerate(files):
                    if i % 100 == 0:
                        self.print(f'Processing {i}/{len(files)} files in folder {folder}')
                    if file.endswith('.txt'):
                        df = pd.read_csv(os.path.join(forcing_dir, folder, file), parse_dates=True,
                                         index_col=0, header=0, sep='\s+')
                        forcing[file.split('.')[0]] = df
            # reorder the forcing data by the elevation band name,
            # make sure the band with lower elevation is in the front and the debris free glacier is in the front
            forcing = OrderedDict(sorted(forcing.items()))
            # save the forcing data
            with open(os.path.join(forcing_dir, 'forc_band.pkl'), 'wb') as f:
                f.write(pickle.dumps(forcing))
            # filter the forcing data by the time range
            for k, v in forcing.items():
                forcing[k] = v[v.index.isin(self.t_range)]
        return forcing

    def load_attrs(self, attr_dir: str):
        self.print(f'Loading glacier attributes...')
        debris_free_attrs = pd.read_csv(f'{attr_dir}/glac_clean_attrs.txt', index_col=0, sep=r'\s+')
        debris_covered_attrs = pd.read_csv(f'{attr_dir}/glac_debris_attrs.txt', index_col=0, sep=r'\s+')
        glac_attrs = pd.concat([debris_free_attrs, debris_covered_attrs])
        # reorder the attributes by the elevation band name
        glac_attrs = glac_attrs.loc[self.band_codes]
        return glac_attrs

    def get_dataset(self, band_forcing, spin_up_len):
        self.print(f'Getting datasets...')
        # get forcing data
        forc = []
        spin_forc = pd.date_range(start=self.t_range[0], end=self.t_range[0] + pd.Timedelta(days=spin_up_len - 1), freq='D')
        for k, v in band_forcing.items():
            v['doy'] = v.index.dayofyear
            forc.append(np.expand_dims(pd.concat([v.loc[spin_forc], v.loc[self.t_range]]).values, axis=0))

        forc = np.concatenate(forc, axis=0)
        forc_vars = list(band_forcing.values())[0].columns.tolist()
        # get time numerical value
        time = np.array(list(pd.to_numeric(spin_forc)) + list(pd.to_numeric(self.t_range)))
        # get glacier attributes
        attrs = self.band_attrs.values
        atts_vars = self.band_attrs.columns.tolist()

        # create data dictionary
        ds = {'forc': forc, 'attrs': attrs, 'time': time, 'forc_vars': forc_vars, 'attr_vars': atts_vars}

        return ds

    def split_dataset(self, band_forc):
        self.print(f'Splitting data into different sets...')
        # split forcing data
        forc_trn, forc_val, forc_tst = [], [], []
        # [seq_len, n_var] -> [n_band, seq_len, n_var]
        for k, v in band_forc.items():  # loop for each band
            v['doy'] = v.index.dayofyear
            forc_trn.append(np.expand_dims(v.loc[self.train].values, axis=0))
            forc_val.append(np.expand_dims(v.loc[self.valid].values, axis=0))
            forc_tst.append(np.expand_dims(v.loc[self.test].values, axis=0))
        forc_trn = np.concatenate(forc_trn, axis=0)
        forc_val = np.concatenate(forc_val, axis=0)
        forc_tst = np.concatenate(forc_tst, axis=0)
        # split time
        time_trn = pd.to_numeric(self.train)
        time_val = pd.to_numeric(self.valid)
        time_tst = pd.to_numeric(self.test)

        # get forcing variables
        forc_vars = list(band_forc.values())[0].columns.tolist()
        # get glacier attributes
        attrs = self.band_attrs.values
        attr_vars = self.band_attrs.columns.tolist()

        # create data dictionary
        data_trn = {'forc': forc_trn, 'attrs': attrs, 'time': time_trn, 'forc_vars': forc_vars, 'attr_vars': attr_vars}
        data_val = {'forc': forc_val, 'attrs': attrs, 'time': time_val, 'forc_vars': forc_vars, 'attr_vars': attr_vars}
        data_tst = {'forc': forc_tst, 'attrs': attrs, 'time': time_tst, 'forc_vars': forc_vars, 'attr_vars': attr_vars}

        return data_trn, data_val, data_tst

    def gen_seq(self, data_trn, data_val, data_tst, seq_len: int, win_sz: int, spin_up_len: int):
        # initialize the number of sequences for the training period
        num_seq = int((data_trn['forc'].shape[1] - seq_len) / win_sz) + 1
        forc_trn = np.zeros((num_seq, data_trn['forc'].shape[0], seq_len + spin_up_len, data_trn['forc'].shape[2]))
        forc_norm_trn = np.zeros((num_seq, data_trn['forc'].shape[0], seq_len + spin_up_len, data_trn['forc'].shape[2]))
        time_trn = np.zeros((num_seq, seq_len + spin_up_len))
        # get the sequences for the training period
        idx_start = 0
        for i in range(num_seq):
            # get the spin-up forcing data
            if idx_start < spin_up_len:
                if idx_start == 0:
                    forc_spin = data_trn['forc'][:, idx_start - spin_up_len:, :]
                    forc_norm_spin = data_trn['forc_norm'][:, idx_start - spin_up_len:, :]
                    time_spin = data_trn['time'][idx_start - spin_up_len:]
                else:
                    forc_spin = np.concatenate((data_trn['forc'][:, idx_start - spin_up_len:, :],
                                                data_trn['forc'][i-1, :, :idx_start, :]), axis=1)
                    forc_norm_spin = np.concatenate((data_trn['forc_norm'][:, idx_start - spin_up_len:, :],
                                                     data_trn['forc_norm'][i-1, :, :idx_start, :]), axis=1)
                    time_spin = np.concatenate((data_trn['time'][idx_start - spin_up_len:], data_trn['time'][:idx_start]))
            else:
                forc_spin = data_trn['forc'][:, idx_start - spin_up_len:idx_start, :]
                forc_norm_spin = data_trn['forc_norm'][:, idx_start - spin_up_len:idx_start, :]
                time_spin = data_trn['time'][idx_start - spin_up_len:idx_start]
            # concatenate the spin-up forcing data with the current forcing data
            forc_trn[i] = np.concatenate((forc_spin, data_trn['forc'][:, idx_start:idx_start+seq_len, :]), axis=1)
            forc_norm_trn[i] = np.concatenate((forc_norm_spin, data_trn['forc_norm'][:, idx_start:idx_start+seq_len, :]), axis=1)
            time_trn[i] = np.concatenate((time_spin, data_trn['time'][idx_start:idx_start+seq_len]))
            # update the start index
            idx_start += win_sz

        # validation period
        forc_spin_val = data_trn['forc'][:, -spin_up_len:, :]
        forc_norm_spin_val = data_trn['forc_norm'][:, -spin_up_len:, :]
        time_spin_val = data_trn['time'][-spin_up_len:]
        # test period
        forc_spin_tst = data_val['forc'][:, -spin_up_len:, :]
        forc_norm_spin_tst = data_val['forc_norm'][:, -spin_up_len:, :]
        time_spin_tst = data_val['time'][-spin_up_len:]
        # all periods
        forc_spin_all = data_trn['forc'][:, :spin_up_len, :]
        forc_norm_spin_all = data_trn['forc_norm'][:, :spin_up_len, :]
        time_spin_all = data_trn['time'][:spin_up_len]

        # update the dataset
        data_all = dict()
        data_all['forc'] = np.expand_dims(np.concatenate((forc_spin_all, data_trn['forc'], data_val['forc'],
                                                          data_tst['forc']), axis=1), axis=0)
        data_all['forc_norm'] = np.expand_dims(np.concatenate((forc_norm_spin_all, data_trn['forc_norm'],
                                                               data_val['forc_norm'], data_tst['forc_norm']), axis=1),
                                               axis=0)
        data_all['time'] = np.concatenate((time_spin_all, data_trn['time'], data_val['time'], data_tst['time']),
                                          axis=0)
        data_all['time'] = np.tile(data_all['time'], (data_all['forc_norm'].shape[0], 1))
        data_all['attrs_norm'] = np.tile(np.expand_dims(data_trn['attrs_norm'], 0),
                                         (data_all['forc_norm'].shape[0], 1, 1))

        data_trn['forc'] = forc_trn
        data_trn['forc_norm'] = forc_norm_trn
        data_trn['time'] = time_trn
        data_trn['attrs_norm'] = np.tile(np.expand_dims(data_trn['attrs_norm'], 0), (num_seq, 1, 1))

        data_val['forc'] = np.expand_dims(np.concatenate((forc_spin_val, data_val['forc']), axis=1), axis=0)
        data_val['forc_norm'] = np.expand_dims(np.concatenate((forc_norm_spin_val, data_val['forc_norm']), axis=1), axis=0)
        data_val['time'] = np.expand_dims(np.concatenate((time_spin_val, data_val['time']), axis=0), axis=0)
        data_val['time'] = np.tile(data_val['time'], (data_val['forc_norm'].shape[0], 1))
        data_val['attrs_norm'] = np.tile(np.expand_dims(data_val['attrs_norm'], 0), (data_val['forc_norm'].shape[0], 1, 1))

        data_tst['forc'] = np.expand_dims(np.concatenate((forc_spin_tst, data_tst['forc']), axis=1), axis=0)
        data_tst['forc_norm'] = np.expand_dims(np.concatenate((forc_norm_spin_tst, data_tst['forc_norm']), axis=1), axis=0)
        data_tst['time'] = np.expand_dims(np.concatenate((time_spin_tst, data_tst['time']), axis=0), axis=0)
        data_tst['time'] = np.tile(data_tst['time'], (data_tst['forc_norm'].shape[0], 1))
        data_tst['attrs_norm'] = np.tile(np.expand_dims(data_tst['attrs_norm'], 0), (data_tst['forc_norm'].shape[0], 1, 1))

        return data_all, data_trn, data_val, data_tst


    def cal_mean_std(self, forc_trn: np.ndarray, forc_var_lst: List[str], attr: np.ndarray, attr_var_lst: List[str]):
        stat_dict = {}
        # forcing data
        for k, var in enumerate(forc_var_lst):
            if var in ['pr', 'flow']:
                stat_dict[var] = self.cal_stat_gamma(forc_trn[:, :, k])
            else:
                stat_dict[var] = self.cal_stat(forc_trn[:, :, k])
        # attributes
        for k, var in enumerate(attr_var_lst):
            stat_dict[var] = self.cal_stat(attr[:, k])

        return stat_dict

    @staticmethod
    def cal_stat_gamma(x):  # for daily streamflow and precipitation
        a = x.flatten()
        b = a[~np.isnan(a)]  # kick out Nan
        b = np.log10(np.sqrt(b) + 0.1)  # do some transformation to change gamma characteristics
        mean = np.mean(b).astype(np.float32)
        std = np.std(b).astype(np.float32)
        if std < 0.001:
            std = 1
        return [mean, std]

    @staticmethod
    def cal_stat(x):
        a = x.flatten()
        b = a[~np.isnan(a)]  # kick out Nan
        mean = np.mean(b).astype(np.float32)
        std = np.std(b).astype(np.float32)
        if std < 0.001:
            std = 1
        return [mean, std]

    def trans_norm(self, x: np.ndarray, var_lst: Union[str, List[str]], to_norm: bool = True):
        """
        :param x: forcing data, basin attributes or river attributes.
        :param var_lst: list or string of variable names.
        :param to_norm: whether to do normalization or reverse normalization.
        """
        if type(var_lst) is str:
            var_lst = [var_lst]
        out = np.zeros(x.shape)
        for k in range(len(var_lst)):
            var = var_lst[k]
            stat = self.stat_dict[var]
            if to_norm is True:  # do normalization
                if len(x.shape) == 3:
                    if var in ['pr', 'flow']:
                        temp = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                        out[:, :, k] = (temp - stat[0]) / stat[1]
                    else:
                        out[:, :, k] = (x[:, :, k] - stat[0]) / stat[1]
                elif len(x.shape) == 2:
                    if var in ['pr', 'flow']:
                        temp = np.log10(np.sqrt(x[:, k]) + 0.1)
                        out[:, k] = (temp - stat[0]) / stat[1]
                    else:
                        out[:, k] = (x[:, k] - stat[0]) / stat[1]
            else:  # reverse normalization
                if len(x.shape) == 3:
                    out[:, :, k] = x[:, :, k] * stat[1] + stat[0]
                    if var in ['pr', 'flow']:
                        trans_tmp = np.power(10, out[:, :, k]) - 0.1
                        trans_tmp[trans_tmp < 0] = 0  # set negative as zero
                        out[:, :, k] = trans_tmp ** 2
                elif len(x.shape) == 2:
                    out[:, k] = x[:, k] * stat[1] + stat[0]
                    if var in ['pr', 'flow']:
                        trans_tmp = np.power(10, out[:, k]) - 0.1
                        trans_tmp[trans_tmp < 0] = 0
                        out[:, k] = trans_tmp ** 2
        return out

    def link_band_basin(self):
        basin_codes = sorted(list(set([band_name.split('_')[0] for band_name in self.band_codes])))
        id_dict = {}
        for i, bsn_code in enumerate(basin_codes):
            id_dict[i] = [j for j, name in enumerate(self.band_codes) if name.startswith(bsn_code)]
        return id_dict, basin_codes

    def cal_init_snow_depth(self, s_dep_path: str, device: Union[str, torch.device]):
        s_depth_df = pd.read_csv(s_dep_path, index_col=0, parse_dates=True, sep=r'\s+')
        s_depth_df = s_depth_df.reindex(columns=self.band_codes)
        s_dep_t0 = torch.tensor(s_depth_df.mean(axis=0).values, dtype=torch.float32, device=device)
        return s_dep_t0

    def cal_init_glacier_area(self, acr_path: str, spin_up_len: int, device: Union[str, torch.device]):
        band_obs_date = pd.to_datetime(self.band_attrs['date'].values)  # observation date
        # calculate the delta year between the observation date and the start of the time range
        delta_year = np.array([date.year for date in band_obs_date]) - self.t_range[0].year
        band_area_rgi = self.band_attrs['area'].values

        # load the glacier area change rate
        with open(acr_path, 'r') as f:
            acr_dict = json.load(f)
        if self.t_range[0].year == 2000:
            acr = acr_dict['2000']
        elif self.t_range[0].year == 1955:
            acr = acr_dict['1955']
        else:
            raise ValueError('The initial year should be 2000 or 1955.')

        # calculate the initial glacier area
        band_area_t0 = []
        for i, band in enumerate(self.band_codes):
            for k, v in acr.items():
                if band.startswith(k):
                    if delta_year[i] > 0:  # RGI-6.0 observation time is later than the initial time
                        area = band_area_rgi[i] / ((1 + v * 0.01) ** (delta_year[i]))
                    else: # RGI-6.0 observation time is earlier than initial time
                        area = band_area_rgi[i] * (1 + v * 0.01) ** (delta_year[i])
                    band_area_t0.append(area)
                    break
        return torch.tensor(band_area_t0, dtype=torch.float32, device=device)


    def print(self, msg):
        if self.logger:
            logging.info(msg)
        else:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {msg}')


class MyDataset(dataset.Dataset):
    def __init__(self, data_dict: dict, device: Union[str, torch.device]):
        self.x = torch.tensor(data_dict['forc'], device=device, dtype=torch.float32)
        self.xn = torch.tensor(data_dict['forc_norm'], device=device, dtype=torch.float32)
        self.attrs_norm = torch.tensor(data_dict['attrs_norm'], device=device, dtype=torch.float32)
        self.time = data_dict['time']

    def __getitem__(self, index):
        forc = self.x[index]
        forc_norm = self.xn[index]
        attrs_norm = self.attrs_norm[index]
        time = self.time[index]
        return forc, forc_norm, attrs_norm, time

    def __len__(self):
        return len(self.x)

