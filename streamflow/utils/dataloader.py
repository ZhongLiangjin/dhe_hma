from functools import reduce
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
import gc


class RainfallRunoffLoader:
    def __init__(self, bsn_forc_dir: str, bsn_riv_attr_dir: str, glac_forc_dir: str, glac_attr_dir: str,
                 glac_sim_path: str, periods: dict, seq_len: int, spin_up_len: int, win_sz: int, padding: int,
                 seq_len_eval: int, device: Union[str, torch.device] = 'cpu', logger: bool = False, mode: str = 'train',
                 sim_bsn_head: Union[List[str], str] = 'all'):
        # define the time range for training, validation and testing
        self.periods = periods
        for k, v in periods.items():
            if isinstance(v, list) and all(isinstance(item, str) for item in v):  # one period
                self.periods[k] = [pd.date_range(start=v[0], end=v[1], freq='D')]
            elif isinstance(v, list) and all(isinstance(item, list) for item in v):  # multiple periods
                self.periods[k] = [pd.date_range(start=period[0], end=period[1], freq='D') for period in v]

        self.t_range = pd.date_range(start=self.periods['train'][0][0], end=self.periods['test'][-1][-1], freq='D')
        self.logger = logger
        # Get glacier loader
        self.glac_loader = GlacierLoader(glac_forc_dir=glac_forc_dir, glac_attr_dir=glac_attr_dir,
                                         glac_sim_path=glac_sim_path, periods=periods, seq_len=seq_len,
                                         spin_up_len=spin_up_len, win_sz=win_sz, device=device, logger=logger,
                                         sim_bsn_head=sim_bsn_head, seq_len_eval=seq_len_eval)

        # load forcing data and attributes for all basins
        bsn_forc = self.load_forc(forc_dir=bsn_forc_dir, sim_bsn_head=sim_bsn_head)
        self.bsn_codes = list(bsn_forc.keys())
           # get the basin index for the glacier basins
        self.glac_bsn_idx = [self.bsn_codes.index(bsn_code) for bsn_code in self.glac_loader.glac_bsn_codes]
        self.bsn_attrs, self.riv_attrs, self.riv_up_bsn_codes, self.riv_up_bsn_idx = self.load_attrs(attr_dir=bsn_riv_attr_dir)
        # correct the forcing data for glacier basins
        # bsn_forc = self.correct_forc(bsn_forc=bsn_forc, forc_vars=['pr', 'tas'])

        # calculate the order of river routing
        # exclude the sub-basins with no routing, i.e., some small basins within the edge of TP
        no_rout_head = ['0106', '0206', '0207', '0305', '0409', '0503', '0604', '0705', '0807', '0902', '1002']
        self.rout_order_code, self.rout_order_idx = self.cal_rout_order(no_rout_head=no_rout_head, region_head_len=4,
                                                                        padding=padding)

        # get the dataset
        data_trn, data_val, data_tst, data_all = self.split_dataset(band_forc=bsn_forc)
        self.bsn_total_area = data_trn['bsn_attrs'][:, data_trn['bsn_attr_vars'].index('area_skm_ssu')]

        # normalize the forcing data and attributes
        self.forc_vars = data_trn['forc_vars']
        self.bsn_attr_vars = data_trn['bsn_attr_vars']
        self.riv_attr_vars = data_trn['riv_attr_vars']
        # calculate mean and std for normalization
        self.stat_dict = self.cal_mean_std(forc_trn=data_trn['forc'], forc_var_lst=data_trn['forc_vars'],
                                           bsn_attrs=data_trn['bsn_attrs'], bsn_attr_vars_lst=data_trn['bsn_attr_vars'],
                                           riv_attrs=data_trn['riv_attrs'], riv_attr_vars_lst=data_trn['riv_attr_vars'])
        for ds in [data_trn, data_val, data_tst, data_all]:
            ds['forc_norm'] = self.trans_norm(x=ds['forc'], var_lst=ds['forc_vars'], to_norm=True, data_type='forc')
            ds['bsn_attrs_norm'] = self.trans_norm(x=ds['bsn_attrs'], var_lst=ds['bsn_attr_vars'], to_norm=True,
                                                   data_type='bsn_attrs')
            ds['riv_attrs_norm'] = self.trans_norm(x=ds['riv_attrs'], var_lst=ds['riv_attr_vars'], to_norm=True,
                                                   data_type='riv_attrs')

        # generate sequences
        data_trn, data_val, data_tst, data_all = self.gen_seq(data_trn=data_trn, data_val=data_val, data_tst=data_tst,
                                                              data_all=data_all, seq_len=seq_len, win_sz=win_sz,
                                                              spin_up_len=spin_up_len, seq_len_eval=seq_len_eval)

        # get dataset and dataloader
        ds_trn = MyDataset(bsn_data_dict=data_trn, glac_data_dict=self.glac_loader.data_trn, device=device, mode='train')
        ds_val = MyDataset(bsn_data_dict=data_val, glac_data_dict=self.glac_loader.data_val, device=device, mode='valid')
        # ds_tst = MyDataset(bsn_data_dict=data_tst, glac_data_dict=self.glac_loader.data_tst, device=device, mode='test')
        ds_all = MyDataset(bsn_data_dict=data_all, glac_data_dict=self.glac_loader.data_all, device=device, mode='all')
        if mode == 'train':
            self.loader_trn = dataloader.DataLoader(dataset=ds_trn, batch_size=1, shuffle=True)
            self.loader_val = dataloader.DataLoader(dataset=ds_val, batch_size=1, shuffle=False)
            # self.loader_tst = dataloader.DataLoader(dataset=ds_tst, batch_size=1, shuffle=False)
        else:
            self.loader_all = dataloader.DataLoader(dataset=ds_all, batch_size=1, shuffle=False)

        # delete the data to save memory
        del self.glac_loader.data_all, self.glac_loader.data_val, self.glac_loader.data_trn, self.glac_loader.data_tst
        gc.collect()
        self.print(f'RainfallRunoffLoader initialized successfully.')

    def load_forc(self, forc_dir: str, sim_bsn_head: Union[List[str], str] = 'all'):
        self.print(f'Loading forcing data for all basins...')
        if os.path.exists(os.path.join(forc_dir, 'forc_basins.pkl')):
            forc = pickle.load(open(os.path.join(forc_dir, 'forc_basins.pkl'), 'rb'))
            if sim_bsn_head != 'all':
                # filter the forcing data by the basin name
                forc = {k: v for k, v in forc.items() if any(k.startswith(prefix) for prefix in sim_bsn_head)}
            for k, v in forc.items():
                    forc[k] = v[v.index.isin(self.t_range)]
        else:
            # initialize lists to store the elevation band name and corresponding forcing data
            forc = dict()
            files = os.listdir(forc_dir)
            for i, file in enumerate(files):
                if i % 100 == 0:
                    self.print(f'Processing {i}/{len(files)} files')
                if file.endswith('.txt'):
                    df = pd.read_csv(os.path.join(forc_dir, file), parse_dates=True,
                                     index_col=0, header=0, sep='\s+')
                    forc[file.split('.')[0]] = df
            # reorder the forcing data by the basin code
            forc = OrderedDict(sorted(forc.items()))
            # save the forcing data
            with open(os.path.join(forc_dir, 'forc_basins.pkl'), 'wb') as f:
                f.write(pickle.dumps(forc))
            # filter the forcing data by the basin name
            if sim_bsn_head != 'all':
                forc = {k: v for k, v in forc.items() if any(k.startswith(prefix) for prefix in sim_bsn_head)}
            # filter the forcing data by the time range
            for k, v in forc.items():
                forc[k] = v[v.index.isin(self.t_range)]
        return forc

    def correct_forc(self, bsn_forc, forc_vars):
        for bsn_code, bsn_forc_val in bsn_forc.items():
            if bsn_code in self.glac_loader.glac_bsn_codes:
                # get the glacier band ids and names in the basin
                glac_band_ids = self.glac_loader.glac_bsn_band_ids_dict[self.glac_loader.glac_bsn_codes.index(bsn_code)]
                glac_band_names = [self.glac_loader.glac_band_codes[glac_band_id] for glac_band_id in glac_band_ids]
                # get the glacier forcing data and area for all glacier bands in the basin
                bsn_area = self.bsn_attrs.loc[bsn_code, 'area_skm_ssu']
                glac_band_forc_tmp, glac_band_area_ratio_tmp = [], []
                for i, glac_band_name in enumerate(glac_band_names):
                    glac_forc_val = self.glac_loader.glac_band_forc[glac_band_name]
                    glac_band_forc_tmp.append(glac_forc_val)
                    glac_band_area_ratio_tmp.append(self.glac_loader.glac_band_area[glac_band_ids[i]] / bsn_area)
                # correct the precipitation and temperature
                if 'pr' in forc_vars:
                    glac_pr_correction = np.array([glac_band_forc_tmp[i]['pr'] * glac_band_area_ratio_tmp[i]
                                                   for i in range(len(glac_band_ids))]).sum(axis=0)
                    total_glac_area_ratio = np.array(glac_band_area_ratio_tmp).sum(axis=0)
                    bsn_forc_val['pr'] = (bsn_forc_val['pr'] - glac_pr_correction) / (1 - total_glac_area_ratio)
                    bsn_forc_val.loc[bsn_forc_val['pr'] < 0, 'pr'] = 0 # avoid negative precipitation
                if 'tas' in forc_vars:
                    glac_tas_correction = np.array([glac_band_forc_tmp[i]['tas'] * glac_band_area_ratio_tmp[i]
                                                    for i in range(len(glac_band_ids))]).sum(axis=0)
                    total_glac_area_ratio = np.array(glac_band_area_ratio_tmp).sum(axis=0)
                    bsn_forc_val['tas'] = (bsn_forc_val['tas'] - glac_tas_correction) / (1 - total_glac_area_ratio)

        return bsn_forc

    def load_attrs(self, attr_dir: str):
        self.print(f'Loading basin and river attributes...')
        # load attributes and reorder them
        basin_attrs = pd.read_csv(f'{attr_dir}/basin_attrs.txt', dtype={'Code': str}, index_col=0, sep=r'\s+')
        basin_attrs = basin_attrs.loc[self.bsn_codes]
        river_attrs = pd.read_csv(f'{attr_dir}/river_attrs.txt', dtype={'Up_basin': str}, index_col=0, sep=r'\s+')
        riv_up_bsn_codes = sorted(river_attrs.index.to_list())
        riv_up_bsn_codes = [code for code in riv_up_bsn_codes if code in self.bsn_codes]
        river_attrs = river_attrs.loc[riv_up_bsn_codes]
        riv_up_bsn_idx = [self.bsn_codes.index(code) for code in riv_up_bsn_codes if code in self.bsn_codes]
        return basin_attrs, river_attrs, riv_up_bsn_codes, riv_up_bsn_idx

    def split_dataset(self, band_forc):
        self.print(f'Splitting data into different sets...')
        train = reduce(pd.Index.union, self.periods['train'])
        valid = reduce(pd.Index.union, self.periods['valid'])
        test = reduce(pd.Index.union, self.periods['test'])
        # split forcing data
        forc_trn, forc_val, forc_tst, forc_all = [], [], [], []
        # [seq_len, n_var] -> [n_basin, seq_len, n_var]
        for k, v in band_forc.items():  # loop for each basin
            v['doy'] = v.index.dayofyear
            forc_trn.append(np.expand_dims(v.loc[train].values, axis=0))
            forc_val.append(np.expand_dims(v.loc[valid].values, axis=0))
            forc_tst.append(np.expand_dims(v.loc[test].values, axis=0))
            forc_all.append(np.expand_dims(v.loc[self.t_range].values, axis=0))
        forc_trn = np.concatenate(forc_trn, axis=0)
        forc_val = np.concatenate(forc_val, axis=0)
        forc_tst = np.concatenate(forc_tst, axis=0)
        forc_all = np.concatenate(forc_all, axis=0)
        # split time
        time_trn = pd.to_numeric(train)
        time_val = pd.to_numeric(valid)
        time_tst = pd.to_numeric(test)
        time_all = pd.to_numeric(self.t_range)

        # get forcing variables
        forc_vars = list(band_forc.values())[0].columns.tolist()
        # get basin and river attributes
        bsn_attrs = self.bsn_attrs.values
        bsn_attr_vars = self.bsn_attrs.columns.tolist()
        riv_attrs = self.riv_attrs.values
        riv_attr_vars = self.riv_attrs.columns.tolist()

        # create data dictionary
        data_trn = {'forc': forc_trn, 'bsn_attrs': bsn_attrs, 'riv_attrs': riv_attrs, 'time': time_trn,
                    'forc_vars': forc_vars, 'bsn_attr_vars': bsn_attr_vars, 'riv_attr_vars': riv_attr_vars}
        data_val = {'forc': forc_val, 'bsn_attrs': bsn_attrs, 'riv_attrs': riv_attrs, 'time': time_val,
                    'forc_vars': forc_vars, 'bsn_attr_vars': bsn_attr_vars, 'riv_attr_vars': riv_attr_vars}
        data_tst = {'forc': forc_tst, 'bsn_attrs': bsn_attrs, 'riv_attrs': riv_attrs, 'time': time_tst,
                    'forc_vars': forc_vars, 'bsn_attr_vars': bsn_attr_vars, 'riv_attr_vars': riv_attr_vars}
        data_all = {'forc': forc_all, 'bsn_attrs': bsn_attrs, 'riv_attrs': riv_attrs, 'time': time_all,
                    'forc_vars': forc_vars, 'bsn_attr_vars': bsn_attr_vars, 'riv_attr_vars': riv_attr_vars}

        return data_trn, data_val, data_tst, data_all

    def gen_seq(self, data_trn, data_val, data_tst, data_all, seq_len: int, win_sz: int, spin_up_len: int, seq_len_eval: int):
        # initialize the number of sequences for different periods
        num_seq = np.array([int((len(v) - seq_len) / win_sz) + 1 for v in self.periods['train']])
        forc_trn = np.zeros(
            (num_seq.sum(), data_trn['forc'].shape[0], seq_len + spin_up_len, data_trn['forc'].shape[2]))
        forc_norm_trn = np.zeros(
            (num_seq.sum(), data_trn['forc'].shape[0], seq_len + spin_up_len, data_trn['forc'].shape[2]))
        time_trn = np.zeros((num_seq.sum(), seq_len + spin_up_len))
        forc_val, forc_norm_val, time_val = [], [], []  # each period has only one sequence with the length of the period
        forc_tst, forc_norm_tst, time_tst = [], [], []  # each period has only one sequence with the length of the period
        forc_all, forc_norm_all, time_all = [], [], []

        # get the sequences for the training period
        for j, period in enumerate(self.periods['train']):
            idx_start = 0 if j == 0 else len(self.periods['train'][j - 1])
            data_trn_tmp = dict()
            data_trn_tmp['forc'] = data_trn['forc'][:, idx_start: idx_start + len(period), :]
            data_trn_tmp['forc_norm'] = data_trn['forc_norm'][:, idx_start: idx_start + len(period), :]
            data_trn_tmp['time'] = data_trn['time'][idx_start: idx_start + len(period)]
            # loop for each period to get the sequences
            idx_start = 0
            for i in range(num_seq[j]):
                # get the spin-up forcing data
                if idx_start < spin_up_len:
                    forc_spin = data_trn_tmp['forc'][:, :spin_up_len, :]
                    forc_norm_spin = data_trn_tmp['forc_norm'][:, :spin_up_len, :]
                    time_spin = data_trn_tmp['time'][:spin_up_len]
                else:
                    forc_spin = data_trn_tmp['forc'][:, idx_start - spin_up_len:idx_start, :]
                    forc_norm_spin = data_trn_tmp['forc_norm'][:, idx_start - spin_up_len:idx_start, :]
                    time_spin = data_trn_tmp['time'][idx_start - spin_up_len:idx_start]
                # concatenate the spin-up forcing data with the current forcing data
                idx = i if j == 0 else i + num_seq[j - 1]
                forc_trn[idx] = np.concatenate((forc_spin, data_trn_tmp['forc'][:, idx_start:idx_start + seq_len, :]),
                                               axis=1)
                forc_norm_trn[idx] = np.concatenate(
                    (forc_norm_spin, data_trn_tmp['forc_norm'][:, idx_start:idx_start + seq_len, :]), axis=1)
                time_trn[idx] = np.concatenate((time_spin, data_trn_tmp['time'][idx_start:idx_start + seq_len]))
                # update the start index
                idx_start += win_sz

            # validation period
            # get the spin-up data first
            idx_start = 0 if j == 0 else len(self.periods['valid'][j - 1])
            data_val_tmp = dict()
            data_val_tmp['forc'] = data_val['forc'][:, idx_start: idx_start + len(period), :]
            data_val_tmp['forc_norm'] = data_val['forc_norm'][:, idx_start: idx_start + len(period), :]
            data_val_tmp['time'] = data_val['time'][idx_start: idx_start + len(period)]
            forc_spin_val = data_trn_tmp['forc'][:, -spin_up_len:, :]
            forc_norm_spin_val = data_trn_tmp['forc_norm'][:, -spin_up_len:, :]
            time_spin_val = data_trn_tmp['time'][-spin_up_len:]
            # concatenate the spin-up forcing data with the current forcing data
            data_val_tmp['forc'] = np.concatenate((forc_spin_val, data_val_tmp['forc']), axis=1)
            data_val_tmp['forc_norm'] = np.concatenate((forc_norm_spin_val, data_val_tmp['forc_norm']), axis=1)
            data_val_tmp['time'] = np.concatenate((time_spin_val, data_val_tmp['time']), axis=0)
            forc_val.append(data_val_tmp['forc'])
            forc_norm_val.append(data_val_tmp['forc_norm'])
            time_val.append(data_val_tmp['time'])

            # test period
            idx_start = 0 if j == 0 else len(self.periods['test'][j - 1])
            data_tst_tmp = dict()
            data_tst_tmp['forc'] = data_tst['forc'][:, idx_start: idx_start + len(period), :]
            data_tst_tmp['forc_norm'] = data_tst['forc_norm'][:, idx_start: idx_start + len(period), :]
            data_tst_tmp['time'] = data_tst['time'][idx_start: idx_start + len(period)]
            forc_spin_tst = data_val['forc'][:, -spin_up_len:, :]
            forc_norm_spin_tst = data_val['forc_norm'][:, -spin_up_len:, :]
            time_spin_tst = data_val['time'][-spin_up_len:]
            # concatenate the spin-up forcing data with the current forcing data
            data_tst_tmp['forc'] = np.concatenate((forc_spin_tst, data_tst_tmp['forc']), axis=1)
            data_tst_tmp['forc_norm'] = np.concatenate((forc_norm_spin_tst, data_tst_tmp['forc_norm']), axis=1)
            data_tst_tmp['time'] = np.concatenate((time_spin_tst, data_tst_tmp['time']), axis=0)
            forc_tst.append(data_tst_tmp['forc'])
            forc_norm_tst.append(data_tst_tmp['forc_norm'])
            time_tst.append(data_tst_tmp['time'])

        data_trn['forc'] = forc_trn
        data_trn['forc_norm'] = forc_norm_trn
        data_trn['time'] = time_trn
        data_trn['bsn_attrs_norm'] = np.tile(np.expand_dims(data_trn['bsn_attrs_norm'], 0), (num_seq.sum(), 1, 1))
        data_trn['riv_attrs_norm'] = np.tile(np.expand_dims(data_trn['riv_attrs_norm'], 0), (num_seq.sum(), 1, 1))

        data_val['forc'] = forc_val
        data_val['forc_norm'] = forc_norm_val
        data_val['time'] = time_val

        data_tst['forc'] = forc_tst
        data_tst['forc_norm'] = forc_norm_tst
        data_tst['time'] = time_tst

        # get the dataset for all data, set the sequence length as 10 years
        data_all_time = pd.to_datetime(data_all['time'])
        num_seq = np.ceil((data_all_time[-1].year - data_all_time[0].year + 1) / seq_len_eval).astype(int)
        idx_start = 0
        for i in range(num_seq):
            time_end = datetime(data_all_time[idx_start].year+seq_len_eval,
                               data_all_time[idx_start].month, data_all_time[idx_start].day)
            idx_end = min((time_end - data_all_time[0]).days, len(data_all_time))
            if idx_start == 0:
                data_all_tmp = dict()
                data_all_tmp['forc'] = np.concatenate((data_all['forc'][:, :spin_up_len],
                                                       data_all['forc'][:, idx_start: idx_end]), axis=1)
                data_all_tmp['forc_norm'] = np.concatenate((data_all['forc_norm'][:, :spin_up_len],
                                                            data_all['forc_norm'][:, idx_start: idx_end]), axis=1)
                data_all_tmp['time'] = np.concatenate((data_all['time'][:spin_up_len],
                                                       data_all['time'][idx_start: idx_end]), axis=0)
                forc_all.append(data_all_tmp['forc'])
                forc_norm_all.append(data_all_tmp['forc_norm'])
                time_all.append(data_all_tmp['time'])
            else:
                forc_all.append(data_all['forc'][:, idx_start: idx_end])
                forc_norm_all.append(data_all['forc_norm'][:, idx_start: idx_end])
                time_all.append(np.array(data_all['time'][idx_start: idx_end]))
            idx_start = idx_end

        data_all['forc'] = forc_all
        data_all['forc_norm'] = forc_norm_all
        data_all['time'] = time_all

        return data_trn, data_val, data_tst, data_all

    def cal_rout_order(self, no_rout_head, region_head_len, padding=-9999):
        # get the order of basins in the dataset
        bsns_rout = [bsn for bsn in self.bsn_codes if bsn[:4] not in no_rout_head] # filter the basins without routing
        # initiate the routing order
        def group_by_tributary(bsn_codes, iter_level=3, tot_level=3):
            """
            :param bsn_codes: basin codes for the current river level.
            :param iter_level: current iteration level.
            :param tot_level: total river levels.
            """
            # initiate a group dictionary, each group represents a tributary
            groups = {}
            for code in bsn_codes:
                # get codes with the same prefix and the last two digits are not 00
                prefix = code[:2 * (iter_level - 1)]
                suffix = code[2 * (iter_level - 1):]
                if suffix != '00':
                    if prefix not in groups:
                        groups[prefix] = []
                    groups[prefix].append(code + '00' * (tot_level - iter_level))
            # sort the code based on the prefix
            for k, v in groups.items():
                groups[k] = sorted(v, reverse=True)

            # transform the group dictionary to a list
            lst = [v for k, v in groups.items()]
            return lst

        def check_rout_order(bsn_codes, rout_order):
            # check if all basins are in the routing order
            tmp = []
            for k, v in rout_order.items():
                tmp.extend([item for sublist in v for item in sublist])
            assert set(bsn_codes) == set(tmp), 'Some basins are missing in the routing order.'

        # get the routing order using iterative grouping
        iter_level, tot_level = len(bsns_rout[0]) // 2, len(bsns_rout[0]) // 2  # tributary levels
        rout_order_code = {}
        while iter_level > region_head_len // 2:
            rout_level = iter_level-region_head_len // 2
            prefixes = [code[:2 * iter_level] for code in bsns_rout]
            suffixes = [code[2 * iter_level:] for code in bsns_rout]
            bsn_code_tmp = [prefixes[i] for i, suf in enumerate(suffixes) if suf == '00' * (tot_level - iter_level)]
            rout_order_code[rout_level] = group_by_tributary(bsn_codes=bsn_code_tmp, iter_level=iter_level,
                                                             tot_level=tot_level)
            iter_level -= 1
        # check if all basins are in the routing order
        check_rout_order(bsns_rout, rout_order_code)
        rout_order_code = {k: v for k, v in rout_order_code.items() if len(v) > 0}

        # padding the tributaries with the most downstream basin
        for rout_level, tributaries in rout_order_code.items():
            max_len_tributary = max(np.array([len(tributary) for tributary in tributaries])) + 1 + 1  # add the downstream-most basin and padding
            iter_level = rout_level + region_head_len // 2
            for i, tributary in enumerate(tributaries):
                prefix = tributary[-1][:2 * iter_level]
                down_bsn = f'{int(prefix)-1}'.zfill(len(prefix)) + '00' * (tot_level - iter_level)
                if down_bsn in bsns_rout:
                    rout_order_code[rout_level][i].append(down_bsn)
                rout_order_code[rout_level][i] = rout_order_code[rout_level][i] + [padding] * (max_len_tributary - len(tributary))
            rout_order_code[rout_level] = np.array(rout_order_code[rout_level])

        # get the bsn_idx in the routing order
        rout_order_idx = {}
        for rout_level, tributaries in rout_order_code.items():
            rout_order_idx[rout_level] = np.full(tributaries.shape, np.nan)
            for row in range(tributaries.shape[0]):
                for col in range(tributaries.shape[1]):
                    if tributaries[row, col] in bsns_rout:
                        idx = self.bsn_codes.index(tributaries[row, col])
                        rout_order_idx[rout_level][row, col] = idx
                    else:
                        rout_order_idx[rout_level][row, col] = padding

        # Determine the maximum dimensions
        max_dim1 = max(tributaries.shape[0] for tributaries in rout_order_idx.values())
        max_dim2 = max(tributaries.shape[1] for tributaries in rout_order_idx.values())
        num_levels = len(rout_order_idx)

        # convert the dictionary to a 3D array, the first dimension is the river level
        merged_rout_order_idx = np.full((num_levels, max_dim1, max_dim2), padding)
        for i, tributaries in enumerate(rout_order_idx.values()):
            merged_rout_order_idx[i, :tributaries.shape[0], :tributaries.shape[1]] = tributaries

        return rout_order_code, merged_rout_order_idx

    def cal_init_snow_depth(self, s_dep_path: str, device: Union[str, torch.device]):
        s_depth_df = pd.read_csv(s_dep_path, index_col=0, parse_dates=True, sep=r'\s+')
        s_depth_df.columns = [str(x).zfill(12) for x in s_depth_df.columns]
        s_depth_df = s_depth_df.reindex(columns=self.bsn_codes)
        s_dep_t0 = torch.tensor(s_depth_df.mean(axis=0).values, dtype=torch.float32, device=device)
        return s_dep_t0.unsqueeze(1)


    def cal_init_LAI(self, LAI_path: str, init_LAI_min: bool, init_LAI_max: bool, device: Union[str, torch.device]):
        lai_df = pd.read_csv(LAI_path, index_col=0, dtype={'basin_id': str}, sep=r'\s+')
        lai_df = lai_df.reindex(index=self.bsn_codes)
        lai_t0 = torch.tensor(lai_df['Jan_1st'].values, dtype=torch.float32, device=device).unsqueeze(1)
        lai_t0 = torch.nan_to_num(lai_t0, nan=0.0)
        lai_min = torch.tensor(lai_df['min'].values, dtype=torch.float32, device=device).unsqueeze(1) if init_LAI_min else None
        lai_min = torch.nan_to_num(lai_min, nan=0.0) if lai_min is not None else None
        lai_max = torch.tensor(lai_df['max'].values, dtype=torch.float32, device=device).unsqueeze(1) if init_LAI_max else None
        lai_max = torch.nan_to_num(lai_max, nan=0.1) if lai_max is not None else None

        return lai_t0, lai_min, lai_max

    def cal_mean_std(self, forc_trn: np.ndarray, forc_var_lst: List[str], bsn_attrs: np.ndarray,
                     bsn_attr_vars_lst: List[str], riv_attrs: np.ndarray, riv_attr_vars_lst: List[str]):
        stat_dict = {}
        # forcing data
        for k, var in enumerate(forc_var_lst):
            if var in ['pr', 'flow']:
                stat_dict[var] = self.cal_stat_gamma(forc_trn[:, :, k])
            else:
                stat_dict[var] = self.cal_stat(forc_trn[:, :, k])
        # basin attributes
        for k, var in enumerate(bsn_attr_vars_lst):
            stat_dict[f'b_{var}'] = self.cal_stat(bsn_attrs[:, k])
        # river attributes
        for k, var in enumerate(riv_attr_vars_lst):
            stat_dict[f'r_{var}'] = self.cal_stat(riv_attrs[:, k])

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

    def trans_norm(self, x: np.ndarray, var_lst: Union[str, List[str]], to_norm: bool = True, data_type: str = 'forc'):
        """
        :param x: forcing data, basin attributes or river attributes.
        :param var_lst: list or string of variable names.
        :param to_norm: whether to do normalization or reverse normalization.
        """
        assert data_type in ['forc', 'bsn_attrs', 'riv_attrs']
        if type(var_lst) is str:
            var_lst = [var_lst]
        out = np.zeros(x.shape)
        for k in range(len(var_lst)):
            var = var_lst[k]
            if data_type == 'forc':
                stat = self.stat_dict[var]
            elif data_type == 'bsn_attrs':
                stat = self.stat_dict[f'b_{var}']
            else:
                stat = self.stat_dict[f'r_{var}']
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

    def print(self, msg):
        if self.logger:
            logging.info(msg)
        else:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {msg}')


class GlacierLoader:
    def __init__(self, glac_forc_dir: str, glac_attr_dir: str, glac_sim_path: str, periods: dict, seq_len: int,
                 spin_up_len: int, win_sz: int, seq_len_eval: int, device: Union[str, torch.device] = 'cpu',
                 logger: bool = False, sim_bsn_head: Union[List[str], str] = 'all'):
        # define the time range for training, validation and testing
        self.periods = periods
        for k, v in periods.items():
            if isinstance(v, list) and all(isinstance(item, str) for item in v):  # one period
                self.periods[k] = [pd.date_range(start=v[0], end=v[1], freq='D')]
            elif isinstance(v, list) and all(isinstance(item, list) for item in v):  # multiple periods
                self.periods[k] = [pd.date_range(start=period[0], end=period[1], freq='D') for period in v]

        self.t_range = pd.date_range(start=self.periods['train'][0][0], end=self.periods['test'][-1][-1], freq='D')
        self.logger = logger

        # load forcing data and glacier attributes
        self.glac_band_forc, glac_band_codes_all = self.load_forc(forc_dir=glac_forc_dir, sim_bsn_head=sim_bsn_head)
        self.glac_band_codes = list(self.glac_band_forc.keys())
        self.glac_band_attrs = self.load_attrs(attr_dir=glac_attr_dir)
        self.glac_band_attrs.drop(columns=['date'], inplace=True)

        # link the elevation band id to the basin id
        self.glac_bsn_band_ids_dict, self.glac_bsn_codes = self.link_glac_band_basin()

        # load initial snow depth and glacier area
        self.glac_band_area, self.bsn_glac_area = self.load_init_glacier_area(glac_sim_path=glac_sim_path,
                                                                              glac_band_codes_all=glac_band_codes_all)

        # get the dataset
        self.glac_band_slope = self.glac_band_attrs['slope'].values
        data_trn, data_val, data_tst, data_all = self.split_dataset(band_forc=self.glac_band_forc)

        # normalize the forcing data and attributes
        self.forc_vars = data_trn['forc_vars']
        self.attr_vars = data_trn['attr_vars']
        # calculate mean and std for normalization
        self.stat_dict = self.cal_mean_std(forc_trn=data_trn['forc'], forc_var_lst=data_trn['forc_vars'],
                                           attr=data_trn['attrs'], attr_var_lst=data_trn['attr_vars'])
        for ds in [data_trn, data_val, data_tst, data_all]:
            ds['forc_norm'] = self.trans_norm(x=ds['forc'], var_lst=ds['forc_vars'], to_norm=True)
            ds['attrs_norm'] = self.trans_norm(x=ds['attrs'], var_lst=ds['attr_vars'], to_norm=True)

        # generate sequences
        self.data_trn, self.data_val, self.data_tst, self.data_all = self.gen_seq(data_trn=data_trn, data_val=data_val,
                                                                                  data_tst=data_tst, data_all=data_all,
                                                                                  seq_len=seq_len, win_sz=win_sz,
                                                                                  spin_up_len=spin_up_len,
                                                                                  seq_len_eval=seq_len_eval)

        # get dataset and dataloader
        # ds_trn = MyDataset(data_dict=data_trn, device=device, mode='train')
        # ds_val = MyDataset(data_dict=data_val, device=device, mode='valid')
        # ds_tst = MyDataset(data_dict=data_tst, device=device, mode='test')
        # self.loader_trn = dataloader.DataLoader(dataset=ds_trn, batch_size=1, shuffle=True)
        # self.loader_val = dataloader.DataLoader(dataset=ds_val, batch_size=1, shuffle=False)
        # self.loader_tst = dataloader.DataLoader(dataset=ds_tst, batch_size=1, shuffle=False)

        self.print(f'GlacierLoader initialized successfully.')

    def load_forc(self, forc_dir: str, sim_bsn_head: Union[List[str], str] = 'all'):
        self.print(f'Loading forcing data for glacier basins...')
        if os.path.exists(os.path.join(forc_dir, 'forc_band.pkl')):
            forc = pickle.load(open(os.path.join(forc_dir, 'forc_band.pkl'), 'rb'))
            glac_band_codes_all = list(forc.keys())  # all glacier bands
            if sim_bsn_head != 'all':
                forc = {k: v for k, v in forc.items() if any(k.startswith(prefix) for prefix in sim_bsn_head)}
            for k, v in forc.items():
                forc[k] = v[v.index.isin(self.t_range)]
        else:
            # initialize lists to store the elevation band name and corresponding forcing data
            forc = dict()
            folders = os.listdir(forc_dir)  # debris-covered and debris-free glacier
            folders = [folder for folder in folders if os.path.isdir(os.path.join(forc_dir, folder))]
            for folder in folders:
                files = os.listdir(os.path.join(forc_dir, folder))
                for i, file in enumerate(files):
                    if i % 100 == 0:
                        self.print(f'Processing {i}/{len(files)} files in folder {folder}')
                    if file.endswith('.txt'):
                        df = pd.read_csv(os.path.join(forc_dir, folder, file), parse_dates=True,
                                         index_col=0, header=0, sep='\s+')
                        forc[file.split('.')[0]] = df
            # reorder the forcing data by the elevation band name,
            # make sure the band with lower elevation is in the front and the debris free glacier is in the front
            forc = OrderedDict(sorted(forc.items()))
            # save the forcing data
            with open(os.path.join(forc_dir, 'forc_band.pkl'), 'wb') as f:
                f.write(pickle.dumps(forc))
            glac_band_codes_all = list(forc.keys())  # all glacier bands
            if sim_bsn_head != 'all':
                forc = {k: v for k, v in forc.items() if any(k.startswith(prefix) for prefix in sim_bsn_head)}
            # filter the forcing data by the time range
            for k, v in forc.items():
                forc[k] = v[v.index.isin(self.t_range)]
        return forc, glac_band_codes_all

    def load_attrs(self, attr_dir: str):
        self.print(f'Loading glacier attributes...')
        debris_free_attrs = pd.read_csv(f'{attr_dir}/glac_clean_attrs.txt', index_col=0, sep=r'\s+')
        debris_covered_attrs = pd.read_csv(f'{attr_dir}/glac_debris_attrs.txt', index_col=0, sep=r'\s+')
        glac_attrs = pd.concat([debris_free_attrs, debris_covered_attrs])
        # reorder the attributes by the elevation band name
        glac_attrs = glac_attrs.loc[self.glac_band_codes]
        return glac_attrs

    def get_dataset(self, band_forc, spin_up_len):
        self.print(f'Getting datasets for glacier basins...')
        # get forcing data
        forc = []
        spin_forc = pd.date_range(start=self.t_range[0], end=self.t_range[0] + pd.Timedelta(days=spin_up_len - 1),
                                  freq='D')
        for k, v in band_forc.items():
            v['doy'] = v.index.dayofyear
            forc.append(np.expand_dims(pd.concat([v.loc[spin_forc], v.loc[self.t_range]]).values, axis=0))

        forc = np.concatenate(forc, axis=0)
        forc_vars = list(band_forc.values())[0].columns.tolist()
        # get time numerical value
        time = np.array(list(pd.to_numeric(spin_forc)) + list(pd.to_numeric(self.t_range)))
        # get glacier attributes
        attrs = self.glac_band_attrs.values
        atts_vars = self.glac_band_attrs.columns.tolist()

        # create data dictionary
        ds = {'forc': forc, 'attrs': attrs, 'time': time, 'forc_vars': forc_vars, 'attr_vars': atts_vars}

        return ds

    def split_dataset(self, band_forc):
        self.print(f'Splitting data into different sets for glacier basins...')
        train = reduce(pd.Index.union, self.periods['train'])
        valid = reduce(pd.Index.union, self.periods['valid'])
        test = reduce(pd.Index.union, self.periods['test'])
        # split forcing data
        forc_trn, forc_val, forc_tst, forc_all = [], [], [], []
        # [seq_len, n_var] -> [n_band, seq_len, n_var]
        for k, v in band_forc.items():  # loop for each band
            v['doy'] = v.index.dayofyear
            forc_trn.append(np.expand_dims(v.loc[train].values, axis=0))
            forc_val.append(np.expand_dims(v.loc[valid].values, axis=0))
            forc_tst.append(np.expand_dims(v.loc[test].values, axis=0))
            forc_all.append(np.expand_dims(v.loc[self.t_range].values, axis=0))
        forc_trn = np.concatenate(forc_trn, axis=0)
        forc_val = np.concatenate(forc_val, axis=0)
        forc_tst = np.concatenate(forc_tst, axis=0)
        forc_all = np.concatenate(forc_all, axis=0)
        # split time
        time_trn = pd.to_numeric(train)
        time_val = pd.to_numeric(valid)
        time_tst = pd.to_numeric(test)
        time_all = pd.to_numeric(self.t_range)

        # get forcing variables
        forc_vars = list(band_forc.values())[0].columns.tolist()
        # get glacier attributes
        attrs = self.glac_band_attrs.values
        attr_vars = self.glac_band_attrs.columns.tolist()

        # create data dictionary
        data_trn = {'forc': forc_trn, 'attrs': attrs, 'time': time_trn, 'forc_vars': forc_vars, 'attr_vars': attr_vars}
        data_val = {'forc': forc_val, 'attrs': attrs, 'time': time_val, 'forc_vars': forc_vars, 'attr_vars': attr_vars}
        data_tst = {'forc': forc_tst, 'attrs': attrs, 'time': time_tst, 'forc_vars': forc_vars, 'attr_vars': attr_vars}
        data_all = {'forc': forc_all, 'attrs': attrs, 'time': time_all, 'forc_vars': forc_vars, 'attr_vars': attr_vars}

        return data_trn, data_val, data_tst, data_all

    def gen_seq(self, data_trn, data_val, data_tst, data_all, seq_len: int, win_sz: int, spin_up_len: int, seq_len_eval: int):
        self.print(f'Generating sequences for glacier basins...')
        # initialize the number of sequences for different periods
        num_seq = np.array([int((len(v) - seq_len) / win_sz) + 1 for v in self.periods['train']])
        forc_trn = np.zeros(
            (num_seq.sum(), data_trn['forc'].shape[0], seq_len + spin_up_len, data_trn['forc'].shape[2]))
        forc_norm_trn = np.zeros(
            (num_seq.sum(), data_trn['forc'].shape[0], seq_len + spin_up_len, data_trn['forc'].shape[2]))
        time_trn = np.zeros((num_seq.sum(), seq_len + spin_up_len))
        forc_val, forc_norm_val, time_val = [], [], []  # each period has only one sequence with the length of the period
        forc_tst, forc_norm_tst, time_tst = [], [], []  # each period has only one sequence with the length of the period
        forc_all, forc_norm_all, time_all = [], [], []

        # get the sequences for the training period
        for j, period in enumerate(self.periods['train']):
            idx_start = 0 if j == 0 else len(self.periods['train'][j - 1])
            data_trn_tmp = dict()
            data_trn_tmp['forc'] = data_trn['forc'][:, idx_start: idx_start + len(period), :]
            data_trn_tmp['forc_norm'] = data_trn['forc_norm'][:, idx_start: idx_start + len(period), :]
            data_trn_tmp['time'] = data_trn['time'][idx_start: idx_start + len(period)]
            # loop for each period to get the sequences
            idx_start = 0
            for i in range(num_seq[j]):
                # get the spin-up forcing data
                if idx_start < spin_up_len:
                    forc_spin = data_trn_tmp['forc'][:, :spin_up_len, :]
                    forc_norm_spin = data_trn_tmp['forc_norm'][:, :spin_up_len, :]
                    time_spin = data_trn_tmp['time'][:spin_up_len]
                else:
                    forc_spin = data_trn_tmp['forc'][:, idx_start - spin_up_len:idx_start, :]
                    forc_norm_spin = data_trn_tmp['forc_norm'][:, idx_start - spin_up_len:idx_start, :]
                    time_spin = data_trn_tmp['time'][idx_start - spin_up_len:idx_start]
                # concatenate the spin-up forcing data with the current forcing data
                idx = i if j == 0 else i + num_seq[j - 1]
                forc_trn[idx] = np.concatenate((forc_spin, data_trn_tmp['forc'][:, idx_start:idx_start + seq_len, :]),
                                               axis=1)
                forc_norm_trn[idx] = np.concatenate(
                    (forc_norm_spin, data_trn_tmp['forc_norm'][:, idx_start:idx_start + seq_len, :]), axis=1)
                time_trn[idx] = np.concatenate((time_spin, data_trn_tmp['time'][idx_start:idx_start + seq_len]))
                # update the start index
                idx_start += win_sz

            # validation period
            # get the spin-up data first
            idx_start = 0 if j == 0 else len(self.periods['valid'][j - 1])
            data_val_tmp = dict()
            data_val_tmp['forc'] = data_val['forc'][:, idx_start: idx_start + len(period), :]
            data_val_tmp['forc_norm'] = data_val['forc_norm'][:, idx_start: idx_start + len(period), :]
            data_val_tmp['time'] = data_val['time'][idx_start: idx_start + len(period)]
            forc_spin_val = data_trn_tmp['forc'][:, -spin_up_len:, :]
            forc_norm_spin_val = data_trn_tmp['forc_norm'][:, -spin_up_len:, :]
            time_spin_val = data_trn_tmp['time'][-spin_up_len:]
            # concatenate the spin-up forcing data with the current forcing data
            data_val_tmp['forc'] = np.concatenate((forc_spin_val, data_val_tmp['forc']), axis=1)
            data_val_tmp['forc_norm'] = np.concatenate((forc_norm_spin_val, data_val_tmp['forc_norm']), axis=1)
            data_val_tmp['time'] = np.concatenate((time_spin_val, data_val_tmp['time']), axis=0)
            forc_val.append(data_val_tmp['forc'])
            forc_norm_val.append(data_val_tmp['forc_norm'])
            time_val.append(data_val_tmp['time'])

            # test period
            idx_start = 0 if j == 0 else len(self.periods['test'][j - 1])
            data_tst_tmp = dict()
            data_tst_tmp['forc'] = data_tst['forc'][:, idx_start: idx_start + len(period), :]
            data_tst_tmp['forc_norm'] = data_tst['forc_norm'][:, idx_start: idx_start + len(period), :]
            data_tst_tmp['time'] = data_tst['time'][idx_start: idx_start + len(period)]
            forc_spin_tst = data_val['forc'][:, -spin_up_len:, :]
            forc_norm_spin_tst = data_val['forc_norm'][:, -spin_up_len:, :]
            time_spin_tst = data_val['time'][-spin_up_len:]
            # concatenate the spin-up forcing data with the current forcing data
            data_tst_tmp['forc'] = np.concatenate((forc_spin_tst, data_tst_tmp['forc']), axis=1)
            data_tst_tmp['forc_norm'] = np.concatenate((forc_norm_spin_tst, data_tst_tmp['forc_norm']), axis=1)
            data_tst_tmp['time'] = np.concatenate((time_spin_tst, data_tst_tmp['time']), axis=0)
            forc_tst.append(data_tst_tmp['forc'])
            forc_norm_tst.append(data_tst_tmp['forc_norm'])
            time_tst.append(data_tst_tmp['time'])

        data_trn['forc'] = forc_trn
        data_trn['forc_norm'] = forc_norm_trn
        data_trn['time'] = time_trn
        data_trn['attrs_norm'] = np.tile(np.expand_dims(data_trn['attrs_norm'], 0), (num_seq.sum(), 1, 1))
        idx = self.t_range.get_indexer(pd.to_datetime(time_trn[:, 0])) # type: ignore
        data_trn['glac_area_t0'] = self.glac_band_area[:, idx].T

        data_val['forc'] = forc_val
        data_val['forc_norm'] = forc_norm_val
        data_val['time'] = time_val
        idx = self.t_range.get_indexer(pd.to_datetime([time_val[i][0] for i in range(len(time_val))])) # type: ignore
        data_val['glac_area_t0'] = self.glac_band_area[:, idx].T

        data_tst['forc'] = forc_tst
        data_tst['forc_norm'] = forc_norm_tst
        data_tst['time'] = time_tst
        idx = self.t_range.get_indexer(pd.to_datetime([time_tst[i][0] for i in range(len(time_tst))])) # type: ignore
        data_tst['glac_area_t0'] = self.glac_band_area[:, idx].T

        # get the dataset for all data
        data_all_time = pd.to_datetime(data_all['time'])
        num_seq = np.ceil((data_all_time[-1].year - data_all_time[0].year + 1) / seq_len_eval).astype(int)
        idx_start = 0
        for i in range(num_seq):
            time_end = datetime(data_all_time[idx_start].year+seq_len_eval,
                               data_all_time[idx_start].month, data_all_time[idx_start].day)
            idx_end = min((time_end - data_all_time[0]).days, len(data_all_time))
            if idx_start == 0:
                data_all_tmp = dict()
                data_all_tmp['forc'] = np.concatenate((data_all['forc'][:, :spin_up_len],
                                                       data_all['forc'][:, idx_start: idx_end]), axis=1)
                data_all_tmp['forc_norm'] = np.concatenate((data_all['forc_norm'][:, :spin_up_len],
                                                            data_all['forc_norm'][:, idx_start: idx_end]), axis=1)
                data_all_tmp['time'] = np.concatenate((data_all['time'][:spin_up_len],
                                                       data_all['time'][idx_start: idx_end]), axis=0)
                forc_all.append(data_all_tmp['forc'])
                forc_norm_all.append(data_all_tmp['forc_norm'])
                time_all.append(data_all_tmp['time'])
            else:
                forc_all.append(data_all['forc'][:, idx_start: idx_end])
                forc_norm_all.append(data_all['forc_norm'][:, idx_start: idx_end])
                time_all.append(np.array(data_all['time'][idx_start: idx_end]))
            idx_start = idx_end

        data_all['forc'] = forc_all
        data_all['forc_norm'] = forc_norm_all
        data_all['time'] = time_all
        idx = self.t_range.get_indexer(pd.to_datetime([time_all[i][0] for i in range(len(time_all))])) # type: ignore
        data_all['glac_area_t0'] = self.glac_band_area[:, idx].T

        return data_trn, data_val, data_tst, data_all

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

    def link_glac_band_basin(self):
        basin_codes = sorted(list(set([band_name.split('_')[0] for band_name in self.glac_band_codes])))
        id_dict = {}
        for i, bsn_code in enumerate(basin_codes):
            id_dict[i] = [j for j, name in enumerate(self.glac_band_codes) if name.startswith(bsn_code)]
        return id_dict, basin_codes

    def cal_init_snow_depth(self, s_dep_path: str, device: Union[str, torch.device]):
        s_depth_df = pd.read_csv(s_dep_path, index_col=0, parse_dates=True, sep=r'\s+')
        s_depth_df.columns = [str(x).zfill(12) for x in s_depth_df.columns]
        s_depth_df = s_depth_df.reindex(columns=self.glac_band_codes)
        s_dep_t0 = torch.tensor(s_depth_df.mean(axis=0).values, dtype=torch.float32, device=device)
        return s_dep_t0

    def load_init_glacier_area(self, glac_sim_path: str, glac_band_codes_all: List[str]):
        with open(f'{glac_sim_path}/sim.pkl', 'rb') as f:
            sim = pickle.load(f)
        # get index of time series
        sim_ts = pd.date_range('1955-1-1', '2019-12-31')
        idx_ts = sim_ts.get_indexer(self.t_range)

        # get index of glacier bands
        idx_band = [glac_band_codes_all.index(band) for band in self.glac_band_codes]
        band_area = np.concatenate((sim['train']['band']['g_area'],
                                    sim['valid']['band']['g_area'],
                                    sim['test']['band']['g_area']), axis=1)
        band_area = band_area[:, idx_ts][idx_band, :]
        # get index of basins
        bsn_codes_all = sorted(list(set([band_name.split('_')[0] for band_name in glac_band_codes_all])))
        idx_bsn = [bsn_codes_all.index(bsn) for bsn in self.glac_bsn_codes]
        bsn_area = np.concatenate((sim['train']['basin']['g_area'],
                                      sim['valid']['basin']['g_area'],
                                      sim['test']['basin']['g_area']), axis=1)
        bsn_area = bsn_area[:, idx_ts][idx_bsn, :]
        return band_area, bsn_area

    def print(self, msg):
        if self.logger:
            logging.info(msg)
        else:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {msg}')


class MyDataset(dataset.Dataset):
    def __init__(self, bsn_data_dict: dict, glac_data_dict: dict, device: Union[str, torch.device],
                 mode: str = 'train'):
        assert mode in ['train', 'valid', 'test', 'all']
        self.mode = mode
        if mode == 'train':
            # glacier band forcing data
            self.glac_x = torch.tensor(glac_data_dict['forc'], device=device, dtype=torch.float32)
            self.glac_xn = torch.tensor(glac_data_dict['forc_norm'], device=device, dtype=torch.float32)
            self.glac_attrs_norm = torch.tensor(glac_data_dict['attrs_norm'], device=device, dtype=torch.float32)
            self.glac_area_t0 = torch.tensor(glac_data_dict['glac_area_t0'], device=device, dtype=torch.float32)
            # basin forcing data
            self.bsn_x = torch.tensor(bsn_data_dict['forc'], device=device, dtype=torch.float32)
            self.bsn_xn = torch.tensor(bsn_data_dict['forc_norm'], device=device, dtype=torch.float32)
            self.bsn_attrs_norm = torch.tensor(bsn_data_dict['bsn_attrs_norm'], device=device, dtype=torch.float32)
            self.riv_attrs_norm = torch.tensor(bsn_data_dict['riv_attrs_norm'], device=device, dtype=torch.float32)
            self.time = glac_data_dict['time']
        else:
            # glacier band forcing data
            self.glac_x = [torch.tensor(arr, device=device, dtype=torch.float32) for arr in glac_data_dict['forc']]
            self.glac_xn = [torch.tensor(arr, device=device, dtype=torch.float32) for arr in
                            glac_data_dict['forc_norm']]
            self.glac_attrs_norm = torch.tensor(glac_data_dict['attrs_norm'], device=device, dtype=torch.float32)
            self.glac_area_t0 = torch.tensor(glac_data_dict['glac_area_t0'], device=device, dtype=torch.float32)
            # basin forcing data
            self.bsn_x = [torch.tensor(arr, device=device, dtype=torch.float32) for arr in bsn_data_dict['forc']]
            self.bsn_xn = [torch.tensor(arr, device=device, dtype=torch.float32) for arr in bsn_data_dict['forc_norm']]
            self.bsn_attrs_norm = torch.tensor(bsn_data_dict['bsn_attrs_norm'], device=device, dtype=torch.float32)
            self.riv_attrs_norm = torch.tensor(bsn_data_dict['riv_attrs_norm'], device=device, dtype=torch.float32)
            self.time = glac_data_dict['time']

    def __getitem__(self, index):
        if self.mode == 'train':
            glac_forc = self.glac_x[index]
            glac_forc_norm = self.glac_xn[index]
            glac_attrs_norm = self.glac_attrs_norm[index]
            glac_area_t0 = self.glac_area_t0[index]
            bsn_forc = self.bsn_x[index]
            bsn_forc_norm = self.bsn_xn[index]
            bsn_attrs_norm = self.bsn_attrs_norm[index]
            riv_attrs_norm = self.riv_attrs_norm[index]
            time = self.time[index]
        else:
            glac_forc = self.glac_x[index]
            glac_forc_norm = self.glac_xn[index]
            glac_attrs_norm = self.glac_attrs_norm
            glac_area_t0 = self.glac_area_t0[index]
            bsn_forc = self.bsn_x[index]
            bsn_forc_norm = self.bsn_xn[index]
            bsn_attrs_norm = self.bsn_attrs_norm
            riv_attrs_norm = self.riv_attrs_norm
            time = self.time[index]
        glac_inputs = [glac_forc, glac_forc_norm, glac_attrs_norm, glac_area_t0]
        bsn_inputs = [bsn_forc, bsn_forc_norm, bsn_attrs_norm, riv_attrs_norm]
        return glac_inputs, bsn_inputs, time

    def __len__(self):
        return len(self.bsn_x)


if __name__ == '__main__':
    # loader = RainfallRunoffLoader(bsn_forc_dir='../../../data/forcing/basin',
    #                               bsn_riv_attr_dir='../../../data/attrs/bsn_riv',
    #                               glac_forc_dir='../../../data/forcing/glacier',
    #                               glac_attr_dir='../../../data/attrs/glacier',
    #                               glac_sim_path='../../../data/pretrain',
    #                               periods={'train': [['1991-1-1', '2009-12-31']],
    #                                    'valid': [['2010-1-1', '2014-12-31']],
    #                                    'test': [['2015-1-1', '2019-12-31']]},
    #                               seq_len=1096, spin_up_len=1096, win_sz=365, device='cpu', logger=False, padding=-9999)
    loader = RainfallRunoffLoader(bsn_forc_dir=r'E:\Research\TP\Forcing\dataset\ERA5-land\basin',
                                  bsn_riv_attr_dir='../../../data/attrs/bsn_riv',
                                  glac_forc_dir=r'E:\Research\TP\Forcing\dataset\ERA5-land\glacier_1.0',
                                  glac_attr_dir='../../../data/attrs/glacier',
                                  glac_sim_path='../../../data/pretrain',
                                  periods={'train': [['1991-1-1', '2009-12-31']],
                                       'valid': [['2010-1-1', '2014-12-31']],
                                       'test': [['2015-1-1', '2019-12-31']]},
                                  seq_len=1096, spin_up_len=1096, win_sz=365, device='cpu', logger=False, padding=-9999,
                                  seq_len_eval=10)
