from collections import defaultdict
import pandas as pd
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def train_model(model, loader, loss_fn, optimizer, scheduler, early_stop, config):
    def train(model, loader_trn, loss_fn, optimizer, scheduler, config, state_t0):
        model.train()
        total_loss, metric, num_nan_loss = 0.0, defaultdict(list), 0
        spin_up_len, win_sz = config['data']['spin_up_len'], config['data']['win_sz']
        metric_name = config['train']['metric']
        for i, (forc, forc_norm, attrs_norm, ts) in enumerate(loader_trn):
            forc, forc_norm, attrs_norm = forc.squeeze(0), forc_norm.squeeze(0), attrs_norm.squeeze(0)
            ts = pd.to_datetime(ts.squeeze(0))
            output_band, output_bsn = model(forc=forc, forc_norm=forc_norm, attrs_norm=attrs_norm, ts=ts,
                                            spin_up_len=spin_up_len, mode='train', initial_state=state_t0)
            loss, metric_dict = loss_fn(sim_band=output_band, sim_bsn=output_bsn, ts=ts, spin_up_len=spin_up_len)
            idx = win_sz - 1 if i < len(loader_trn) - 1 else -spin_up_len
            if config['train']['gwe_swe_t0']:
                state_t0 = {
                    'area_band': output_band['g_area'].detach().clone()[:, idx],
                    'area_bsn': output_bsn['g_area'].detach().clone()[:, idx],
                    'gwe_band': output_band['g_we'].detach().clone()[:, idx],
                    'swe_band': output_band['s_we'].detach().clone()[:, idx]
                }
            else:
                state_t0 = {
                    'area_band': output_band['g_area'].detach().clone()[:, idx],
                    'area_bsn': output_bsn['g_area'].detach().clone()[:, idx],
                }

            # update the loss
            if ~torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip'])
                optimizer.step()
                total_loss += loss.item()
            else:
                num_nan_loss += 1
            # update the metric
            for k, v in metric_dict.items():
                if ~np.isnan(v):
                    metric[k].append(v)
            metric_logs = ', '.join([f'{k} {metric_name}: {v:.3f}' for k, v in metric_dict.items() if ~np.isnan(v)])
            logging.info(f'Iter {i + 1} of {len(loader_trn)}: loss：{loss.item():.3f}, {metric_logs}')
        # calculate the epoch loss and update the scheduler
        loss_epoch = total_loss / (len(loader_trn) - num_nan_loss)
        # calculate the epoch metric
        metric_epoch = {k: np.mean(v) for k, v in metric.items()}

        return loss_epoch, metric_epoch, state_t0

    def valid(model, loader_trn, loss_fn, config, state_t0):
        model.eval()
        total_loss, metric, num_nan_loss = 0.0, defaultdict(list), 0
        spin_up_len, metric_name = config['data']['spin_up_len'], config['train']['metric']
        with torch.no_grad():
            for i, (forc, forc_norm, attrs_norm, ts) in enumerate(loader_trn):
                forc, forc_norm, attrs_norm = forc.squeeze(0), forc_norm.squeeze(0), attrs_norm.squeeze(0)
                ts = pd.to_datetime(ts.squeeze(0))
                output_band, output_bsn = model(forc=forc, forc_norm=forc_norm, attrs_norm=attrs_norm, ts=ts,
                                                spin_up_len=spin_up_len, mode='train', initial_state=state_t0)
                loss, metric_dict = loss_fn(sim_band=output_band, sim_bsn=output_bsn, ts=ts, spin_up_len=spin_up_len)
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

                # update the loss
                if ~torch.isnan(loss):
                    total_loss += loss.item()
                else:
                    num_nan_loss += 1
                # update the metric
                for k, v in metric_dict.items():
                    if ~np.isnan(v):
                        metric[k].append(v)
        # calculate the epoch loss
        loss_epoch = total_loss / (len(loader_trn) - num_nan_loss)
        # calculate the epoch metric
        metric_epoch = {k: np.mean(v) for k, v in metric.items()}

        return loss_epoch, metric_epoch, state_t0

    loss_trn_lst, loss_val_lst = [], []
    metric_name = config['train']['metric']
    for epoch in range(config['train']['epochs']):
        logging.info('*' * 100)
        logging.info('Epoch:{:d}/{:d}'.format(epoch, config['train']['epochs']))
        loss_trn, metric_trn, state_t0 = train(model=model, loader_trn=loader.loader_trn, loss_fn=loss_fn,
                                               optimizer=optimizer, scheduler=scheduler, config=config, state_t0=None)
        loss_trn_lst.append(loss_trn)
        metric_logs = ', '.join([f'{k} {metric_name}: {v:.3f}' for k, v in metric_trn.items()])
        logging.info(f'Epoch training loss: {loss_trn:.3f}, {metric_logs}')

        loss_val, metric_val, state_t0 = valid(model=model, loader_trn=loader.loader_val, loss_fn=loss_fn,
                                               config=config,
                                               state_t0=state_t0)
        loss_val_lst.append(loss_val)
        metric_logs = ', '.join([f'{k} {metric_name}: {v:.3f}' for k, v in metric_val.items()])
        logging.info(f'Epoch validation loss: {loss_val:.3f}, {metric_logs}')
        if scheduler is not None:
            scheduler.step(loss_val)
        early_stop(loss_val, model)
        if early_stop.stop:
            logging.info(f'Early stopping with best loss: {early_stop.best_loss: .3f}')
            break
    plot_loss_curve(loss_trn_lst, loss_val_lst, os.path.join(config['out'], 'loss.png'))


def plot_loss_curve(loss_trn_lst: list, loss_val_lst: list, out_path: str):
    plt.rcParams.update({'mathtext.fontset': 'custom'})
    plt.rcParams['font.family'] = 'Arial'
    sns.set_theme(style='ticks')

    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(range(len(loss_trn_lst)), loss_trn_lst, linewidth=1, label='trn')
    ax.plot(range(len(loss_val_lst)), loss_val_lst, linewidth=1, label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path)
