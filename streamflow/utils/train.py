from collections import defaultdict
import pandas as pd
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def train_model(glac_model, rr_model, glac_optimizer, rr_optimizer, glac_scheduler, rr_scheduler, loader, loss_fn, early_stop, config):
    def train(glac_model, rr_model, glac_optimizer, rr_optimizer, loader_trn, loss_fn, config):
        # set the model to training mode
        if config['train']['glac_loss']['retrain']:
            glac_model.train()
        else:
            glac_model.eval()
        rr_model.train()

        total_loss, metric, num_nan_loss = 0.0, defaultdict(list), 0
        spin_up_len, win_sz = config['data']['spin_up_len'], config['data']['win_sz']
        metric_name = config['train']['metric']

        for i, (glac_inputs, bsn_inputs, time) in enumerate(loader_trn):
            # glacier model
            ts = pd.to_datetime(time.squeeze(0))
            glac_forc, glac_forc_norm, glac_attrs_norm, glac_area_t0 = glac_inputs
            glac_forc, glac_forc_norm, glac_attrs_norm = (glac_forc.squeeze(0), glac_forc_norm.squeeze(0),
                                                          glac_attrs_norm.squeeze(0))
            glac_out_band, glac_out_bsn = glac_model(forc=glac_forc, forc_norm=glac_forc_norm,
                                                     attrs_norm=glac_attrs_norm, glac_area_t0=glac_area_t0,
                                                     ts=ts, spin_up_len=spin_up_len, mode='train')
            if config['train']['glac_loss']['retrain'] is False:
                glac_out_bsn = {k: v.detach() for k, v in glac_out_bsn.items()}
            # rainfall-runoff model
            if sum(config['train']['rr_loss']['weight'].values()) > 0:
                bsn_forc, bsn_forc_norm, bsn_attrs_norm, riv_attrs_norm = bsn_inputs
                bsn_forc, bsn_forc_norm, bsn_attrs_norm, riv_attrs_norm = (bsn_forc.squeeze(0), bsn_forc_norm.squeeze(0),
                                                                           bsn_attrs_norm.squeeze(0), riv_attrs_norm.squeeze(0))
                bsn_output = rr_model(forc=bsn_forc, forc_norm=bsn_forc_norm,
                                      bsn_attrs_norm=bsn_attrs_norm, riv_attrs_norm=riv_attrs_norm,
                                      glac_sim_bsn=glac_out_bsn, spin_up_len=spin_up_len, mode='train')
            else:
                bsn_output = None
            loss, metric_dict = loss_fn(glac_sim_band=glac_out_band, glac_sim_bsn=glac_out_bsn, rr_sim_bsn=bsn_output,
                                        ts=ts, spin_up_len=spin_up_len)
            # update the loss
            if ~torch.isnan(loss):
                # Always zero out all gradients that might be used
                if config['train']['glac_loss']['retrain'] and sum(config['train']['rr_loss']['weight'].values()) > 0:
                    glac_optimizer.zero_grad()
                if sum(config['train']['rr_loss']['weight'].values()) > 0:
                    rr_optimizer.zero_grad()
                # Calculate gradients for the entire graph based on the loss
                loss.backward()
                # Step only the optimizers that are meant to be trained in this iteration
                if config['train']['glac_loss']['retrain'] and sum(config['train']['rr_loss']['weight'].values()) > 0:
                    glac_optimizer.step()
                if sum(config['train']['rr_loss']['weight'].values()) > 0:
                    rr_optimizer.step()
                total_loss += loss.item()
            else:
                num_nan_loss += 1

            # update the metric
            log_parts = []
            for k, v in metric_dict.items():
                if ~np.isnan(v):
                    metric[k].append(v)
                    log_parts.append(f'{k}: {v:.3f}')
            metric_logs = ', '.join(log_parts)
            logging.info(f'Iter {i + 1} of {len(loader_trn)}: loss: {loss.item():.3f}, {metric_logs}')
        # calculate the epoch loss and update the scheduler
        loss_epoch = total_loss / (len(loader_trn) - num_nan_loss)
        # calculate the epoch metric
        metric_epoch = {k: np.mean(v) for k, v in metric.items()}

        return loss_epoch, metric_epoch

    def valid(glac_model, rr_model, loader_val, loss_fn, config):
        glac_model.eval()
        rr_model.eval()
        total_loss, metric, num_nan_loss = 0.0, defaultdict(list), 0
        spin_up_len, metric_name = config['data']['spin_up_len'], config['train']['metric']
        with torch.no_grad():
            for i, (glac_inputs, bsn_inputs, time) in enumerate(loader_val):
                # glacier model
                ts = pd.to_datetime(time.squeeze(0))
                glac_forc, glac_forc_norm, glac_attrs_norm, glac_area_t0 = glac_inputs
                glac_forc, glac_forc_norm, glac_attrs_norm = (glac_forc.squeeze(0), glac_forc_norm.squeeze(0),
                                                              glac_attrs_norm.squeeze(0))
                glac_out_band, glac_out_bsn = glac_model(forc=glac_forc, forc_norm=glac_forc_norm,
                                                         attrs_norm=glac_attrs_norm, glac_area_t0=glac_area_t0,
                                                         ts=ts, spin_up_len=spin_up_len, mode='train')

                # rainfall-runoff model
                if sum(config['train']['rr_loss']['weight'].values()) > 0:
                    bsn_forc, bsn_forc_norm, bsn_attrs_norm, riv_attrs_norm = bsn_inputs
                    bsn_forc, bsn_forc_norm, bsn_attrs_norm, riv_attrs_norm = (bsn_forc.squeeze(0), bsn_forc_norm.squeeze(0),
                                                                               bsn_attrs_norm.squeeze(0), riv_attrs_norm.squeeze(0))
                    bsn_output = rr_model(forc=bsn_forc, forc_norm=bsn_forc_norm,
                                          bsn_attrs_norm=bsn_attrs_norm, riv_attrs_norm=riv_attrs_norm,
                                          glac_sim_bsn=glac_out_bsn, spin_up_len=spin_up_len, mode='train')
                else:
                    bsn_output = None

                loss, metric_dict = loss_fn(glac_sim_band=glac_out_band, glac_sim_bsn=glac_out_bsn,
                                            rr_sim_bsn=bsn_output, ts=ts, spin_up_len=spin_up_len)

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
        loss_epoch = total_loss / (len(loader_val) - num_nan_loss)
        # calculate the epoch metric
        metric_epoch = {k: np.mean(v) for k, v in metric.items()}

        return loss_epoch, metric_epoch

    loss_trn_lst, loss_val_lst = [], []
    metric_name = config['train']['metric']
    # get the initial learning rate
    if config['train']['glac_loss']['retrain']:
        glac_last_lr = glac_optimizer.param_groups[0]['lr']
    else:
        glac_last_lr = None
    rr_last_lr = rr_optimizer.param_groups[0]['lr']
    # loop over the epochs
    for epoch in range(config['train']['epochs']):
        logging.info('*' * 100)
        logging.info('Epoch:{:d}/{:d}'.format(epoch, config['train']['epochs']))
        loss_trn, metric_trn = train(glac_model=glac_model, rr_model=rr_model, glac_optimizer=glac_optimizer,
                                     rr_optimizer=rr_optimizer, loader_trn=loader.loader_trn, loss_fn=loss_fn, config=config)
        loss_trn_lst.append(loss_trn)
        metric_logs = ', '.join([f'{k} {metric_name}: {v:.3f}' for k, v in metric_trn.items()])
        logging.info(f'Epoch training loss: {loss_trn:.3f}, {metric_logs}')

        loss_val, metric_val = valid(glac_model=glac_model, rr_model=rr_model, loader_val=loader.loader_val,
                                     loss_fn=loss_fn, config=config)
        loss_val_lst.append(loss_val)
        metric_logs = ', '.join([f'{k} {metric_name}: {v:.3f}' for k, v in metric_val.items()])
        logging.info(f'Epoch validation loss: {loss_val:.3f}, {metric_logs}')
        # step the scheduler and update the learning rate
        if config['train']['glac_loss']['retrain'] and sum(config['train']['rr_loss']['weight'].values()) > 0:
            glac_scheduler.step(loss_val)
            rr_scheduler.step(loss_val)
            glac_current_lr = glac_optimizer.param_groups[0]['lr']
            if glac_current_lr != glac_last_lr:
                logging.info(f"Learning rate of glac_optimizer changed from {glac_last_lr} to {glac_current_lr}")
                glac_last_lr = glac_current_lr
            rr_current_lr = rr_optimizer.param_groups[0]['lr']
            if rr_last_lr != rr_current_lr:
                logging.info(f"Learning rate of rr_optimizer changed from {rr_last_lr} to {rr_current_lr}")
                rr_last_lr = rr_current_lr
        elif sum(config['train']['rr_loss']['weight'].values()) > 0:
            rr_scheduler.step(loss_val)
            rr_current_lr = rr_optimizer.param_groups[0]['lr']
            if rr_last_lr != rr_current_lr:
                logging.info(f"Learning rate of rr_optimizer changed from {rr_last_lr} to {rr_current_lr}")
                rr_last_lr = rr_current_lr
        else:
            glac_scheduler.step(loss_val)
            glac_current_lr = glac_optimizer.param_groups[0]['lr']
            if glac_current_lr != glac_last_lr:
                logging.info(f"Learning rate of glac_optimizer changed from {glac_last_lr} to {glac_current_lr}")
                glac_last_lr = glac_current_lr
        # update the early stopping
        early_stop(loss_val, glac_model, rr_model)
        if early_stop.stop:
            logging.info(f'Early stopping with best loss: {early_stop.best_loss: .3f}')
            break
        # torch.save(glac_model.state_dict(), config['out'] + '/model_glac.pt')
        # torch.save(rr_model.state_dict(), config['out'] + '/model_rr.pt')
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