'''
Optimizing and testing the Kalman Filter from data of trajectories with both observations and system states.

Contents:
- TRAIN a KF from data: both noise-estimation and gradient-based optimization are supported.
- TEST a trained model.
- ANALYSIS of the test results.

Written by Ido Greenberg, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from warnings import warn
from time import time

import torch
from torch import optim

from . import utils

###################   TRAIN   ###################

def train_models(models, X, Y, **kwargs):
    '''Run train() iteratively for every model.'''
    res_per_iter, res_per_sample = pd.DataFrame(), pd.DataFrame()
    for m in models:
        r1, r2 = train(m, X, Y, **kwargs)
        res_per_iter = pd.concat((res_per_iter, r1))
        res_per_sample = pd.concat((res_per_sample, r2))
    return res_per_iter, res_per_sample

def train(model, X, Y, split_data=None, p_valid=0.15, n_epochs=1, batch_size=10,
          lr=1e-2, lr_decay=0.5, lr_decay_freq=150, optimizer=optim.Adam, weight_decay=0.0,
          loss_after_pred=False, log_interval=300, reset_model=True, noise_estimation_initialization=True,
          best_valid_loss=np.inf, verbose=1, valid_hor=8, to_save=True, save_best=True, **kwargs):
    '''
    Training of the Kalman Filter from data of trajectories that includes both observations (X) and system states (Y).

    If the input model is configured not to be optimized, then this function only triggers the noise-estimation tuning
    from the given data.

    :param model: An instance of OKF to train.
    :param X: Observations data - list of numpy arrays of shape (n_time_steps, observaton dimension).
    :param Y: System states data - list of numpy arrays of shape (n_time_steps, state dimension).
    :param split_data: Train and validation data (Xt,Yt,Xv,Yv). If None, these are generated from X,Y. Default = None.
    :param p_valid: Percent of data used for validation (if split_data is None). Default = 0.15.
    :param n_epochs: Epochs to train (how many times to sample each trajectory). Default = 1.
    :param batch_size: Number of trajectories per training iteration. Default = 10.
    :param lr: Initial learning rate. Default = 1e-2.
    :param lr_decay:      Every lr_decay_freq iterations, lr is multiplied by lr_decay. Default = 0.5.
    :param lr_decay_freq: Every lr_decay_freq iterations, lr is multiplied by lr_decay. If None, frequency is once per
                          epoch. Default = 150.
    :param optimizer: Optimizer constructor to use. Default = optim.Adam.
    :param weight_decay: Weight decay of the parameters in the optimizer. Default = 0. Decay>0 may harm stability.
    :param loss_after_pred: Whether to calculate the loss after the prediction step or after the update (filter) step.
                            Default = False (after update step).
    :param log_interval: Validation loss is calculated every log_interval training samples. Default = 300.
    :param reset_model: Whether to reset the model before the training begins. Default = True.
    :param noise_estimation_initialization: Whether to initialize the weights using noise estimation. Default = True.
    :param best_valid_loss: Best validation loss until now. Used for continuing training. Default = inf.
    :param verbose: Verbosity level. Default = 2.
    :param valid_hor: After valid_hor validation calculations in a row w/o improvement, the training stops. Default = 8.
    :param to_save: Whether to save the resulting model. If string, saves the model under this name. If True, saves
                    under model.model_name. Default = True.
    :param save_best: Whether to save the last model or the one with the best validation loss. Default = True.
    :param kwargs: Arguments to pass to train_step().
    :return: Two data frames describing the results: losses per iteration; and final validation loss per data sample.
    '''

    # pre-processing
    Xt0, Yt0, Xv, Yv = split_train_valid(X, Y, p_valid) if split_data is None else split_data
    n_samples = len(Xt0)
    n_batches = n_samples // batch_size
    log_interval = log_interval // batch_size
    if lr_decay_freq is None: lr_decay_freq = n_batches

    if reset_model:
        model.reset_model()

    if noise_estimation_initialization or not model.optimize:
        # estimate noise covariance
        model.estimate_noise(Y,X)

    if model.optimize:
        # initialize variables
        early_stop = False
        no_improvement_seq = 0
        e = 0
        b = 0
        # train monitor
        t = []
        losses = []
        RMSE = []
        # validation monitor
        t_valid = []
        losses_valid = []
        RMSE_valid = []

        # initialize model and optimizer
        model.train()
        if weight_decay != 0:
            warn('Note: weight decay is known to cause instabilities in the tracker training.')
        params = model.parameters()
        o = optimizer(params, lr=lr, weight_decay=weight_decay)
        sched = optim.lr_scheduler.StepLR(o, step_size=lr_decay_freq, gamma=lr_decay)

        if verbose >= 1:
            print(f'\nTraining {model.model_name:s}:')
            print(f'samples={len(Xt0):d}(t)+{len(Xv):d}(v)={len(X):d}; batch_size={batch_size:d}; ' + \
                  f'iterations={n_epochs}(e)x{n_batches}(b)={n_epochs * n_batches:d}.')

        # train
        T0 = time()
        for e in range(n_epochs):
            t0 = time()

            # shuffle batches
            ids = np.arange(len(Xt0))
            np.random.shuffle(ids)
            Xt = [Xt0[i] for i in ids]
            Yt = [Yt0[i] for i in ids]

            for b in range(n_batches):
                tt = e * n_batches + b
                x = [Xt[b*batch_size+i] for i in range(batch_size)]
                y = [Yt[b*batch_size+i] for i in range(batch_size)]

                loss_batch = train_step(x, y, model, o, loss_after_pred=loss_after_pred, **kwargs)

                t.append(tt+1)
                losses.append(loss_batch)
                RMSE.append(np.sqrt(loss_batch))

                if log_interval > 0 and ((tt % log_interval == 0) or tt==n_epochs*n_batches-1):
                    # calculate validation loss
                    loss_batch = test_model(model, Xv, Yv, detailed=False, loss_after_pred=loss_after_pred)
                    model.train()

                    t_valid.append(tt+1)
                    losses_valid.append(loss_batch)
                    RMSE_valid.append(np.sqrt(loss_batch))

                    if verbose >= 2:
                        print(f'\t[{model.model_name:s}] {e + 1:02d}.{b + 1:04d}/{n_epochs:02d}.{n_batches:04d}:\t' + \
                              f'train_RMSE={RMSE[-1]:.2f}, valid_RMSE={RMSE_valid[-1]:.2f}   |   {time() - t0:.0f} [s]')

                    if len(losses_valid)>1 and np.min(losses_valid[:-1])<best_valid_loss:
                        best_valid_loss = np.min(losses_valid[:-1])
                    improved = loss_batch < best_valid_loss
                    if improved:
                        no_improvement_seq = 0
                        if to_save and save_best:
                            model.save_model(to_save if isinstance(to_save, str) else None)
                    else:
                        no_improvement_seq += 1
                    if no_improvement_seq >= valid_hor:
                        early_stop = True
                        break

                # update lr
                sched.step()

            if verbose >= 2:
                print(f'[{model.model_name:s}] Epoch {e + 1}/{n_epochs} ({time() - t0:.0f} [s])')

            if early_stop:
                break

        if verbose >= 1:
            print_train_summary(n_epochs, n_batches, early_stop, e, b, best_valid_loss, T0, model.model_name)

        # summarize losses per iteration
        res = pd.concat((
            pd.DataFrame(dict(model=len(t)*[model.model_name], t=t, group=len(t)*['train'],
                              loss=losses, RMSE=RMSE)),
            pd.DataFrame(dict(model=len(t_valid)*[model.model_name], t=t_valid, group=len(t_valid)*['valid'],
                              loss=losses_valid, RMSE=RMSE_valid))
        )).copy()

    else:
        res = pd.DataFrame({})

    # save model
    if to_save:
        if save_best and model.optimize:
            # in this case saving was already done during the training - every time validation loss achieved new record.
            # need to reload the last saved model, which is the best one over all validations.
            model.load_model(to_save if isinstance(to_save, str) else None)
        else:
            model.save_model(to_save if isinstance(to_save, str) else None)

    # detailed test over the validation data
    res_valid = test_model(model, Xv, Yv, detailed=True, loss_after_pred=loss_after_pred).copy()

    return res, res_valid

def train_step(X, Y, model, optimizer, clip=1, loss_after_pred=False, optimize_per_target=False):
    '''
    A single training step - one batch of trajectories.

    For most parameters - see train() documentation.
    :param clip: Clip the gradients to this value. Default = 1.
    :param optimize_per_target: When calculating the loss, assign the same weight to every trajectory (target) rather
                                than every sample (time-step). That is, don't give more weight to longer trajectories.
    :return: The loss of this training step.
    '''

    # assign weights to errors (uniform over time-steps or uniform over targets)
    targets_lengths = np.array([len(x) for x in X])
    targets_weights = 1/targets_lengths if optimize_per_target else np.ones(len(X))
    targets_weights = targets_weights / np.sum(targets_weights*targets_lengths)

    optimizer.zero_grad()
    tot_loss = torch.tensor(0.)
    for x,y,w in zip(X,Y,targets_weights):
        model.init_state()
        for t in range(len(x)):
            xx = x[t,:]
            yy = y[t,:]

            loss = None
            model.predict()
            if loss_after_pred:
                loss = model.loss_fun(model.x, torch.tensor(yy)) if t>0 else torch.tensor(0.)

            model.update(xx)
            if not loss_after_pred:
                loss = model.loss_fun(model.x, torch.tensor(yy))

            tot_loss = tot_loss + w * loss

    tot_loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return tot_loss.item()

def split_train_valid(X, Y, p=0.15, seed=9):
    n_valid = int(np.round(p*len(X)))
    np.random.seed(seed)
    ids_valid = set(list(np.random.choice(np.arange(len(X)), n_valid, replace=False)))
    Xt = [x for i,x in enumerate(X) if i not in ids_valid]
    Yt = [x for i,x in enumerate(Y) if i not in ids_valid]
    Xv = [x for i,x in enumerate(X) if i in ids_valid]
    Yv = [x for i,x in enumerate(Y) if i in ids_valid]
    return Xt, Yt, Xv, Yv

def print_train_summary(n_epochs, n_batches, early_stop, epoch, i_batch, valid_loss, T0, tit):
    print(f'[{tit:s}] Training done ({time() - T0:.0f} [s])')
    if early_stop:
        print(
            f'\tbest valid loss: {valid_loss:.0f};\tearly stopping:\t{epoch + 1:d}.{i_batch + 1:03d}/{n_epochs:d}.{n_batches:03d} ({100 * (epoch*n_batches + i_batch + 1) / (n_epochs * n_batches):.0f}%)')
    else:
        print(
            f'\tbest valid loss: {valid_loss:.0f};\tno early stopping:\t{n_epochs:d} epochs, {n_batches:d} batches, {n_epochs * n_batches:d} total iterations.')

###################   TEST   ###################

def test_model(model, X, Y, detailed=False, loss_fun=None, loss_after_pred=False, count_base=0, verbose=0):
    # detailed = Whether to return a detailed data-frame or just the final loss.
    with torch.no_grad():
        model.eval()
        if loss_fun is None: loss_fun = model.loss_fun
        # per-step data
        targets = []
        times = []
        losses = []
        SE = []
        AE = []
        # per-batch data
        tot_loss = 0

        count = 0
        t0 = time()
        if verbose >= 1:
            print(f'\nTesting {model.model_name:s}:')
        for tar, (XX, YY) in enumerate(zip(X, Y)):
            model.init_state()
            for t in range(len(XX)):
                count += 1
                x = XX[t,:]
                y = YY[t,:]

                model.predict()
                if loss_after_pred:
                    loss = loss_fun(model.x, y) if t>0 else torch.tensor(0.)

                model.update(x)
                if not loss_after_pred:
                    loss = loss_fun(model.x, y)

                loss = loss.item()
                tot_loss += loss

                if detailed:
                    targets.append(count_base+tar)
                    times.append(t)
                    SE.append(loss)
                    AE.append(np.sqrt(loss))
                    losses.append(loss)

        if verbose >= 1:
            print(f'done.\t({time()-t0:.0f} [s])')
        if detailed:
            return pd.DataFrame(dict(
                model = len(times) * [model.model_name],
                target = targets,
                t = times,
                SE = SE,
                AE = AE,
                loss=losses,
            ))
        tot_loss /= count
        return tot_loss

###################   ANALYSIS   ###################

def analyze_test_results(res):
    models = np.unique(res.model)
    axs = utils.Axes(6, 4, axsize=(5, 3.8))
    a = 0

    for m in models:
        mean = res[res.model==m].AE.mean()
        utils.plot_quantiles(res[res.model==m].AE.values, showmeans=True, ax=axs[a], label=f'{m}  (mean={mean:.1f})',
                             linewidth=2, means_args=dict(linewidth=2))
    axs[a].legend(fontsize=15)
    axs.labs(a, 'quantile [%]', 'error', fontsize=15)
    a += 1

    sns.boxplot(data=res, x='model', y='AE', showmeans=True, showfliers=False, ax=axs[a],
                meanprops=dict(marker='o', markerfacecolor='w', markeredgecolor='r', markersize=10))
    axs.labs(a, 'model', 'error', fontsize=16)
    axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=16)
    a += 1

    sns.barplot(data=res, x='model', y='SE', capsize=.07, ci=99, ax=axs[a])
    axs.labs(a, 'model', 'error^2 (99% confidence)', fontsize=16)
    a += 1

    sns.lineplot(data=res, x='t', hue='model', y='AE', ax=axs[a])
    axs.labs(a, 'time-step', 'error', fontsize=16)
    a += 1

    axs[a].axhline(0, color='k')
    for i1,m1 in enumerate(models):
        for m2 in models[i1+1:]:
            delta = res[res.model == m2].AE.values - res[res.model==m1].AE.values
            zval = np.mean(delta) / np.std(delta) * np.sqrt(len(delta))
            utils.plot_quantiles(delta, showmeans=True, ax=axs[a], label=f'{m2}-{m1} (z-val={zval:.1f})',
                                 linewidth=2, means_args=dict(linewidth=2))
    axs[a].legend(fontsize=15)
    axs.labs(a, 'quantile [%]', 'error(model_2) - error(model_1)',
             title=f'Models comparison in pairs\n(sample = time-step)', fontsize=13)
    a += 1

    axs[a].axhline(0, color='k')
    for i1,m1 in enumerate(models):
        for m2 in models[i1+1:]:
            delta = (res[res.model == m2].groupby('target').apply(lambda d: np.sqrt(d.SE.mean())) -
                     res[res.model == m1].groupby('target').apply(lambda d: np.sqrt(d.SE.mean())) )
            zval = np.mean(delta) / np.std(delta) * np.sqrt(len(delta))
            utils.plot_quantiles(delta, showmeans=True, ax=axs[a], label=f'{m2}-{m1} (z-val={zval:.1f})',
                                 linewidth=2, means_args=dict(linewidth=2))
    axs[a].legend(fontsize=15)
    axs.labs(a, 'target quantile [%]', 'error(model_2) - error(model_1)',
             title=f'Models comparison in pairs\n(sample = target)', fontsize=13)
    a += 1

    plt.tight_layout()
    return axs

def display_tracking(models, X, Y, n=4, t_min=0, xdim=0, ydim=1, plot_after_pred=False, show_observations=False):
    axs = utils.Axes(n, 4, axsize=(5, 4))
    colors = ['r', 'b', 'g', 'y']
    preds = {}
    for i in range(n):
        ax = axs[i]
        preds[i] = []
        XX = X[i]
        # plot target
        ax.plot(Y[i][t_min:, xdim], Y[i][t_min:, ydim], 'k.-', label='ground-truth')
        # plot observations
        if show_observations:
            ax.plot(X[i][t_min:, xdim], X[i][t_min:, ydim], 'm.', label='observation')
        ax.plot(Y[i][t_min:t_min+1, xdim], Y[i][t_min:t_min+1, ydim], 'k>', markersize=10)
        # plot models outputs
        for j, m in enumerate(models):
            preds[i].append([])
            m.init_state()
            for t in range(len(XX)):
                x = XX[t, :]
                m.predict()
                if plot_after_pred and t>0:
                    preds[i][j].append(m.x.detach().numpy())
                m.update(x)
                if not plot_after_pred:
                    preds[i][j].append(m.x.detach().numpy())
            preds[i][j] = np.stack(preds[i][j])  # prediction[target][model]
            ax.plot(preds[i][j][t_min:, xdim], preds[i][j][t_min:, ydim], colors[j%len(colors)], linewidth=1.5,
                    label=f'{m.model_name}')

        axs.labs(i, None, None, f'{t_min:d}<t<{len(XX):d}', fontsize=14)
        ax.legend(fontsize=14)

    plt.tight_layout()
    return axs
