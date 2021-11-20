'''
General utils (figures, etc.).

Written by Ido Greenberg, 2020
'''

import gc
from time import sleep
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_quantiles(X, Q=100, assume_sorted=False):
    if not assume_sorted:
        X = sorted(X)
    N = len(X)
    try:
        n = len(Q)
        Q = np.array(Q)
    except:
        Q = np.arange(0,1+1e-6,1/Q)
        Q[-1] = 1
        n = len(Q)
    x = [X[int(q*(N-1))] for q in Q]
    return x, Q

def plot_quantiles(x, ax=None, Q=100, showmeans=False, plot_mean=False, means_args=None, **kwargs):
    if plot_mean: showmeans = True # backward compatibility
    if ax is None: ax = Axes(1,1)[0]
    m = np.mean(x)
    x, q = get_quantiles(x, Q)
    h = ax.plot(100*q, x, '-', **kwargs)
    if showmeans:
        if means_args is None: means_args = {}
        ax.axhline(m, linestyle='--', color=h[0].get_color(), **means_args)
    return ax

def labels(ax, xlab=None, ylab=None, title=None, fontsize=12):
    if isinstance(fontsize, int):
        fontsize = 3*[fontsize]
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=fontsize[0])
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=fontsize[1])
    if title is not None:
        ax.set_title(title, fontsize=fontsize[2])

def fontsize(ax, labs=16, ticks=12, leg=None, draw=True, wait=0):
    if wait:
        sleep(wait)
    if draw:
        plt.draw()
    if labs is not None:
        if not isinstance(labs, (tuple,list)):
            labs = 3*[labs]
            ax.set_xlabel(ax.get_xlabel(), fontsize=labs[0])
            ax.set_ylabel(ax.get_ylabel(), fontsize=labs[1])
            ax.set_title(ax.get_title(), fontsize=labs[2])
    if ticks is not None:
        if not isinstance(ticks, (tuple,list)):
            ticks = 2*[ticks]
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=ticks[0])
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=ticks[1])
    if leg is not None:
        ax.legend(fontsize=leg)

class Axes:
    def __init__(self, N, W=2, axsize=(5,3.5), grid=1, fontsize=13):
        self.fontsize = fontsize
        self.N = N
        self.W = W
        self.H = int(np.ceil(N/W))
        self.axs = plt.subplots(self.H, self.W, figsize=(self.W*axsize[0], self.H*axsize[1]))[1]
        for i in range(self.N):
            if grid == 1:
                self[i].grid(color='k', linestyle=':', linewidth=0.3)
            elif grid ==2:
                self[i].grid()
        for i in range(self.N, self.W*self.H):
            self[i].axis('off')

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if self.H == 1 and self.W == 1:
            return self.axs
        elif self.H == 1 or self.W == 1:
            return self.axs[item]
        return self.axs[item//self.W, item % self.W]

    def labs(self, item, *args, **kwargs):
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self.fontsize
        labels(self[item], *args, **kwargs)

def to_device(model, tensor_list):
    if model is not None:
        model.to(DEVICE)
    for i,tns in enumerate(tensor_list):
        if tns is not None:
            if type(tns) is not torch.Tensor:
                tns = torch.from_numpy(tns)
            tensor_list[i] = tns.to(DEVICE)
    return tensor_list

def clean_device(tensor_list):
    for i in reversed(list(range(len(tensor_list)))):
        del tensor_list[i]
    gc.collect()
    torch.cuda.empty_cache()

def update_dict(base_dict, dict_to_add=None, force=False, copy=False):
    if copy:
        base_dict = base_dict.copy()
    if dict_to_add is not None:
        for k, v in dict_to_add.items():
            if force or k not in base_dict:
                base_dict[k] = v
    return base_dict

def pd_merge_cols(dd, cols_to_merge, values_cols=('Y',), case_col='case', cases_names=None, cols_to_keep=None):
    if isinstance(cols_to_merge[0], str):   cols_to_merge = [cols_to_merge]
    if isinstance(values_cols, str):        values_cols = [values_cols]
    if cases_names is None:                 cases_names = cols_to_merge[0]
    if cols_to_keep is None:
        all_cols_to_merge = [c for cols in cols_to_merge for c in cols]
        cols_to_keep = [c for c in dd.columns if c not in all_cols_to_merge]

    n_merges = len(cols_to_merge)
    n_to_merge = len(cols_to_merge[0])
    d = pd.DataFrame()
    for c in cols_to_keep:
        d[c] = n_to_merge * list(dd[c].values)

    d[case_col] = np.repeat(cases_names, len(dd))
    for cols, values_col in zip(cols_to_merge, values_cols):
        d[values_col] = np.concatenate([dd[c].values for c in cols])

    return d

def overwrite_legend(ax, fun, filter=None, fontsize=12):
    h, cur_labs = ax.get_legend_handles_labels()
    if filter is not None:
        tmp = [(hh, l) for hh, l in zip(h, cur_labs) if filter(l)]
        h = [tt[0] for tt in tmp]
        cur_labs = [tt[1] for tt in tmp]
    labs = [fun(l) for l in cur_labs]
    ax.legend(h, labs, fontsize=fontsize)

def compare_with_ref(dd, y, x, ref=None, x_ord=None, hue=None, sort_hue=True, ax=None, font=14, **kwargs):
    if ax is None: ax = Axes(1,1,axsize=(6,4))[0]
    if x_ord is None: x_ord = sorted(np.unique(dd[x]))
    if ref is None: ref = x_ord[0]
    x_ord = [xx for xx in x_ord if xx!=ref]

    dd = dd.copy()
    dd.loc[:,'delta'] = 0.
    ids0 = dd[x] == ref
    for xx in x_ord:
        ids = dd[x] == xx
        if np.sum(ids) != np.sum(ids0):
            # cant run pairwise comparison (unpaired comparison is not implemented)
            return
        dd.loc[ids,'delta'] = dd.loc[ids,y].values - dd.loc[ids0,y].values

    dd = dd[dd[x]!=ref]
    if len(dd) == 0:
        warn('A single x - nothing to compare to.')
        return

    sns.barplot(data=dd, x=x, y='delta', hue=hue, hue_order=sorted(dd[hue]) if hue and sort_hue else None,
                ax=ax, capsize=.1, **kwargs)
    if hue is not None:
        ax.legend(fontsize=12)
    ylab = f'{y:s} difference'
    labels(ax, 'Alternative', ylab, f'Alternative vs. {ref:s}', fontsize=font)
    plt.draw()
    fontsize(ax, font, 12, None if hue is None else 12)
    if len(x_ord) > 4:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25)

def qgroups(x, nbins=5, apply_labs=True):
    qs, _ = get_quantiles(x, nbins)
    qs = np.array(qs)
    qs[-1] += 1
    g = [int(np.sum(xx>=qs))-1 for xx in x]
    qs[-1] -= 1
    v = [f'{qs[gg]:.0f}-{qs[gg+1]:.0f}' for gg in g] if apply_labs else g
    sorted_v = [vv for _,vv in sorted(zip(g,v))]
    ids = np.unique(sorted_v, return_index=True)[1]
    v_labs = [sorted_v[i] for i in sorted(ids)]
    return g, v, v_labs

def compare_quantiles(dd, x, y, hue=None, fac=None, xbins=5, hbins=5, fbins=4, ci=95, box=False,
                      mean=np.mean, copy=True, axs=None, axs_args=None, **kwargs):
    if copy: dd = dd.copy()
    # prepare groups
    xx = x+'_tmp'
    relabel = False
    if xbins and len(np.unique(dd[x]))>2.5*xbins:
        g, v, relabel = qgroups(dd[x], xbins)
        dd[xx] = g
    else:
        dd[xx] = dd[x]
    hues = None
    if hue:
        hh = hue + '_tmp'
        dd[hh] = dd[hue]
        hues = np.unique(dd[hh])
        if len(hues) > 1.7*hbins:
            _, dd[hue], hues = qgroups(dd[hh], hbins)
    facs = ff = None
    if fac:
        ff = fac + '_tmp'
        facs = np.unique(dd[fac])
        if len(facs) > 2*fbins:
            _, dd[ff], facs = qgroups(dd[fac], fbins)
        else:
            dd[ff] = dd[fac]
        nf = len(facs)
    else:
        nf = 1

    # prepare axes
    if axs is None:
        if axs_args is None: axs_args = {}
        axs_def_args = dict(W=3, axsize=(7,5), fontsize=15)
        axs_args = update_dict(axs_args, axs_def_args)
        axs = Axes(nf, **axs_args)

    # plot
    pd.options.mode.chained_assignment = None  # default='warn'
    for a in range(nf):
        ax = axs[a]
        d = dd[dd[ff]==facs[a]] if fac else dd
        curr_hues = [h for h in hues] if hue else None
        if mean is not None and hue:
            for i,h in enumerate(curr_hues):
                m = mean(d[d[hue]==h][y])
                hh = f'{h:s} ({m:.0f})'
                d.loc[d[hue]==h, hue] = hh
                curr_hues[i] = hh
        if box:
            sns.boxplot(data=d, x=xx, y=y, hue=hue, hue_order=curr_hues, ax=ax, showmeans=True, **kwargs)
        else:
            sns.lineplot(data=d, x=xx, y=y, hue=hue, hue_order=curr_hues, ax=ax, ci=ci, **kwargs)
        if relabel:
            ax.set_xticks(np.arange(len(relabel)))
            ax.set_xticklabels(relabel, fontsize=12, rotation=20)
        labels(axs[a], x, y, f'{fac}: {facs[a]}' if fac else None, fontsize=15)
        axs[a].legend(title=hue, fontsize=12)
    pd.options.mode.chained_assignment = 'warn'
    plt.tight_layout()

    return axs
