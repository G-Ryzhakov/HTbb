import matplotlib as mpl
import numpy as np
import os


mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


FOLD = 'result_func_appr_deps'
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DIMS = [5, 10, 50, 100, 200]
FUNCS = ['Alpine', 'Chung', 'Dixon']


def load(func):
    e_ht, e_ht_min, e_ht_max = [], [], []
    e_bs, e_bs_min, e_bs_max = [], [], []
    for d in DIMS:
        fpath = os.path.join(FOLD, f'Func{func}', f'dim{d}', 'result.npz')
        result = np.load(fpath, allow_pickle=True).get('result').item()
        _e_ht = [result[seed][f'Func{func}']['htbb']['e'] for seed in SEEDS]
        _e_bs = [result[seed][f'Func{func}']['tt-cross']['e'] for seed in SEEDS]
        e_ht.append(np.mean(_e_ht))
        e_bs.append(np.mean(_e_bs))
        e_ht_min.append(np.min(_e_ht))
        e_bs_min.append(np.min(_e_bs))
        e_ht_max.append(np.max(_e_ht))
        e_bs_max.append(np.max(_e_bs))
    return e_ht, e_ht_min, e_ht_max, e_bs, e_bs_min, e_bs_max


def plot(func, e_ht, e_ht_min, e_ht_max, e_bs, e_bs_min, e_bs_max):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.)

    ax.set_title(f'{func} function')
    ax.set_xlabel('Dimension')
    # ax.set_ylabel('Approximation relative error')

    ax.plot(DIMS, e_ht, label='htbb', color='#8b1d1d',
        marker='o', markersize=8, linestyle='-', linewidth=3)
    ax.fill_between(DIMS, e_ht_min, e_ht_max, alpha=0.4, color='#8b1d1d')

    ax.plot(DIMS, e_bs, label='ttcross', color='#000099',
        marker='s', markersize=8, linestyle='-', linewidth=1)
    ax.fill_between(DIMS, e_bs_min, e_bs_max, alpha=0.4, color='#000099')

    _prep_ax(ax, xlog=True, ylog=True, leg=True)

    fpath = os.path.join(FOLD, f'Func{func}.png')
    plt.savefig(fpath, bbox_inches='tight')
    plt.close(fig)


def _prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()
        #ax.set_yscale('symlog')

    if leg:
        ax.legend(loc='upper left', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)


if __name__ == '__main__':
    for func in FUNCS:
        e_ht, e_ht_min, e_ht_max, e_bs, e_bs_min, e_bs_max = load(func)
        plot(func, e_ht, e_ht_min, e_ht_max, e_bs, e_bs_min, e_bs_max)
