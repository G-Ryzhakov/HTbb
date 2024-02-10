import argparse
import numpy as np
import os
import teneva
from teneva_bm import *
from time import perf_counter as tpc


from node_htopt import simple_set_htree
from node_htopt import uniform_walk
from utils import my_cach


FOLD = 'result_func_appr'
NAME = ''
BMS = [
    BmFuncAlpine,
    BmFuncChung,
    BmFuncDixon,
    BmFuncGriewank,
    BmFuncPathological,
    BmFuncPinter,
    BmFuncQing,
    BmFuncRastrigin,
    BmFuncSchaffer,
    BmFuncSchwefel,
    BmFuncSphere,
    BmFuncSquares,
    BmFuncTrigonometric,
    BmFuncWavy
]


SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEEDS_HALF = [6, 7, 8, 9, 10]


def build_cross(func, m, d, n, e=1.E-16, nswp=100, seed=0):
    Y = teneva.rand([n]*d, r=1, seed=seed)
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=1, dr_max=1, log=True)
    Y = teneva.truncate(Y, e)
    return lambda I: teneva.get(Y, I)


def build_our(func, nswp, d, n, r, dr_max, with_balanced=True, seed=0):
    @my_cach(cache_size=1.E+7, cache_max=1.E+9, check_cach=False)
    def func_cache(I):
        return func(np.asanyarray(I, dtype=int))

    tr = simple_set_htree(d, func_cache, r, [n]*d, random_split=True, seed=seed,
        norm_A=False, is_max=True, how_to_switch='max', dr_max=dr_max,
        rank_reduce_tresh=1.E-9, tau=1.01, tau0=1.01,
        balanced_tree=with_balanced)

    try:
        def cb(bm=None, fr_upd=100, info={}):
            return
        uniform_walk(tr, nswp, alpha=1.E-5, callback=cb, callback_freq=100000,
            finalize=True, log=False)
    except BmBudgetOverException as e:
        pass

    return lambda I: tr.get(I)


def check(getter, I_tst, y_tst, m, t, name):
    y_our = getter(I_tst)
    error = np.linalg.norm(y_tst - y_our) / np.linalg.norm(y_tst)
    name = name + ' ' * max(0, 10 - len(name))
    log(f'{name} | Error {error:-8.2e} | Evals {m:-7.1e} | Time {t:-8.2e}')
    return error


def log(text, is_new=False):
    os.makedirs(FOLD, exist_ok=True)
    fpath = os.path.join(FOLD, NAME + '.txt')
    with open(fpath, 'w' if is_new else 'a') as f:
        f.write(text + '\n')
    print(text)


def run(d, n, nswp, r, dr_max, with_balanced, without_bs, m_tst,
        func_only=None, seed_only=None, postfix=None, seed_half=False):
    result = {}
    for seed in (SEEDS_HALF if seed_half else SEEDS):
        if seed_only is not None and seed_only != seed:
            continue
        res = run_one(d, n, nswp, r, dr_max, with_balanced, without_bs, m_tst,
            seed, func_only)
        result[seed] = res
        fpath = os.path.join(FOLD, 'result' + (postfix or '') + '.npz')
        np.savez_compressed(fpath, result=result)


def run_one(d, n, nswp, r, dr_max, with_balanced, without_bs, m_tst, seed=0,
            func_only=None):
    global NAME
    NAME = f'calc__seed_{seed}'
    log('', is_new=True)

    result = {}

    for Bm in BMS:
        bm = Bm(d, n)
        bm.prep()

        if func_only and func_only != bm.name:
            continue

        result[bm.name] = {'htbb': {}, 'tt-cross': {}}

        text = f'd {d}; n {n}; nswp {nswp}; r {r}; dr_max {dr_max}; seed {seed}'
        text = f'\n------------ {bm.name} ({text})'
        log(text)

        info = {'m': 0}

        rand = np.random.default_rng(seed)
        inp_tens_mix = rand.permutation(np.arange(d))

        def func(I):
            if len(I.shape) == 2:
                info['m'] += I.shape[0]
                I = I[:, inp_tens_mix]
            else:
                info['m'] += 1
                I = I[inp_tens_mix]
            return bm.get(I)

        I_tst = teneva.sample_rand([n]*d, m_tst, seed=0)
        y_tst = func(I_tst)

        _t = tpc()
        info['m'] = 0
        getter = build_our(func, nswp, d, n, r, dr_max, with_balanced, seed)
        t = tpc() - _t
        e = check(getter, I_tst, y_tst, info['m'], t, 'htbb')
        result[bm.name]['htbb'] = {'m': info['m'], 'e': e, 't': t}

        if without_bs:
            continue

        _t = tpc()
        m = info['m']
        info['m'] = 0
        getter = build_cross(func, m, d, n, seed=seed)
        t = tpc() - _t
        e = check(getter, I_tst, y_tst, info['m'], t, 'tt-cross')
        result[bm.name]['tt-cross'] = {'m': info['m'], 'e': e, 't': t}

    return result


def show(without_bs=False, seed_half=False):
    global NAME
    NAME = f'show'
    log('', is_new=True)

    seeds = SEEDS_HALF if seed_half else SEEDS

    try:
        fpath = os.path.join(FOLD, 'result.npz')
        result = np.load(fpath, allow_pickle=True).get('result').item()
    except Exception as e:
        result = {}
        for s in seeds:
            fpath = os.path.join(FOLD, f'result_seed{s}.npz')
            result = {
                **result,
                **np.load(fpath, allow_pickle=True).get('result').item()}

    for name in list(result[seeds[0]].keys()):
        log(f'\n------------ {name} (mean result)')

        for method in ['htbb', 'tt-cross']:
            if without_bs and method == 'tt-cross':
                continue

            e = [result[seed][name][method]['e'] for seed in seeds]
            e = np.mean(e)

            m = [result[seed][name][method]['m'] for seed in seeds]
            m = np.mean(m)

            t = [result[seed][name][method]['t'] for seed in seeds]
            t = np.mean(t)

            method = method + ' ' * max(0, 10 - len(method))
            log(f'{method} | Err {e:-8.2e} | Evals {m:-7.1e} | Time {t:-8.2e}')


def _args_build():
    parser = argparse.ArgumentParser(
        prog='htbb > run_func_appr',
        description='Numerical experiments for approximation of the multidimensional analytical functions with the method based on hierarchical Tucker decomposition for black-boxes (HTBB) and comparison with TT-cross method, which is based on the tensor train format.')

    parser.add_argument('--d',
        type=int,
        help='Dimension of the problem',
        default=256)
    parser.add_argument('--n',
        type=int,
        help='Mode size for each dimension',
        default=8)
    parser.add_argument('--nswp',
        type=int,
        help='Number of sweeps for the HT-method',
        default=100)
    parser.add_argument('--r',
        type=int,
        help='Rank for the HT-method',
        default=2)
    parser.add_argument('--dr_max',
        type=int,
        help='Rank increment for the HT-method',
        default=1)
    parser.add_argument('--with_balanced',
        action='store_false',
        help='Do we balance HT-tree (True by default)')
    parser.add_argument('--without_bs',
        action='store_true',
        help='Do we disable baseline-based computations (False by default)')
    parser.add_argument('--m_tst',
        type=float,
        help='Number of test points',
        default=1.E+4)
    parser.add_argument('--fold',
        type=str,
        help='Folder to save the results',
        default='result_func_appr')
    parser.add_argument('--func_only',
        type=str,
        help='Optional function name for computations',
        default=None)
    parser.add_argument('--seed_only',
        type=int,
        help='Optional seed value for computations',
        default=None)
    parser.add_argument('--show',
        action='store_true',
        help='If true, then just show saved results (False by default)')
    parser.add_argument('--postfix',
        type=str,
        help='Optional postfix for file names',
        default=None)
    parser.add_argument('--seed_half',
        action='store_true',
        help='If true, then compute only for half of seeds.')

    args = parser.parse_args()

    global FOLD
    FOLD = args.fold

    return (args.d, args.n, args.nswp, args.r, args.dr_max, args.with_balanced,
        args.without_bs, args.m_tst, args.func_only, args.seed_only,
        args.postfix, args.seed_half), args.show


if __name__ == '__main__':
    args, is_show = _args_build()

    if is_show:
        show(without_bs=args[6], seed_half=args[-1])
    else:
        run(*args)
