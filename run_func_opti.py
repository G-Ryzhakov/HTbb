import argparse
from copy import deepcopy as copy
from teneva_bm import *
from teneva_opti import *
from opti_tens_htbb import OptiTensHtbb


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
BMS_ARGS = {'d': 256, 'n': 8}


SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


OPTI_NAME_OUR = 'htbb'
OPTIS_BASELINE = [OptiTensTtopt, OptiTensOpo, OptiTensSpsa, OptiTensPso]
OPTI_NAMES_BASELINE = ['ttopt', 'opo', 'spsa', 'pso']
WITH_BASELINES = True


OPTI_ARGS = {'m': 1.E+4, 'seed': 0, 'with_cache': False}
OPTI_OPTS_OUR = {}


def run(fold, seed, with_calc, with_show):
    if with_calc:
        tasks = _tasks_build(seed_only=seed)
        oman = OptiManager(tasks, fold=fold)
        oman.run()

    if with_show:
        oman = OptiManager(fold=fold, load=True)

        for i in range(2):
            for Bm in BMS:
                bm_name = Bm.__name__[2:]

                oman.reset()
                oman.filter_by_bm(arg='name', value=bm_name)
                oman.join_by_op_seed()
                oman.sort_by_op(values=[OPTI_NAME_OUR] + OPTI_NAMES_BASELINE)

                if i == 0:
                    prefix = bm_name[4:]
                    postfix = '\\\\ \hline \n'
                    oman.show_table(prefix, postfix, prec=2, kind='mean')

                if i == 1:
                    fpath = f'{fold}/_plots/{bm_name[4:]}'
                    title = bm_name[4:] + ' function'
                    oman.show_plot(fpath, title=title, name_spec=OPTI_NAME_OUR)


def _args_build():
    parser = argparse.ArgumentParser(
        prog='htbb > run_func_opti',
        description='Numerical experiments for optimization of the multidimensional analytical functions with an optimizer based on hierarchical Tucker decomposition for black-boxes (HTBB) and comparison with various baselines.')

    parser.add_argument('--fold',
        type=str,
        help='Path to the folder with results',
        default='result_func_opti')
    parser.add_argument('--seed',
        type=int,
        help='Optional random seed value (if we compute for only one seed)',
        default=None)
    parser.add_argument('--with_calc',
        action='store_true',
        help='Do we perform computations')
    parser.add_argument('--with_no_calc',
        action='store_false',
        dest='with_calc',
        help='Do we skip computations')
    parser.set_defaults(with_calc=True)
    parser.add_argument('--with_show',
        action='store_true',
        help='Do we present full results of computations')
    parser.add_argument('--with_no_show',
        action='store_false',
        dest='with_show',
        help='Do we skip full results of computations')
    parser.set_defaults(with_show=True)

    args = parser.parse_args()
    return (args.fold, args.seed, args.with_calc, args.with_show)


def _tasks_build(seed_only=None):
    tasks = []
    for seed in SEEDS if seed_only is None else [seed_only]:
        for Bm in BMS:
            tasks.append({
                'bm': Bm,
                'bm_opts': {'budget_raise': True},
                'bm_args': copy(BMS_ARGS),
                'opti': OptiTensHtbb,
                'opti_args': {**OPTI_ARGS, 'name': OPTI_NAME_OUR, 'seed': seed},
                'opti_opts': {**OPTI_OPTS_OUR}})
            if WITH_BASELINES:
                for Opti in OPTIS_BASELINE:
                    tasks.append({
                        'bm': Bm,
                        'bm_args': copy(BMS_ARGS),
                        'opti': Opti,
                        'opti_args': {**OPTI_ARGS, 'seed': seed}})
    return tasks


if __name__ == '__main__':
    run(*_args_build())
