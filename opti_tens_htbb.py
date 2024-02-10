import numpy as np
from teneva_bm import BmBudgetOverException
from teneva_opti import OptiTens


from node_htopt import simple_set_htree
from node_htopt import uniform_walk
from utils import my_cach


class OptiTensHtbb(OptiTens):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'htbb')
        super().__init__(*args, **kwargs)
        self.set_desc("""
            HTBB optimizer based on the hierarchical Tucker decomposition.
        """)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'balanced_tree': {
                'desc': 'Do we use full binary tree or smth else',
                'kind': 'bool',
                'dflt': True
            },
            'dr_max': {
                'desc': 'Rank increment for the HT-method',
                'kind': 'int',
                'dflt': 1
            },
            'norm_a': {
                'desc': 'Do we normalize the batch',
                'kind': 'bool',
                'dflt': True
            },
            'quan': {
                'desc': 'Allow quantization of modes',
                'kind': 'bool',
                'dflt': False
            },
            'r': {
                'desc': 'Rank for the HT-method',
                'kind': 'int',
                'dflt': 2
            },
            'random_split': {
                'desc': 'Do we perform random split',
                'kind': 'bool',
                'dflt': True
            },
        }

    def _optimize(self):
        @my_cach(cache_max=self.bm.budget_m * 5, is_max=self.is_max)
        def func(I):
            I = np.array(I, dtype=int)
            return self.target(I)

        tr = simple_set_htree(self.d_inner, func, self.r, self.n_inner,
            random_split=self.random_split, seed=self.seed, norm_A=self.norm_a,
            rank_reduce_tresh=1.E-9, tau=1.01, tau0=1.01, is_max=self.is_max,
            how_to_switch='max', balanced_tree=self.balanced_tree)

        try:
            def cb(bm=None, fr_upd=100, info={}):
                return
            uniform_walk(tr, 1e10, alpha=1.E-5, callback=cb,
                callback_freq=100000, finalize=False, log=False)
        except BmBudgetOverException as e:
            pass
