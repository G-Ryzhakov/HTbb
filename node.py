from fractions import Fraction
from functools import reduce
import itertools
import numpy as np
from opt_einsum import contract
import scipy
import scipy.special
import teneva
import sys


from utils import  (
   FatherKnowsBetter,
   find_next_free_num,
   index_tens_prod,
   matrix_skeleton_sep,
   matrix_skeleton_spec,
   select_r_different,
   skel_two,
   sort_T,
   stoh_argmax_num,
)

import bisect

def max_arr(l):
    if isinstance(l, (list, np.ndarray)):
        return max_arr(max(l))
    else:
        return l

class Node():
    def __init__(self, idx=None, r=None, A=None, shape=None,  level=0,
                 parent=None, *, random_split=False, d=None, d_real=None,
                 use_maxvol=True, how_to_build_cores='mv',
                 how_to_switch='max', norm_A=False, is_max=True,
                 is_min=None, temp=0, num=0, keep_max_while_maxvol=False,
                 balanced_tree=True, unbalanced_mode="TT_R", cursed_together=True,
                 func_type='discrete', func_self=False, r_from_bottom=False,
                 emulate_TT=False, dr_max=1, tau0=1.01, tau=1.01, k0=500,
                        rank_reduce_tresh=1e-10
                 ):
        """
        Note: A is a function to find max or approximate; use_maxvol -- how to
        select indices (MaxVol or just max); how_to_build_cores: 'mv' --
        maxvol, otherwise -- thru S^-1; how_to_switch --  'rand': random
        (without coming to cursed nodes), 'max': using found maximum; 'seq':
        sequential passes
        unbalanced_mode: one of "TT_R", "TT_L", "arbitrary"
        func_type: 'discrete of contunouse
        func_self -- wether to get values from self.get
        emulate_TT -- doesn't work!!
        """
        if idx is None:
            idx = []

        self.idx = idx
        self.L = None
        self.R = None
        self.level = level
        self.direction = 'l' if np.random.randint(2) else 'r'
        self.count_enters = 0
        self.frac_rl_enters = None
        self.parent = parent
        self.counter_left = 0
        self.counter_right = 0
        self.num = num
        # self.cyclic = False # if our argument is cyclic

        if parent is None:

            self._r_raw = r
            self._shape = np.asarray(shape, dtype=int)
            self._use_maxvol = use_maxvol
            self._how_to_build_cores = how_to_build_cores
            self._temp = temp
            self._how_to_switch = how_to_switch
            self._norm_A = norm_A
            self._is_max = is_max if is_min is None else not is_min
            self._keep_max_while_maxvol = keep_max_while_maxvol
            self._cursed_together = cursed_together
            self._balanced_tree = balanced_tree
            self._unbalanced_mode = unbalanced_mode
            if unbalanced_mode == "arbitrary":
                random_split = False

            self._func_type = func_type
            self._func_self = func_self
            self._get_par = self.get
            self._info =  dict(tau=tau,
                               tau0 = tau0, 
                               k0=k0,
                               rank_reduce_tresh=rank_reduce_tresh,
                               dr_max=dr_max,
                               r_from_bottom=r_from_bottom,
                               emulate_TT=emulate_TT
                              )



            if idx is not None:
                if d is None:
                    #self._d = len(idx)
                    self._d = max_arr(max(idx)) + 1
                else:
                    self._d = d

                if d_real is None:
                    self._d_real = self._d
                else:
                    self._d_real = d_real

                if self._d_real < self._d:
                    if len(self._shape) < self._d:
                        self._shape = np.pad(self._shape, (0, self._d - len(self._shape)))
                    self._shape[-(self._d - self._d_real):] = 1

                self.split_children(random=random_split)

            # recursion limit for big dims
            need_rec_lim = (len(bin(self._d_real)) + 30) if balanced_tree is True else (self._d + 30)
            if need_rec_lim > sys.getrecursionlimit():
                sys.setrecursionlimit(need_rec_lim)
                print(f"WARN: setting recursion limit to {need_rec_lim}")

            self._row_A = A

            if self._d != self._d_real:
                def A_real(I):
                    return A(I[:self._d_real])
                self._A = A_real
            else:
                self._A = A

            self.build_raw_ranks()
            while self.check_ranks():
                pass

    def build_raw_ranks(self):
        self.r
        for nn in self.children:
            nn.r

    def check_ranks(self):
        was_changed = False
        if not self.is_root:
            if self.r > self.parent.r*self.siblis.r:
                new_rank = self.parent.r*self.siblis.r
                print(f"for {self} changed rank: {self.r} -> {new_rank}")
                self.r = new_rank
                was_changed = True

        return any([was_changed] + [nn.check_ranks() for nn in self.children])



    def __repr__(self):
        res  = f'H-Node with idx: {self.idx}, num: {self.full_num}'
        if self.is_leaf:
            res += '| leaf'
        if self.is_root:
            res += '| ROOT'
        return res

    @FatherKnowsBetter
    def is_max(self):
        pass

    @FatherKnowsBetter
    def norm_A(self):
        pass

    @FatherKnowsBetter
    def A(self):
        pass

    @FatherKnowsBetter
    def row_A(self):
        pass

    @FatherKnowsBetter
    def keep_max_while_maxvol(self):
        pass

    @FatherKnowsBetter
    def balanced_tree(self):
        pass

    @FatherKnowsBetter
    def unbalanced_mode(self):
        pass

    @FatherKnowsBetter
    def func_type(self):
        pass

    @FatherKnowsBetter
    def func_self(self):
        pass

    @FatherKnowsBetter
    def get_par(self):
        pass

    @FatherKnowsBetter
    def info(self):
        pass

    def set_func_self(self, val):
        if self.is_root:
            self._func_self = val
        else:
            try:
                del self._func_self
            except AttributeError:
                pass


        for nn in self.children:
            nn.set_func_self(val)


    @property
    def G(self):
        try:
            return self._G
        except:
            if self.how_to_build_cores == 'mv':
                self.make_cores_mv()
            else:
                self.make_cores()
            return self._G

    @property
    def S_mat(self):
        try:
            return self._S_mat
        except:
            self.make_mat_S()
            return self._S_mat

    @property
    def children(self):
        res = []
        if self.L is not None:
            res.append(self.L)
        if self.R is not None:
            res.append(self.R)
        return res

    @property
    def all_children(self):
        res = []
        for nn in self.children:
            res.append(nn)

        for nn in self.children:
            res.extend(nn.all_children)

        res.sort(key=lambda x: x.level)
        return res

    @property
    def all_children_iter(self):
        for nn in self.children:
            yield nn
            for c in nn.all_children_iter:
                yield c

    @property
    def count_enters_with_childrens(self):
        if self.cursed:
            return 0

        res = self.count_enters
        for nn in self.children:
            res += nn.count_enters_with_childrens

        return res

    @property
    def count_enters_of_childrens(self):
        return self.count_enters_with_childrens - self.count_enters


    @property
    def number_of_all_childrens(self):
        if self.cursed:
            return 0

        res = 1
        for nn in self.children:
            res += nn.number_of_all_childrens

        return res


    @property
    def full_num(self):
        return self.level, self.num

    @property
    def convolv(self):
        try:
            return self._convolv
        except:
            if self.is_leaf:
                self._convolv = np.einsum('jk->j', self.G)

                self._convolv = np.einsum('ijk,i,k->j',
                    self.G, self.L.convolv, self.R.convolv)
            return self._convolv

    @property
    def cursed(self):
        try:
            return self._cursed
        except  AttributeError:
            if self.is_leaf:
                self._cursed = self.idx[0] >= self.d_real
            else:
                self._cursed = self.L.cursed and self.R.cursed
            return self._cursed

    @FatherKnowsBetter
    def d(self):
        pass

    @FatherKnowsBetter
    def d_real(self):
        pass

    @property
    def down_idx(self):
        try:
            return self._down_idx
        except:
            if self.is_root:
                self._down_idx = [] # np.arange(self.d)
            else:
                #self._down_idx = np.setdiff1d(np.arange(self.d), self.up_idx)
                self._down_idx = list(self.parent.down_idx)
                self._down_idx = self._down_idx + list(self.siblis.up_idx)
            return self._down_idx

    @property
    def down_idx_vals(self):
        """
        Note: this function is only for initialization.
        TODO: add initialization from h-tensor (as in ttpy).
        """
        if self.parent.is_root: # well, it's not a hack
            return self.siblis.up_idx_vals

        try:
            return self._down_idx_vals
        except:
            r = self.r
            if self.is_root:
                self._down_idx_vals = [[]]
            else:
                arng = index_tens_prod(self.parent.down_idx_vals, self.siblis.up_idx_vals)
                #arng = [list(a) + list(b) for a, b in itertools.product(
                #    self.parent.down_idx_vals, self.siblis.up_idx_vals)]
                self._down_idx_vals = select_r_different(arng, r)

            return self._down_idx_vals

    @down_idx_vals.setter
    def down_idx_vals(self, val):
        self._down_idx_vals = val

    @FatherKnowsBetter
    def how_to_build_cores(self):
        pass

    @FatherKnowsBetter
    def how_to_switch(self):
        pass

    @property
    def is_leaf(self):
        return self.R is None and self.L is None

    @property
    def is_root(self):
        return self.parent is None

    @property
    def max_level(self):
        return max([i.max_level for i in self.children] + [self.level])

    @property
    def popularity(self):
        if self.is_leaf:
            res = np.zeros(2)
            res[0 + (self.idx[0] >= self.d_real)] += 1
            return res
        else:
            return self.L.popularity + self.R.popularity

    @property
    def popularity_with_max(self):
        if self.cursed:
            return 1e10

        res = 0

        for nn in self.children:
            res = max(res, nn.popularity_with_max)

        try:
            res = max(res, abs(self._up_max_A))
        except  AttributeError:
            pass

        return res


    @property
    def r(self):
        if self.cursed:
            return 1
        try:
            return self.my_r
        except:
            pass
        r = self.r_raw
        if isinstance(r, int):
            return r
        else:
            r_from_bottom = self.info['r_from_bottom']
            if r_from_bottom:
                cur_r = r[self.level_from_bottom]
                if not self.is_root and self.parent.is_root:
                    cur_r = min(cur_r, r[self.siblis.level_from_bottom])
                self.my_r = cur_r
                return cur_r
            else:
                self.my_r = r[self.level]
                return r[self.level]

    @r.setter
    def r(self, rank):
        self._r_raw = rank
        try:
            del self.my_r
        except:
            pass
        for nn in self.children:
            try:
                del nn.my_r
            except:
                pass
            nn.r_raw = rank

    @FatherKnowsBetter
    def r_raw(self):
        pass

    @FatherKnowsBetter
    def shape(self):
        pass

    @property
    def siblis(self):
        return self.parent.R if self.parent.L is self else self.parent.L

    @FatherKnowsBetter
    def temp(self):
        pass


    @property
    def level_from_bottom(self):
        if self.is_leaf:
            return 0
        return min([nn.level_from_bottom for nn in self.children]) + 1

    @property
    def tp(self):
        try:
            return self._tp
        except:
            return []

    @property
    def up_det(self):
        if self.is_leaf:
            if self.cursed:
                return 0
            else:
                return self._up_det
        else:
            return self.R.up_det + self.L.up_det

    @property
    def up_idx(self):
        try:
            return self._up_idx
        except:
            if self.is_leaf:
                self._up_idx = list(self.idx)
            else:
                self._up_idx = self.L.up_idx + self.R.up_idx

            return self._up_idx

    @property
    def up_idx_vals(self):
        """
        Note: this function is only for initialization.
        TODO: add initialization from h-tensor (as in ttpy).
        """
        try:
            return self._up_idx_vals
        except:
            r = self.r

            if self.is_leaf:
                arng = np.arange(self.shape[self.idx[0]])[:, None]
                self._up_idx_vals = select_r_different(arng, r)

            elif not self.is_root:
                arng = index_tens_prod(self.L.up_idx_vals, self.R.up_idx_vals)
                #arng = [list(a) + list(b) for a, b in itertools.product(
                #    self.L.up_idx_vals, self.R.up_idx_vals)]

                self._up_idx_vals = select_r_different(arng, r)

            else:
                self._up_idx_vals = []

            return self._up_idx_vals

    @up_idx_vals.setter
    def up_idx_vals(self, val):
        self._up_idx_vals = val


    def children_up_idx_list(self):

        if self.is_leaf:
            for i in self.up_idx_vals:
                l = self.ch_up_idx_list = []
                i = i[0]
                found, pos = where_bisect(self.full_range_idx, i)
                if not found:
                    raise ValueError(f"Bad index combs: list={self.full_range_idx}, value={i} from {self.up_idx_vals}. I am {self}")

                l.append(pos)

        elif not self.is_root:

            pass

        for nn in self.children:
            nn.children_up_idx_list()


    @FatherKnowsBetter
    def use_maxvol(self):
        pass

    def add_nx_edge(tr, G, name='R', use_only_idx=False):
        if tr.L is not None:
            nname = f"{name}L"

            if use_only_idx is False:
                try:
                    sh = tr.G.shape[0]
                    sh_txt = str(sh)
                except AttributeError:
                    use_only_idx = True

            if use_only_idx:
                sh = len(tr.L.up_idx_vals)
                sh_txt = str(sh)
                sh2 = len(tr.L.down_idx_vals)
                if sh != sh2:
                    sh_txt += f" ({sh2})"
            G.add_edge(f"{name}", nname, weight=sh_txt)
            #G.edges[f"{name}", nname]['minlen'] = '300'
            tr.L.add_nx_edge(G, name=nname, use_only_idx=use_only_idx)
        else:
            sh = tr.shape[tr.idx[0]]
            G.add_edge(name, f"{tr.idx[0]}", weight=str(sh))
            #G.edges[name, f"{tr.idx[0]}"]['minlen'] = '100'

        if tr.R is not None:
            if use_only_idx is False:
                try:
                    sh = tr.G.shape[-1]
                    sh_txt = str(sh)
                except AttributeError:
                    use_only_idx = True

            if use_only_idx:
                sh = len(tr.R.up_idx_vals)
                sh_txt = str(sh)
                sh2 = len(tr.R.down_idx_vals)
                if sh != sh2:
                    sh_txt += f" ({sh2})"
            nname = f"{name}R"
            G.add_edge(f"{name}", nname, weight=sh_txt)
            #G.edges[f"{name}", nname]['minlen'] = '300'
            tr.R.add_nx_edge(G, name=nname, use_only_idx=use_only_idx)
        else:
            sh = tr.shape[tr.idx[0]]
            G.add_edge(name, f"{tr.idx[0]}", weight=str(sh))
            #G.edges[name, f"{tr.idx[0]}"]['minlen'] = '100'

        return G

    def all_leaves(self):
        if self.is_leaf:
            return [self]
        else:
            return self.L.all_leaves() + self.R.all_leaves()

    def contract_with_TT(self, cores, ret_val=True):
        def leaf_func(self, cores, num_up, big_num, res):
            cur_core = cores[self.idx[0]]
            res.append([
                cur_core,
                [2*big_num + self.idx[0], num_up, 2*big_num + self.idx[0] + 1]
            ])

        res = self.univeral_contract(cores, leaf_func)
        if ret_val:
            res = contract(*res, [])
        return res

    def contract_with_idx(self, idx, ret_val=False):
        def leaf_func(self, idx, num_up, big_num, res):
            nums, I = idx
            nums = list(nums)
            if self.idx[0] in nums:
                pos = nums.index(self.idx[0])
                res.extend([self.G[:, I[:, pos]], [num_up, 0]])
            else: # free var
                res.extend([self.G, [num_up, self.idx[0] + 1]])

        res = self.univeral_contract(idx, leaf_func)
        if ret_val:
            res = contract(*res)
        return res

    def full(self, res=None, num_up=0):
        need_eins = False
        if res is None:
            res = [np.array([1]), [0]]
            need_eins = True

        if self.is_root:
            res.extend([self.G, [1, 0, 2]])
            self.L.full(res, 1)
            self.R.full(res, 2)
        elif self.is_leaf:
            next_num = find_next_free_num(res)
            res.extend([self.G, [num_up, next_num]])
        else:
            next_num = find_next_free_num(res)
            res.extend([self.G, [next_num, num_up, next_num + 1]])
            self.L.full(res, next_num)
            self.R.full(res, next_num + 1)

        if need_eins:
            self._contract_rules = res
            return contract(*res)

    def get(self, I):
        if self.is_root:
            I = np.asanyarray(I)
            ndim = I.ndim
            if ndim == 1:
                I = I[None, :]

        if self.is_leaf:
            if self.cursed:
                return np.ones([1, I.shape[0]])
            else:
                return self.G[:, I[:, self.idx[0]]]

        y = np.einsum('rsq,rk,qk->sk',
            self.G, self.L.get(I), self.R.get(I))

        return (y[0] if ndim > 1 else y[0][0]) if self.is_root else y

    def is_offspring(self, node):
        for nn in self.children:
            if nn is node:
                return True

        for nn in self.children:
            if nn.is_offspring(node):
                return True

        return False



    def leafs_of_level(self, lev):
        if self.level == lev:
            return [self]
        else:
            res = [nn.leafs_of_level(lev) for nn in self.children]
            if res:
                res = reduce(lambda x, y: x + y, res)
            return res

    def make_cores(self):
        if self.cursed:
            self._G = np.array([[1]])
            if not self.is_leaf:
                self._G = np.array([[[1]]])
            return

        if not self.is_root:
            if self.is_leaf:
                bt, btp = self.up_idx_vals, self.down_idx_vals
                B = np.empty([self.shape[self.idx[0]], len(btp)])
                I = np.empty(self.d, dtype=int)
                for i in range(B.shape[0]):
                    for j, val in enumerate(btp):
                        I[self.idx[0]] = i
                        I[self.down_idx] = val
                        B[i, j] = self.A(I)

                self._G = B.T

            else:
                # Not root and not leaf
                S1, S2 = self.L.S_mat, self.R.S_mat
                bt, btp = self.up_idx_vals, self.down_idx_vals

                Pt1 = self.L.idx
                Pt2 = self.R.idx
                Ptp = self.down_idx

                A_cur = np.empty([len(self.L.up_idx_vals), len(self.R.up_idx_vals), len(btp)])
                I = np.empty(self.d, dtype=int)

                for ind1, iPt1_val in enumerate(self.L.up_idx_vals):
                    for ind2, iPt2_val in enumerate(self.R.up_idx_vals):
                        for ind3, iPtp_val in enumerate(btp):
                            I[Pt1] = iPt1_val
                            I[Pt2] = iPt2_val
                            I[Ptp] = iPtp_val
                            A_cur[ind1, ind2, ind3] = self.A(I)

                make_stable = True
                if make_stable:
                    # Let's make it more stable
                    self._G = stable_double_inv(A_cur, S1, S2)
                else:
                    self._G = np.einsum("rlk,ir,jl->ikj", A_cur, np.linalg.inv(S1), np.linalg.inv(S2))

        else:
            # Root node
            S1, S2 = self.L.S_mat, self.R.S_mat

            Pt1 = self.L.idx
            Pt2 = self.R.idx
            A_cur = np.empty(
                [len(self.L.up_idx_vals), len(self.R.up_idx_vals)])
            I = np.empty(self.d, dtype=int)
            for ind1, iPt1_val in enumerate(self.L.up_idx_vals):
                for ind2, iPt2_val in enumerate(self.R.up_idx_vals):
                        I[Pt1] = iPt1_val
                        I[Pt2] = iPt2_val
                        A_cur[ind1, ind2] = self.A(I)

            make_stable = True
            if make_stable:
                self._G = stable_double_inv(A_cur[..., None], S1, S2)
            else:
                self._G = np.einsum('rl,ir,jl->ij',
                    A_cur, np.linalg.inv(S1), np.linalg.inv(S2))[:, None, :]

    def make_cores_mv(self):
        if self.cursed:
            self._G = np.array([[1]])
            if not self.is_leaf:
                self._G = self._G[None]
            return

        if self.is_root:
            Pt1 = self.L.idx
            Pt2 = self.R.idx
            A_cur = np.empty(
                [len(self.L.up_idx_vals), len(self.R.up_idx_vals)])
            I = np.empty(self.d, dtype=int)

            for ind1, iPt1_val in enumerate(self.L.up_idx_vals):
                for ind2,  iPt2_val in enumerate(self.R.up_idx_vals):
                        I[Pt1] = iPt1_val
                        I[Pt2] = iPt2_val
                        A_cur[ind1, ind2] = self.A(I)

            self._G = A_cur[:, None, :]

        else: # Not root
            if self.is_leaf:
                self._G = self._B.T

            else:
                self._G = np.reshape(self._B, [len(self.L.up_idx_vals),len(self.R.up_idx_vals), -1])
                self._G = np.transpose(self._G, [0, 2, 1])
                # TODO: add check that indices are consisten,
                # i.e. maxvol works not many time ago
                #print(self.L.up_idx_vals, self.R.up_idx_vals, self.up_idx_vals)


    def make_mat_S(self):
        #for nn in [self.L, self.R]:
        #    if nn is None:
        #        continue
        #    nn.make_mat_S()

        if self.is_root:
            return

        if self.cursed:
            self._S_mat = np.ones([[1]])
            return

        t = self.up_idx
        tp = self.down_idx
        d = max(max(t), max(tp)) + 1
        idx = np.empty(d, dtype=int)

        bt, btp = self.up_idx_vals, self.down_idx_vals

        res = self._S_mat = np.empty([len(bt), len(btp)])
        for jt, ti in enumerate(bt):
            for jtp, tpi in enumerate(btp):
                idx[t] = ti
                idx[tp] = tpi
                res[jt, jtp] = self.A(idx)

    def orthoginolize_up(self):
        try:
            del self._convolv
        except:
            pass

        if self.is_leaf:
            other, self._G = np.linalg.qr(self.G)
            return other

        elif self.is_root:
            self._G = contract('inj,ik,jl->knl',
                        self.G,
                        self.L.orthoginolize_up(),
                        self.R.orthoginolize_up())

        else: # Intermediate core
            G = contract('inj,ik,jl->nkl',
                        self.G,
                        self.L.orthoginolize_up(),
                        self.R.orthoginolize_up())

            n, r1, r2 = G.shape
            other, G = np.linalg.qr(G.reshape(n, -1))

            self._G = np.transpose(G.reshape(-1, r1, r2), [1, 0, 2])

            return other

    def orthoginolize_up_TTlike(self, sigma_ratio=1e-6, rmax=10000,
                                use_int_rank=False, rel=True):
        try:
            del self._convolv
        except:
            pass

        if use_int_rank:
            rmax = self.r

        if self.is_leaf:
            other, self._G = teneva.matrix_skeleton(self.G, e=0, r=int(1e8),
                give_to='l')
            return other

        elif self.is_root:
            self._G = contract('inj,ik,jl->knl',
                self.G,
                self.L.orthoginolize_up_TTlike(sigma_ratio=sigma_ratio,
                    rmax=rmax, use_int_rank=use_int_rank),
                self.R.orthoginolize_up_TTlike(sigma_ratio=sigma_ratio,
                    rmax=rmax, use_int_rank=use_int_rank))

        else: # Intermediate core
            G = contract('inj,ik,jl->nkl',
                self.G,
                self.L.orthoginolize_up_TTlike(sigma_ratio=sigma_ratio,
                    rmax=rmax, use_int_rank=use_int_rank),
                self.R.orthoginolize_up_TTlike(sigma_ratio=sigma_ratio,
                    rmax=rmax, use_int_rank=use_int_rank))

            n, r1, r2 = G.shape
            other, G = teneva.matrix_skeleton(G.reshape(n, -1),
                e=0, r=int(1e8), give_to='l')

            self._G = np.transpose(G.reshape(-1, r1, r2), [1, 0, 2])

            return other

    def plot(self, use_only_idx=False):
        import matplotlib.pyplot as plt
        import networkx as nx
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout

        G = self.add_nx_edge(nx.Graph(), use_only_idx=use_only_idx)

        # Nodes
        plt.figure(figsize=(16, 16))

        pos = graphviz_layout(G, prog="dot")
        nx.draw_networkx_edges(G, pos)

        nodes_all_almost = [n for n in list(G) if n.startswith('R') or n == 'T']
        nodes_other = [n for n in list(G) if not n.startswith('R')]

        nx.draw_networkx_nodes(G, pos,  nodelist=nodes_all_almost,
            node_size=300)
        nx.draw_networkx_nodes(G, pos,  nodelist=nodes_other,
            node_size=400, node_shape='s', node_color="#FF78b4")

        labels = {n: n for n in G if not n.startswith('R') or len(n) == 1}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=11,
            font_family='sans-serif');

        # Edge weight labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

    def print(self, sp_len=0, p_idx=True, p_pop=True, p_rank=False):
        """
        Note: add the following code to best use in jupyter:
        ```
        from IPython.display import display, HTML
        display(HTML("<style>.container { width:70% !important; }</style>"))
        display(HTML("<style>.output_result { max-width:100% !important; }</style>"))

        ```
        """
        idx_str = str(self.idx)
        sp_len_ch = sp_len + len(idx_str) + 1
        if self.L is not None:
            self.L.print(sp_len_ch, p_idx=p_idx, p_rank=p_rank)

        add_str = ''
        if p_idx:
            try:
                add_str = f" (UP: {self.up_idx}, {list(self.up_idx_vals)}; DOWN: {self.down_idx}, {list(self.down_idx_vals)})"
            except AttributeError:
                pass

        if p_pop:
            add_str += f" (pop: {self.popularity})"

        if p_rank:
            add_str += f" (real rank: {self.G.shape})"


        print(f"{' ' * sp_len}-> {idx_str}{add_str}")
        if self.R is not None:
            self.R.print(sp_len_ch, p_idx=p_idx, p_rank=p_rank)


    def sample(self, up_mat=None, verb=False,
               sample_func=lambda p: np.random.choice(len(p), p=p/p.sum()),
               trans_func=np.abs):
        """
        Note: trans_func = np.abs -- usual sampling, x**2 -- from square

        """
        if up_mat is None: # For root node
            up_mat = np.ones(self.G.shape[1])

        if self.is_leaf:
            p = contract('ij,i->j', self.G, up_mat)
            if verb:
                print(f"leaf: {p}", end=",")
            p = trans_func(p)
            p /= p.sum()
            if verb:
                print(f"normed: {p}", end=",")

            idx = sample_func(p)
            if verb:
                print(f"idx: {idx}")

            return [idx]

        A = contract('ijk,j,i,k->ik',
            self.G, up_mat, self.L.convolv, self.R.convolv)

        A = trans_func(A)
        idx_L, idx_R = np.unravel_index(sample_func(A.reshape(-1)), A.shape)

        if False:
            l2r = False
            if l2r:
                p_L = np.sum(A, axis=1)
                p_L /= p_L.sum()

                #idx_L = np.random.choice(len(p_L), p=p_L)
                idx_L = sample_func(p_L)

                p_R = trans_func(A[idx_L])
                p_R /= p_R.sum()

                #idx_R = np.random.choice(len(p_R), p=p_R)
                idx_R = sample_func(p_R)

            else:
                p_R = np.sum(A, axis=0)
                p_R /= p_R.sum()

                idx_R = sample_func(p_R)

                p_L = trans_func(A[:, idx_R])
                p_L /= p_L.sum()

                idx_L = sample_func(p_L)

        e_L = np.zeros(A.shape[0])
        e_L[idx_L] = 1

        e_R = np.zeros(A.shape[1])
        e_R[idx_R] = 1

        if verb:
            print(f"level {self.level}: {idx_L}, {idx_R}")

        idx1 = self.L.sample(e_L, verb=verb, sample_func=sample_func, trans_func=trans_func)
        idx2 = self.R.sample(e_R, verb=verb, sample_func=sample_func, trans_func=trans_func)

        return idx1 + idx2

    def sample_max(self, verb=False, num=3, square=False):
        tf = (lambda x: x**2) if square else (lambda x: x)
        sf = lambda a : stoh_argmax_num(a, n=num)
        return self.sample(verb=verb, sample_func=sf, trans_func=tf)

    def sample_max_old(self, up_mat=None):
        """
        Please, use sample with sample_func = max.
        """
        if up_mat is None: # For root node
            up_mat = np.ones(self.G.shape[1])

        if self.is_leaf:
            p = contract('ij,i->j', self.G, up_mat)
            p = np.abs(p)

            idx = np.argsort(p)[-1]

            return [idx]

        A = contract('ijk,j,i,k->ik',
            self.G, up_mat, self.L.convolv, self.R.convolv)
        U, V = matrix_skeleton_spec(A, e=1.E-8)

        p = contract('ir,rj->r', U, V)
        p = np.abs(p)

        idx = np.argsort(p)[-1]

        idx1 = self.L.sample_max(U[:, idx])
        idx2 = self.R.sample_max(V[idx, :])

        return idx1 + idx2

    def sample_old(self, up_mat=None):
        if up_mat is None: # For root node
            up_mat = np.ones(self.G.shape[1])

        if self.is_leaf:
            p = contract('ij,i->j', self.G, up_mat)
            p = np.abs(p)
            p /= p.sum()

            idx = np.random.choice(len(p), p=p)

            return [idx]

        A = contract('ijk,j,i,k->ik',
            self.G, up_mat, self.L.convolv, self.R.convolv)
        U, V = matrix_skeleton_spec(A, e=1.E-8)

        p = contract('ir,rj->r', U, V)
        p = np.abs(p)
        p /= p.sum()

        idx = np.random.choice(len(p), p=p)
        idx1 = self.L.sample(U[:, idx])
        idx2 = self.R.sample(V[idx, :])

        return idx1 + idx2

    def sample_square(self, verb=False,
                      sample_func=lambda p: np.random.choice(len(p),p=p)):
        return self.sample(
            verb=verb, sample_func=sample_func, trans_func=lambda x: x**2)

    def sample_square_old(self, up_mat=None, verb=False,
                          sample_func=lambda p: np.random.choice(len(p),p=p)):
        """
        Do not forget to othogonolize up first;
        TODO -- join with sample.

        """
        if up_mat is None: # For root node
            up_mat = np.ones(self.G.shape[1])

        if self.is_leaf:
            p = contract('ij,i->j', self.G, up_mat)
            if verb:
                print(f"leaf: {p}", end=",")
            p = p*p
            p /= p.sum()
            if verb:
                print(f"normed: {p}", end=",")

            #idx = np.random.choice(len(p), p=p)
            idx = sample_func(p)
            if verb:
                print(f"idx: {idx}")

            return [idx]

        A = contract('ijk,j->ik',
            self.G, up_mat)

        p_L = np.sum(A, axis=1)**2
        p_L /= p_L.sum()

        #idx_L = np.random.choice(len(p_L), p=p_L)
        idx_L = sample_func(p_L)

        p_R = A[idx_L]**2
        p_R /= p_R.sum()

        #idx_R = np.random.choice(len(p_R), p=p_R)
        idx_R = sample_func(p_R)

        e_L = np.zeros(A.shape[0])
        e_L[idx_L] = 1

        e_R = np.zeros(A.shape[1])
        e_R[idx_R] = 1

        if verb:
            print(f"level {self.level}: {p_L}, {p_R}, {idx_L}, {idx_R}")

        idx1 = self.L.sample_square(e_L, verb=verb, sample_func=sample_func)
        idx2 = self.R.sample_square(e_R, verb=verb, sample_func=sample_func)

        return idx1 + idx2

    def set_use_maxvol(self, use_maxvol):
        self._use_maxvol = use_maxvol
        for nn in self.children:
            nn.set_use_maxvol(use_maxvol)

    def split_children(self, random=False):
        try:
            n = len(self.idx)
        except TypeError:
            n = 1
            self.idx = [int(self.idx)]

        if n > 1:
            idx = self.idx
            if random and self.is_root:
                if self._cursed_together:
                    idx = np.concatenate((np.random.permutation(self.idx[:self._d_real]), self.idx[self._d_real:]))
                else:
                    idx = np.random.permutation(self.idx)

            to_rebuild_idx = False
            if self.balanced_tree:
                idx_L = idx[:n//2]
                idx_R = idx[n//2:]
            else:
                if self.unbalanced_mode == "TT_R":
                    idx_L = idx[:1]
                    idx_R = idx[1:]
                elif self.unbalanced_mode == "TT_L":
                    idx_L = idx[:-1]
                    idx_R = idx[-1:]
                elif self.unbalanced_mode == "arbitrary":
                    idx_L, idx_R = idx
                    to_rebuild_idx = True
                else:
                    raise ValueError(f'Unknown unbalanced_mode: "{self.unbalanced_mode}"')


            self.L = Node(idx_L, level=self.level + 1, parent=self, num=2*self.num)
            self.R = Node(idx_R, level=self.level + 1, parent=self, num=2*self.num + 1)

            self.L._tp = self.R.idx
            self.R._tp = self.L.idx

            self.L.split_children()
            self.R.split_children()

            if to_rebuild_idx:
                self.idx = self.L.idx + self.R.idx

        else: # n == 1
            self.idx = [int(self.idx[0])]
            self.cyclic = False # if our argument is cyclic. Make sense only for leaf.



    def switch_way(self):
        if self.is_leaf:
            return None

        popL = self.L.popularity[0]
        popR = self.R.popularity[0]
        assert (popL + popR) > 0, "We should not be here"

        return self.R if np.random.rand()*(popL + popR) > popL else self.L

    def switch_way_by_max(self):
        if self.is_leaf:
            return None

        popL = self.L.popularity_with_max
        popR = self.R.popularity_with_max
        prob = scipy.special.softmax([-popL, -popR])[0]

        return self.R if np.random.rand() > prob else self.L

    def switch_way_counter(self):
        l = self.counter_left
        r = self.counter_right
        print(l, r)
        go_to_left = np.random.rand() > np.exp(l) / (np.exp(l) + np.exp(r))
        if go_to_left:
            self.counter_left += 1
        else:
            self.counter_right += 1
        return self.L if go_to_left else self.R

    def switch_way_simple_with_pop(self):
        cnt = self.count_enters
        self.count_enters += 1

        if self.is_leaf:
            return None

        fr = self.frac_rl_enters

        if cnt == 0:
            assert not (self.L.cursed and self.R.cursed), "never get here!"
            if self.L.cursed:
                self.direction = 'r'

            if self.R.cursed:
                self.direction = 'l'

            res = dict(r=self.R, l=self.L)[self.direction]
            self.frac_rl_enters = dict(r=Fraction(1, 1), l=Fraction(0, 1))[self.direction]
        else:
            popL = self.L.popularity[0]
            popR = self.R.popularity[0]
            pop_sum = popL + popR
            assert pop_sum > 0, "We shoudnt be here"
            # print(popR, pop_sum)
            need_frac = Fraction(int(popR), int(pop_sum))
            if fr > need_frac:
                res = self.L
                cur_d = 'l'
            elif  fr < need_frac:
                res = self.R
                cur_d = 'r'
            else:
                # cur_d = 'l' if np.random.randint(2) else 'r'
                cur_d = self.direction
                res = dict(r=self.R, l=self.L)[cur_d]

            if cur_d == 'r':
                self.frac_rl_enters = (fr*cnt + 1)/(cnt + 1)
                res = self.R
            else:
                self.frac_rl_enters = (fr*cnt)/(cnt + 1)
                res = self.L

        return res


    def switch_way_simple(self):
        if self.is_leaf:
            return None

        if self.direction == 'l':
            self.direction = 'r'
            res = self.L

        else:
            self.direction = 'l'
            res = self.R

        return res


    def test_func(tr, eps=1e-7, details=True):
        ssum = 0
        for idx in itertools.product(*[range(i) for i in list(tr.shape)]):
            idx = list(idx)
            delta = tr.get(idx) - tr.A(idx)
            if details:
                if abs(delta) > eps:
                    print(idx, tr.get(idx), tr.A(idx))

            ssum += delta**2

        return np.sqrt(ssum)

    def topk(self, k=1, seq=None):
        if seq is None:
            seq = range(self.d_real)

        seq = np.asarray(seq, dtype=int)

        # First iter
        contract_rules = self.contract_with_idx([[], None])
        contract_rules.append([seq[0] + 1])
        y1 = contract(*contract_rules)

        if len(y1) > k:
            idx_sort = np.argsort(y1)[::-1][:k]
        else:
            idx_sort = np.arange(len(y1))

        I = idx_sort[:, None]

        for i, sv in enumerate(seq[1:], start=1):
            idx = [seq[:i], I]
            contract_rules = self.contract_with_idx(idx)
            contract_rules.append([0, sv + 1])

            y = contract(*contract_rules).reshape(-1)

            add_idx = range(self.shape[sv])
            I = np.array([list(a) + [b] for a, b in itertools.product(
                    I, add_idx)])

            I = I[np.argsort(y)[::-1][:k], :]

        return I

    def truncate(self, sigma_ratio=1e-6, rmax=10000, use_int_rank=False,
                 rel=True):
        assert self.is_root

        self.orthoginolize_up_TTlike()
        self.truncate_down(sigma_ratio=sigma_ratio, rmax=rmax,
            use_int_rank=use_int_rank, rel=rel)

    def truncate_down(self, U=None, sigma_ratio=1e-6, rmax=10000,
                      use_int_rank=False, rel=True):
        try:
            del self._convolv
        except:
            pass

        # assert not self.is_root, "Cannot call this func directly"
        if self.is_root:
            self._G, U1, U2 = matrix_skeleton_sep(self._G[:, 0, :],
                e=sigma_ratio, r=rmax, rel=True)
            self._G = self._G[:, None, :]

            self.L.truncate_down(U1.T, sigma_ratio=sigma_ratio, rmax=rmax,
                use_int_rank=use_int_rank, rel=rel)
            self.R.truncate_down(U2,   sigma_ratio=sigma_ratio, rmax=rmax,
                use_int_rank=use_int_rank, rel=rel)

        elif self.is_leaf:
            self._G =  U @ self.G

        else: # Intermediate core
            G = contract("ki,mij->mkj", U, self.G)
            self._G, U1, U2 = skel_two(G, e=sigma_ratio, rmax=rmax, rel=rel)

            self.L.truncate_down(U1.T, sigma_ratio=sigma_ratio, rmax=rmax,
                use_int_rank=use_int_rank)
            self.R.truncate_down(U2,   sigma_ratio=sigma_ratio, rmax=rmax,
                use_int_rank=use_int_rank)

    def univeral_contract(self, idx, leaf_func, res=None, num_up=0):
        """
        Return eins rules for summation with fixed indices in idx, summing
        over all others can be used as get if all indices are fixed.
        """
        big_num = 1000

        need_eins = False
        if res is None:
            res = [np.array([1]), [big_num + 0]]
            need_eins = True

        if self.is_root:
            res.extend([self.G, [big_num + 1, big_num + 0, big_num + 2]])
            self.L.univeral_contract(idx, leaf_func, res, num_up=big_num + 1)
            self.R.univeral_contract(idx, leaf_func, res, num_up=big_num + 2)

        elif self.is_leaf:
            leaf_func(self, idx, num_up, big_num,  res)

        else:
            next_num = find_next_free_num(res)
            res.extend([self.G, [next_num, num_up, next_num + 1]])
            self.L.univeral_contract(idx, leaf_func, res, num_up=next_num)
            self.R.univeral_contract(idx, leaf_func, res, num_up=next_num + 1)

        return res

    def update_cores(self, update_up=False):
        try:
            del self._G
        except:
            pass

        try:
            del self._S_mat
        except:
            pass

        try:
            del self._convolv
        except:
            pass

        for nn in self.children:
            nn.update_cores(update_up)

        # global level
        # level = self.level
        # self.row_A.level = self.level
        if self.how_to_build_cores == 'mv' and not self.is_root:
            was_use_maxvol = self.use_maxvol
            self.use_maxvol = True
            if update_up:
                self.update_up()
            self.use_maxvol = was_use_maxvol

    def update_down(self, internal=False):
        # global level
        # level = -self.level
        self.row_A.level = -self.level

        assert not self.is_root, 'Cannot jump over the head!'

        if not self.parent.is_root:
            A, A_tru, I, col_vals  = self.get_matrix_for_MV((self.up_idx, self.up_idx_vals),
                                          (self.down_idx, index_tens_prod(self.parent.down_idx_vals, self.siblis.up_idx_vals) ))
            if A is None:
                return 'end'

            self.down_idx_vals, _, self._down_det, self._down_max_A = select_idx_by_maxvol(A, A_tru, I, col_vals,
                                                                         use_maxvol=self.use_maxvol, T=self.temp, keep_max=self.keep_max_while_maxvol, node=self, whowhen='update_down')

            if internal:
                return True

            # res = self.siblis.update_down(internal=True)
            # if res == "end":
                # return "end"

        if self.down_idx_vals is None and self._down_det is None:
            return 'end'


        if self.how_to_switch == 'seq':
            return self.switch_way_simple()
        elif self.how_to_switch == 'seq_pop':
            return self.switch_way_simple_with_pop()
        elif self.how_to_switch == 'max':
            return self.switch_way_by_max()
        elif self.how_to_switch == 'rand':
            return self.switch_way()
        else:
            raise ValueError("Unknown turn strategy")


    def update_down_with_childrens(self):
        self.update_down()
        for nn in self.children:
            nn.update_down()


    def update_up(self):
        # global level
        # level = self.level
        self.row_A.level = self.level

        assert not self.is_root, 'Cannot jump over the head!'

        if self.is_leaf:
            n = self.shape[self.idx[0]]
            if self.info['emulate_TT'] and self.r == n:
                self.up_idx_vals, self._B, self._up_det, self._up_max_A = [[i] for i in range(n)], np.eye(n), 1, 0
                return self.parent


            # our tensor now is 2D, so it's special case
            my_range = np.asarray(self.full_range_idx)[:, None]
            # print(f"range: {self.idx[0]}, {my_range}")
            A, A_tru, I, col_vals = self.get_matrix_for_MV(
                                          (self.down_idx, self.down_idx_vals),
                                          (self.up_idx, my_range),
                                           can_grow=True
                                          )

        else:
            A, A_tru, I, col_vals  = self.get_matrix_for_MV(
                                          (self.down_idx, self.down_idx_vals),
                                          (self.up_idx, index_tens_prod(self.L.up_idx_vals, self.R.up_idx_vals) )
                                          )

        if A is None:
            return 'end'

        self.up_idx_vals, B, self._up_det, self._up_max_A = select_idx_by_maxvol(A, A_tru, I, col_vals,
                                                    use_maxvol=self.use_maxvol, T=self.temp, keep_max=self.keep_max_while_maxvol, node=self, whowhen='update_up')

        if self.up_idx_vals is None and B is None and self._up_det is None:
            return 'end'

        if B is not None:
            self._B = B

        return self.parent

    def get_matrix_for_MV(self, row, col, can_grow=False, batch=False):

        if self.func_self:
            func = self.get_par
            batch = True
        else:
            func = self.A

        row_idx, row_vals = row
        col_idx, col_vals = col

        d = len(col_idx) + len(row_idx)
        will_grow = can_grow and self.is_leaf and self.func_type[0] == 'c' and not batch  # we shall add new idx

        added_cols = 0 + will_grow

        A = np.empty([len(col_vals) + added_cols , len(row_vals)])
        I = np.zeros(d, dtype=float if self.func_type[0] == 'c' else int)

        I_all = []
        for ci, cv in enumerate(col_vals):
            for ri, rv in enumerate(row_vals):
                I[col_idx] = cv
                I[row_idx] = rv

                if batch:
                    I_all.append(np.copy(I))
                else:
                    res = func(I)
                    if res is None:
                        return [None]*3
                    A[ci, ri] = res

        if batch:
            A[...] = np.asarray(func(np.asarray(I_all))).reshape(*A.shape, order='C')


        if will_grow:
            A_vals = A[:-added_cols]
            # A_vals = A
            if self.is_max:
                max_A_val = np.max(A_vals)
            else:
                max_A_val = np.min(A_vals)

            srch = np.where(A_vals==max_A_val)
            idx_max = srch[0][0]
            idx_max_y = srch[1][0]

            # i12_full = self.append_idx_near_max(col_vals[idx_max][0], 10, False)

            max_res = []

            rv = row_vals[idx_max_y]
            I[row_idx] = rv
            def f_min(cv):
                nonlocal I, col_idx, self
                # self.log(f"test, x={cv}")
                I[col_idx] = float(cv)
                res = self.A(I)
                if self.is_max:
                    res = -res
                # self.log(f"in min func, x={cv}, res={res}")
                return res


            idx_max = float(col_vals[idx_max][0])
            bounds = self.get_bounds_for_arg(idx_max)
            sc_res = scipy.optimize.minimize(f_min, idx_max, tol=1e-5, method="Nelder-Mead", options=dict(maxiter=10), bounds=(bounds, ))
            i12 = [sc_res.x.item()]

            for ci, cv in enumerate(i12, start=-len(i12)):
                for ri, rv in enumerate(row_vals):
                    I[col_idx] = cv
                    I[row_idx] = rv

                    res = func(I)
                    if res is None:
                        return [None]*3
                    A[ci, ri] = res

            col_vals = list(col_vals) + [[i] for i in i12]
            self.append_to_full_range(i12)

        A_tru = A

        if (not batch) and self.norm_A:
            A_tru = A.copy()
            A -= A.mean()
            sigma = np.std(A)
            if sigma > 1e-6:
                A /= sigma

            if not self.is_max:
                A *= -1
                A_tru *= -1

            A = np.exp(A)

        return A, A_tru, I, col_vals

    def get_matrix_for_MV_old(self, row, col, can_grow=False):
        row_idx, row_vals = row
        col_idx, col_vals = col

        d = len(col_idx) + len(row_idx)
        will_grow = can_grow and self.is_leaf and self.func_type[0] == 'c' # we shall add new idx

        added_cols = 0 + will_grow
        A = np.empty([len(col_vals) + added_cols , len(row_vals)])
        I = np.empty(d, dtype=float if self.func_type[0] == 'c' else int)

        for ci, cv in enumerate(col_vals):
            for ri, rv in enumerate(row_vals):
                I[col_idx] = cv
                I[row_idx] = rv

                res = self.A(I)
                if res is None:
                    return [None]*3
                A[ci, ri] = res


        if will_grow:
            A_vals = A[:-added_cols]
            # A_vals = A
            if self.is_max:
                max_A_val = np.max(A_vals)
            else:
                max_A_val = np.min(A_vals)

            srch = np.where(A_vals==max_A_val)
            idx_max = srch[0][0]
            idx_max_y = srch[1][0]

            i12_full = self.append_idx_near_max(col_vals[idx_max][0], 10, False)

            max_res = []
            rv = row_vals[idx_max_y]
            I[row_idx] = rv
            for ci, cv in enumerate(i12_full):
                I[col_idx] = cv

                res = self.A(I)
                if res is None:
                    return [None]*3
                max_res.append(res)



            for ci, cv in enumerate(i12, start=-2):
                for ri, rv in enumerate(row_vals):
                    I[col_idx] = cv
                    I[row_idx] = rv

                    res = self.A(I)
                    if res is None:
                        return [None]*3
                    A[ci, ri] = res

            col_vals = list(col_vals) + [[i1], [i2]]


        A_tru = A

        if self.norm_A:
            A_tru = A.copy()
            A -= A.mean()
            sigma = np.std(A)
            if sigma > 1e-6:
                A /= sigma

            if not self.is_max:
                A *= -1
                A_tru *= -1

            A = np.exp(A)

        return A, A_tru, I, col_vals

    #@property
    #def remap_idx(self):
    #    assert self.is_leaf, "Remap is only valid for leaf nodes"

    #    try:
    #        res = self._remap_dict
    #    except AttributeError:
    #        n = self.shape[self.up_idx[0]]
    #        res = self._remap_dict = {i: i for i in range(n)}
    #        self.init_n = n  # needed for cyclic arguments

    #    return res

    @property
    def full_range_idx(self):
        assert self.is_leaf, "full_range_idx is only valid for leaf nodes"
        try:
            res = self._full_range_idx
        except AttributeError:
            n = self.shape[self.up_idx[0]]
            if self.func_type[0] == 'd':
                res = np.arange(n)
            else:
                res = list(range(n))

            self._full_range_idx = res
            self.init_n = n  # needed for cyclic arguments

        return res

    def get_bounds_for_arg(self, idx):
        assert self.is_leaf, "works only valid for leaf nodes"

        vals = self.full_range_idx
        # vals.sort()
        n = len(vals)

        pos = bisect.bisect_left(vals, idx)
        assert pos != n and vals[pos] == idx, "No such element"

        if pos > 0 and pos < n - 1:
            return vals[pos-1], vals[pos+1]

        else:
            if not self.cyclic:
                if pos == 0:
                    return 2*idx - vals[1], vals[1]
                elif pos == n - 1:
                    return vals[-2], 2*idx - vals[-2]
                else:
                    assert False, "Should not get here"
            else: # cyclic
                if pos == 0:
                    # assert idx == 0
                    return -1, vals[1]
                elif pos == n - 1:
                    return vals[-2], self.init_n
                else:
                    assert False, "Should not get here"



    def append_idx_near_max(self, idx, num=1, append_to_array=False):
        """
        !!! not used
        """
        self.full_range_idx.sort()
        vals = self.full_range_idx
        n = len(vals)
        pos = vals.index(idx)

        if pos > 0 and pos < n - 1:
            i1 = (idx + vals[pos+1])/2
            i2 = (idx + vals[pos-1])/2

        else:
            if not self.cyclic:
                if pos == 0:
                    i1 = (idx + vals[1])/2
                    i2 = 2*idx - vals[1]
                elif pos == n - 1:
                    i1 = 2*idx - vals[-2]
                    i2 = (idx + vals[-2])/2
                else:
                    assert False, "Should not get here"
            else: # cyclic
                if pos == 0:
                    assert idx == 0
                    i1 = vals[1] / 2.
                elif pos == n - 1:
                    i1 = (idx + vals[-2])/2
                else:
                    assert False, "Should not get here"
                i2 = (vals[-1] + self.init_n)/2

        i12 = i1 + i2
        if append_to_array:
            self.append_to_full_range(i12)
        return i1 + i2


    def append_to_full_range(self, i12):
        assert self.is_leaf, "Can add idx only to leaf"
        vals = self.full_range_idx
        real_ex = []
        for i in i12:
            if self.cyclic and i < 0:
                i += self.init_n

            # if i not in :
                # self.full_range_idx.append(i)
            pos = bisect.bisect_left(vals, i)
            if not (pos != len(vals) and vals[pos] == i):
                bisect.insort(vals, i)
                self.shape[self.up_idx[0]] += 1
                real_ex.append(i)
        # assert self.shape[self.up_idx[0]] == n + 2
        # self.full_range_idx.extend(i12)
        self.log(f" was extend by {i12} (real: {real_ex})")

    def log(self, txt, introduce=True):
        try:
            to_print = self.debug
        except:
            to_print = True

        if to_print:
            if introduce:
                txt = str(self) + " : " + txt
            print(txt)



def select_idx_by_maxvol(A, A_tru, I, col_vals, *, use_maxvol=True, T=0, keep_max=False,
                        node=None, whowhen='unknown'):
    
    
    try:
        inf = node.info
    except:
        inf = dict()
    
    tau  = inf.get('tau',  1.01)
    tau0 = inf.get('tau0', 1.01)
    k0   = inf.get('k0',   500)
    rank_reduce_tresh = inf.get('rank_reduce_tresh', 1e-10)
    dr_max_init = inf.get('dr_max', 1)
    
    
    try:
        debug = node.debug
    except:
        debug = True

    max_A = np.max(A_tru)
    B = None
    Q = A
    det = None

    if not use_maxvol:
        if T == 0:
            ind = np.argsort(np.max(np.abs(Q), axis=1))[::-1]
            ind = ind[:Q.shape[1] + dr_min]
        else:
            ind = sort_T(np.abs(Q), T, Q.shape[1] + dr_min)

    else:
        #Q, _ = np.linalg.qr(A)
        #dr_min = dr_max = max(rank - Q.shape[1], 0)
        #try:
        #    ind, B = teneva._maxvol(Q, tau, dr_min, dr_max, tau0, k0)
        #except np.linalg.LinAlgError: # singular matrix
        #    Qm, Rm, Pm = scipy.linalg.qr(Q, pivoting=True, mode='economic')
        #    real_rank = (np.abs(np.diag(Rm) / Rm[0, 0]) > 1e-9).sum()
        #    print(f"WARN: rank reduced from {Q.shape[1]} to {real_rank}")
        #    Q = Qm[:, Pm[:real_rank]]
        #    dr_min = dr_max = 0

        prev_rank = A.shape[1]
        Qm, Rm, Pm = scipy.linalg.qr(A, pivoting=True, mode='economic')
        real_rank = (np.abs(np.diag(Rm) / Rm[0, 0]) > rank_reduce_tresh).sum()
        dr_min = dr_max = 0
        if real_rank == prev_rank:
            dr_max = dr_max_init
        else:
            if debug:
                print(f"WARN: rank reduced from {prev_rank} to {real_rank} during {whowhen} for {node}")

        #Q = Qm[:, Pm[:real_rank]]
        Q = Qm[:, :real_rank]
        #print(Q.shape, tau, dr_min, dr_max, tau0, k0)
        ind, B = teneva._maxvol(Q, tau, dr_min, dr_max, tau0, k0)

        # where is the max:
        if keep_max:
            if max_A > np.max(A_tru[ind]):
                # print(f"We are missing him: {max_A} > {np.max(A_tru[ind])} ({np.max(A)} ?> {np.max(A[ind])}, ({np.max(Q)} ?> {np.max(Q[ind])})")
                ind_to_insert = np.where(A==np.max(A))[0][0]
                where_to_insert = np.argmax([Q[i] @ Q[ind_to_insert] for i in ind])
                ind[where_to_insert] = ind_to_insert

                assert np.max(A[ind])==np.max(A), "!!!!!"



        Az = A_tru[ind]
        # det = np.linalg.det(Az.T @ Az)
        sgn, det = np.linalg.slogdet(Az.T @ Az)
        # print(A.shape, Az.shape, det, ind, Az)
        if sgn <= 0:
            det = -np.inf
        # else:
            # det = np.log(det)

    return [col_vals[i] for i in ind], B, det, max_A

def where_bisect(l, i):
    pos = bisect.bisect_left(l, i)
    if pos >= len(l) or l[pos] != i:
        return False, pos
    else:
        return True, pos


def stable_double_inv(A_cur, S1, S2):
    shapes_A = A_cur.shape
    T1 = np.linalg.solve(S1, A_cur.reshape(A_cur.shape[0], -1))
    T1 = T1.reshape(*shapes_A)
    T1 = np.transpose(T1, [1, 0, 2])
    T1_shapes = T1.shape
    T1 = T1.reshape(T1.shape[0], -1)
    T2 = np.linalg.solve(S2, T1)
    T2 = T2.reshape(*T1_shapes)
    T2 = np.transpose(T2, [1, 2, 0])
    return np.copy(T2)
