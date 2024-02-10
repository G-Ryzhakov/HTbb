from functools import lru_cache
import itertools
import numpy as np
import teneva


from teneva_bm import BmBudgetOverException


class FatherKnowsBetter():
    def __init__(self, func):
        self._name = func.__name__
        self._name_int = f"_{self._name}"

    def __get__(self, me, owner=None):
        try:
            return getattr(me, self._name_int)
        except  AttributeError:
            setattr(me, self._name_int, getattr(me.parent, self._name))
            return getattr(me, self._name_int)

    def __set__(self, me, value):
        setattr(me, self._name_int, value)
        for nn in me.children:
            setattr(nn, self._name, value)


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def find_next_free_num(arr):
    res = []
    for l in arr[1::2]:
        res.extend(l)
    return max(res) + 1


def index_tens_prod(A, B):
    return [list(a) + list(b) for a, b in itertools.product(list(A), list(B))]


def matrix_skeleton_sep(A, e=1.E-10, r=1.E+12, hermitian=False, rel=False):
    U, s, V = np.linalg.svd(A, full_matrices=False, hermitian=hermitian)

    ss = s/s[0] if rel else s
    where = np.where(np.cumsum(ss[::-1]**2) <= e**2)[0]
    dlen = 0 if len(where) == 0 else int(1 + where[-1])
    r = max(1, min(int(r), len(s) - dlen))

    return np.diag(s[:r]), U[:, :r], V[:r, :]


def matrix_skeleton_spec(A, e=1.E-10, r=1.E+12):
    U, s, V = np.linalg.svd(A, full_matrices=False, hermitian=False)

    where = np.where(np.cumsum(s[::-1]**2) <= e**2)[0]
    dlen = 0 if len(where) == 0 else int(1 + where[-1])
    r = max(1, min(int(r), len(s) - dlen))

    S = np.diag(np.sqrt(s[:r]))
    return U[:, :r] @ S, S @ V[:r, :]


# level = -1
# cur_rank = -1


def my_cach(*args, **kwargs):
    def my_cach_inner(func, cache_size=9.E+6, cache_max=5*9.E+6, log=False,
                      is_max=True, check_cach=True, dtype=int):
        func = lru_cache(maxsize=int(cache_size))(func)
        def f(I):
            hits = func.cache_info().hits
            if check_cach:
                if hits >= cache_max:
                    raise BmBudgetOverException(hits, is_cache=True)

            I = np.asarray(I, dtype=dtype)
            # global level, cur_rank
            # global cur_rank
            try:
                level = f.level
            except:
                level = None
            if I.ndim == 1:
                y = func(tuple(I))
                if (is_max and y > f.loc_max) or (not is_max and y < f.loc_max):
                    f.loc_max = y
                    # print(f">>> loc max: {f.loc_max}")

                if (is_max and y > f.max) or (not is_max and y < f.max):
                    f.max = y
                    if log:
                        print(f">>> max: {f.max} ({func.cache_info().misses} evals) (level={level}) (I = {I})")
                return y
            elif I.ndim == 2:
                y = [func(tuple(i)) for i in I]
                max_y = max(y) if is_max else min(y)
                if (is_max and max_y > f.loc_max) or (not is_max and max_y < f.loc_max):
                    f.loc_max = max_y
                    # print(f">>> loc max: {f.loc_max}")

                if (is_max and max_y > f.max) or (not is_max and max_y < f.max):
                    f.max = max_y
                    if log:
                        print(f">>> max: {f.max}")
                return y
            else:
                raise TypeError('Bad argument')

        f.max = -np.inf if is_max else np.inf
        f.loc_max = -np.inf if is_max else np.inf
        f.func = func

        return f

    if len(args) == 1 and callable(args[0]):
        return my_cach_inner(args[0])

    return lambda func: my_cach_inner(func, *args, **kwargs)


def select_r_different(arng, r):
    rng = len(arng)

    if rng >= r:
        return [arng[i] for i in np.random.choice(rng, size=r, replace=False)]
    else:
        return arng


def skel_two(G, e=1e-9, rmax=100000, rel=True):
    r1, n, r2 = G.shape
    U1, G = teneva.matrix_skeleton(
        G.reshape(r1, -1), e=e, r=rmax, rel=rel, give_to='l')
    G = G.reshape(-1, n, r2)
    r1_new = G.shape[0]

    G, U2 = teneva.matrix_skeleton(
        G.reshape(-1, r2), e=e, r=rmax, rel=rel, give_to='r')
    G = G.reshape(r1_new, n, -1)

    return G, U1, U2


def sort_T(Q, T, num):
    l = scipy.special.logsumexp(Q/T, axis=1)
    p = scipy.special.softmax(l)
    if (p > 0).sum() < num:
        p += 1e-8
        p /= p.sum()
    return np.random.choice(len(l), p=p, size=num, replace=False)


def stoh_argmax_num(a, n=3):
    # TODO: sett outer random generator
    a = np.abs(a)
    a_max = a[np.argsort(a)[::-1][:n]]
    idxs = np.hstack([np.where(a==a_m)[0] for a_m in a_max if a_m > 0])
    return np.random.choice(idxs)
