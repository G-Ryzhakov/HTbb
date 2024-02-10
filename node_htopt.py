import numpy as np


from node import Node
from utils import argsort
from utils import my_cach
import inspect


def _log(text):
    log = inspect.currentframe().f_back.f_locals.get('log', True)
    if log:
        print(text)


def eval_opt(trees, bm, expected_max=None, add_rank=1, max_rank=1e10,
             num_iter=100, iter_with_maxvoll=50, ktop=0, sample=0,
             remove_strat=None, value_to_compare='max', use_full_path=False,
             log=False, to_update_tree=True):
    """
    args:
    bm -- function to be optimized
    tree -- root node of HTucker

    """
    bm.max = -np.inf if trees[0].is_max else np.inf
    if expected_max is None:
        expected_max = -bm.max

    hits = bm.func.cache_info().hits
    miss = bm.func.cache_info().misses

    LNodes, RNodes = dict(), dict()

    for i in range(len(trees)):
        LNodes[i] = None
        RNodes[i] = None


    def print_stat():
        nonlocal hits
        nonlocal miss
        nonlocal bm
        _log(str(bm.func.cache_info()) + " Δ hits: " + str(bm.func.cache_info().hits - hits) + " Δ misses: " + str(bm.func.cache_info().misses - miss))
        hits = bm.func.cache_info().hits
        miss = bm.func.cache_info().misses

    def _iter(tree, it):
        nonlocal iter_with_maxvoll, ktop, sample, bm, add_rank, max_rank, LNodes, RNodes

        # bm.loc_max = -np.inf
        bm.loc_max = -np.inf if tree.is_max else np.inf
        _log(f"iterations... tree num {it}")
        tree.use_maxvol = True
        node = None
        for _ in range(iter_with_maxvoll):
            _log(".")
            if use_full_path:
                res = loop_parallel_half(tree, bm, n=1, node_list=[LNodes[it], RNodes[it]], expected_max=None, log=log, add_rank=0)
                if res is False:
                    return
                else:
                    LNodes[it], RNodes[it] = res
            else:
                node = loop(tree, node)

        _log(f"\nMV iterations stop,. tree num {it}")
        print_stat()

        if to_update_tree:
            tree.update_cores()

        if ktop > 0:
            _log("ktop...")

            idx_ktop = tree.topk(ktop)
            bm(idx_ktop)

            _log("end ktop")
            print_stat()

        if sample > 0:
            _log("sample...")

            for _ in range(sample):
                bm(tree.sample())

            bm(tree.sample_max())

            _log("end sample")
            print_stat()

        if add_rank:
            tree.r = min(tree.r + add_rank, max_rank)
            _log(f"Now rank is: {tree._r_raw}")

        tree._loc_max = bm.loc_max
        _log(f"local max: {bm.loc_max}")

    for i in range(num_iter):
        _log(f"Iteration number {i}")
        for it, tree in enumerate(trees):
            _iter(tree, it)


        k_ismax = 1 if trees[0].is_max else -1
        if k_ismax*bm.max >= k_ismax*expected_max:
            _log("Optimum reached")
            break

        if remove_strat is None or len(remove_strat) <= i:
            continue

        to_keep = remove_strat[i]
        if value_to_compare[:3] == 'max':
            new_values = [(tree._loc_max, k_ismax*tree.up_det) for tree in trees]
        else: # value_to_compare[:3] == 'det':
            new_values = [(k_ismax*tree.up_det, tree._loc_max) for tree in trees]

        _log(f"len of tree will be: {to_keep}, values of det: {new_values}")
        #idx_vals = np.argsort(new_values)[::-1]
        # print(f"good vals: {new_values}, {sorted(new_values)[::-1]}")
        idx_vals = argsort(new_values)
        if trees[0].is_max:
            idx_vals = idx_vals[::-1]

        idx_vals = idx_vals[:to_keep]

        _log(f"new tree: det: {[new_values[i] for i in idx_vals]}")
        trees = [trees[i] for i in idx_vals]


def loop(tr, start_node=None):
    if start_node is None:
        leaves = tr.all_leaves()
        start_node = leaves[np.random.choice(len(leaves))]

    while not start_node.parent.is_root:
        start_node = start_node.update_up()

    start_node.update_up()

    start_node = start_node.siblis

    while start_node is not None:
        prev = start_node
        start_node = start_node.update_down()

    return prev


def loop_parallel(tr, bm, n=1e4, *, log=True, add_rank=0, max_rank=100, add_rank_freq=2, to_upd_cores=False):
    n = int(n)
    tr_nodes = [tr.leafs_of_level(i + 1) for i in range(tr.max_level)]

    hits = bm.func.cache_info().hits
    miss = bm.func.cache_info().misses


    for it in range(n):
        bm.loc_max = -np.inf if tr.is_max else np.inf
        # _log("up")
        for nn in tr_nodes[::-1]:
            # bm.loc_max = -np.inf if tr.is_max else np.inf
            nn_cur = [i for i in nn if not i.cursed]
            np.random.shuffle(nn_cur)
            for i in nn_cur:
                if i.update_up() == 'end':
                    return False
            # _log(f"iterations, loc.max {bm.loc_max}, iter num {it}")

        # _log("down")
        for nn in tr_nodes:
            # bm.loc_max = -np.inf if tr.is_max else np.inf
            nn_cur = [i for i in nn if not i.cursed]
            np.random.shuffle(nn_cur)
            for i in nn_cur:
                if i.update_down() == 'end':
                    return False
            # _log(f"iterations, loc.max {bm.loc_max}, iter num {it}")

        _log(f"iterations, loc.max {bm.loc_max}, iter num {it}")
        _log(str(bm.func.cache_info()) + " Δ hits: " + str(bm.func.cache_info().hits - hits) + " Δ misses: " + str(bm.func.cache_info().misses - miss))
        hits = bm.func.cache_info().hits
        miss = bm.func.cache_info().misses

        if to_upd_cores:
            tr.update_cores()

        if add_rank:
            if (it + 1) % add_rank_freq == 0:
                tr.r = min(tr.r + add_rank, max_rank)
            _log(f"Now rank is: {tr._r_raw}")
    return True


def loop_parallel_half(tr, bm, n=1e4, *, node_list=None, expected_max=None, log=False, add_rank=0):
    n = int(n)

    # bm.max = -np.inf if tr.is_max else np.inf

    hits = bm.func.cache_info().hits
    miss = bm.func.cache_info().misses


    def print_stat():
        nonlocal hits
        nonlocal miss
        nonlocal bm
        _log(str(bm.func.cache_info()) + " Δ hits: " + str(bm.func.cache_info().hits - hits) + " Δ misses: " + str(bm.func.cache_info().misses - miss))
        hits = bm.func.cache_info().hits
        miss = bm.func.cache_info().misses

    build_LR = False
    if node_list is None:
        build_LR = True

        Lnodes = []
        Rnodes = []
    else:
        Lnodes, Rnodes = node_list

    if Lnodes is None or Rnodes is None:
        build_LR = True

        Lnodes = []
        Rnodes = []


    if build_LR:
        for i in range(tr.max_level):
            cur_L = []
            cur_R = []
            for i in tr.leafs_of_level(i + 1):
                if i.cursed:
                    continue

                if tr.L.is_offspring(i):
                    cur_L.append(i)
                else:
                    cur_R.append(i)

            Lnodes.append(cur_L)
            Rnodes.append(cur_R)


    def all_subtree_up(nodes):
        for nn in nodes[::-1]:
            nn_cur = list(nn)
            np.random.shuffle(nn_cur)
            for i in nn_cur:
                if i.update_up() == 'end':
                    return False

        return True

    def all_subtree_down(nodes):
        for nn in nodes:
            nn_cur = list(nn)
            np.random.shuffle(nn_cur)
            for i in nn_cur:
                if i.update_down() == 'end':
                    return False

        return True

    _log("Starting iterations...")

    for _ in range(n):
        if not all_subtree_up(Lnodes):
            return False
        print_stat()

        if not all_subtree_down(Rnodes):
            return False
        print_stat()


        if not all_subtree_up(Rnodes):
            return False
        print_stat()

        if not all_subtree_down(Lnodes):
            return False
        print_stat()

        if add_rank:
            tr.r = tr.r + add_rank
            _log(f"Now rank is: {tr._r_raw}")


    return Lnodes, Rnodes


def simple_opt(tr2, bm_test_1, *, expected_max=None, add_rank=1, num_iter=1000,
               iter_with_max=0, iter_with_maxvoll=10, ktop=0, sample=1000):
    """
    args:
    bm_test_1 -- function to be optimized
    tr2 -- root node of HTucker

    """
    #bm_test_1 = tr2._A
    bm_test_1.max = -np.inf if tr2.is_max else np.inf
    if expected_max is None:
        expected_max = -bm_test_1.max

    node = None
    hits = bm_test_1.func.cache_info().hits
    miss = bm_test_1.func.cache_info().misses


    def print_stat():
        nonlocal hits
        nonlocal miss
        nonlocal bm_test_1
        print(bm_test_1.func.cache_info(), bm_test_1.func.cache_info().hits - hits, bm_test_1.func.cache_info().misses - miss)
        hits = bm_test_1.func.cache_info().hits
        miss = bm_test_1.func.cache_info().misses

    for i in range(num_iter):
        tr2.use_maxvol = True
        for _ in range(iter_with_maxvoll):
            node = loop(tr2, node)

        print_stat()

        tr2.update_cores()
        print("Update up all ended")
        if ktop > 0:
            tr2.update_cores()
            print("ktop")

            idx_ktop = tr2.topk(ktop)
            bm_test_1(idx_ktop)

            print("end ktop")

            print_stat()
        # print("sample")
        # for _ in range(sample):
            # bm_test_1(tr2.sample())
        # bm_test_1(tr2.sample_max())
        # print("end sample")

        # print_stat()

        tr2.use_maxvol = False
        for _ in range(iter_with_max):
            node = loop(tr2, node)

        print_stat()

        if ktop > 0:
            tr2.update_cores()
            print("ktop")

            idx_ktop = tr2.topk(ktop)
            bm_test_1(idx_ktop)

            print("end ktop")

        print_stat()
        # print("sample")
        # for _ in range(sample):
            # bm_test_1(tr2.sample_max())
        # bm_test_1(tr2.sample_max())
        # print("end sample")

        # print_stat()

        if add_rank:
            tr2.r = tr2.r + add_rank
            print(f"Now rank is: {tr2._r_raw}")

        if (tr2.is_max and bm_test_1.max >= expected_max) or (not tr2.is_max and bm_test_1.max <= expected_max):
            break


def which_direct_freq_softmax(v1r, v2r, n, alpha=0.1):
    v1 = alpha*v1r/n
    v2 = alpha*v2r/n

    #print(f"probs: {scipy.special.softmax([v1, v2])}, {v1}, {v2}")
    prob = scipy.special.softmax([v1, v2])[0]
    return "down" if np.random.rand() > prob else "up"


def which_direct_freq(v1, v2, n=None, alpha=0.1):
    v1 = - v1
    v2 = - v2

    ss = abs(v1 + v2)/n*alpha
    # ss = 0

    if v1 > v2 + ss:
        direc = 'up'
    elif v1 < v2 - ss:
        direc = "down"
    else:
        direc = 'up' if np.random.uniform() < 0.5 else "down"

    return direc




def uniform_walk(tr, nswp=50, *, use_path=False, update=True, log=False,
                 callback=None, callback_freq=5000, alpha=0.1, solidly_first=False, max_path_len=100, path=None, finalize=True):
    #n = int(n)
    
    
    cur_enters = 0
    direc = "down"
    node = tr

    if path is None:
        if use_path:
            path = []
    else:
        use_path = True

    max_level = tr.max_level

    def new_node(n, direc=None):
        nonlocal path, use_path
        n.count_enters += 1
        #print(n.full_num, n.count_enters, n.cursed)
        if use_path:
            path.append([n.full_num, direc])
            while len(path) > max_path_len:
                del path[0]

    def size_of_tree(node):
        return node.number_of_all_childrens + 1e-6

    def size_of_tree_old(node):
        level = node.level
        n = max_level - level + 1
        return 2**n - 1


    if solidly_first:
        go_solidly_down(tr)
        lvs = tr.all_leaves()
        lvs.sort(key=lambda x: x._down_max_A)
        node = lvs[-1]
        _log(f"solidly end, starting from {node._down_max_A}")

    prev_node = None

    cnt = 0
    while nswp > 0:
        cnt += 1
        #n -= 1

        if callback is not None:
            if cnt % callback_freq == 0:
                callback(info=dict(n=nswp, cnt=cnt, r=tr.r))


        #if add_rank:
        entrs = [nn.count_enters for nn in tr.all_children_iter if not nn.cursed]
        entrs_min = min(entrs)
        if entrs_min > cur_enters:
            cur_enters = entrs_min
            num_entr_max = max(entrs)
            num_entr_avg = np.mean(entrs)

            nswp -= 1
            _log(f"{nswp=}, enters: {entrs_min}/{num_entr_max}/{num_entr_avg:.3f}")
            
        #if add_rank_start < entrs_min:
        # if cnt % add_rank_freq == 0:
            #add_rank_start += add_rank_freq
            #tr.r = min(tr.r + add_rank, max_rank)
            #_log(f"Now rank is: {tr._r_raw}")

        if direc == "up":
            if update:
                node.update_up()
            #make_move_after_up(node, prev_node, val_compare='max')
            if node.parent.is_root:
                #print(1)
                node = node.siblis
                new_node(node, direc)
                direc = "down"
                prev_node = None
                continue
            else:
                #print(2)
                prev_node = node
                node = node.parent
                new_node(node, direc)
                # now, where to go
                if prev_node.siblis.cursed:
                    direc = "up"
                else:
                    v1 = (tr.count_enters_with_childrens - node.count_enters_with_childrens)/(size_of_tree(tr) - size_of_tree(node)) # "up"
                    v2 = prev_node.siblis.count_enters_with_childrens/size_of_tree(prev_node.siblis) # "down"
                    direc = which_direct_freq(v1, v2, cnt, alpha=alpha)
        else: #  direc == "down":
            if node.is_leaf: # just reflect
                #print(3)

                direc = "up"
                #new_node(node, direc)
                # continue
            else:
                if not node.is_leaf:
                    if all([nn.cursed for nn in node.children]):
                        direc = "up"
                        continue

                if prev_node is None:
                    #print(41)

                    # now, where to go
                    if node.L.cursed:
                        v1 = 1e10
                        dLR = 'down'
                    else:
                        v1 = node.L.count_enters_with_childrens/size_of_tree(node.L) # "left"


                    if node.R.cursed:
                        #assert v1 < 1e10, "Never be1"
                        v2 = 1e10
                        dLR = 'up'
                    else:
                        v2 = node.R.count_enters_with_childrens/size_of_tree(node.R) # "right"


                    if v1 < 1e10 and v2 < 1e10:
                        dLR = which_direct_freq(v1, v2, cnt, alpha=alpha)

                    if dLR == 'up':
                        node = node.L
                    else:
                        node = node.R


                else:
                    #print(42)
                    node = prev_node.siblis
                    prev_node = None

            if update:
                node.update_down()

            new_node(node, direc)


    callback(info=dict(n=nswp, cnt=cnt, r=tr.r))
    if finalize:
        _log(f"Finalizing...")
        dr_max = tr.info['dr_max']
        tr.info['dr_max'] = 0
        tr.update_cores(True)
        tr.info['dr_max'] = dr_max
        callback(info=dict(n=nswp, cnt=cnt, r=tr.r))
        
    _log("Ended")
    if use_path:
        return path




def plot_path(path):
    def coord_to_xy(l, n):
        y = float(-l)
        x = float(n - 2**(l-1))
        return x, y

    np.random.seed(42)

    plt.figure(figsize=(15, 10))
    pnts = np.array([coord_to_xy(l, n) for l, n in path])
    pnts += np.random.rand(*pnts.shape)*0.3
    plt.plot(pnts[:, 0], pnts[:, 1], "r-")



def simple_set_htree(d, func, r, shape, *, seed=42,  **kwargs):
    np.random.seed(seed) # TODO: replace with inner generator
    pow_2 = 2**np.arange(20)
    d_full = pow_2[np.searchsorted(pow_2, d, side='left')]
    idxx = kwargs.pop("idx", None)
    if idxx is None:
        idxx = np.arange(d_full)
    return Node(idxx, A=func, r=r, shape=list(shape) + [1]*(d_full - d),
        d_real=d,  **kwargs)


@my_cach(log=True)
def test_T(I):
    return sum(I) + np.prod(I) + sum(I)*np.prod(I)


def test_T_simple(I):
    return sum(I)

