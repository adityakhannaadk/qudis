"""ilpdistance.py
A handwritten derivation of the ILP formulation here is discussed in "distances.pdf".
This has the option of being warm-started with a heuristic solution from BP-OSD or QDistRnd. 
Originally based on the ILP code from https://github.com/txhaug/QuditLDPCCodes/blob/main/DecodeQuditMIP.py 
"""
import mip
import numpy as np
import time
from bposddistance import distance_upper_bound_bposd_qudit as heuristic_distance_test_qudit


def distance_test_qudit(stab, logicOp, qudit, ub=None,heuristic_solution=None, max_time=mip.INF):
        

    assert qudit != 4

    logicOp_tmp = np.array(logicOp)
    stab_tmp = np.array(stab)
    start_time = time.time()

    n = stab_tmp.shape[1]
    m = stab_tmp.shape[0]
    wstab = np.max([np.sum((stab_tmp[i, :] % qudit) != 0) for i in range(m)])
    wlog = np.count_nonzero(logicOp_tmp)
    num_anc_stab = int(np.ceil(np.log((qudit - 1) ** 2 * wstab) / np.log(qudit)))
    num_anc_logical = int(np.ceil(np.log((qudit - 1) ** 2 * wlog) / np.log(qudit)))
    num_var = n + m * num_anc_stab + num_anc_logical

    model = mip.Model()
    model.verbose = 0
    model.threads = 1  # avoid CBC pseudocost-branching race/assert

    x = [model.add_var(var_type=mip.INTEGER, lb=0, ub=qudit - 1) for _ in range(num_var)]

    if qudit == 2:
        weight_expr = mip.xsum(x[i] for i in range(n))
    else:
        y = [[model.add_var(var_type=mip.INTEGER, lb=0, ub=1) for _ in range(qudit - 1)]
             for _ in range(n)]
        for i in range(n):
            model += x[i] - mip.xsum((j + 1) * y[i][j] for j in range(qudit - 1)) == 0
            model += mip.xsum(y[i][j] for j in range(qudit - 1)) <= 1
        weight_expr = mip.xsum(y[i][j] for i in range(n) for j in range(qudit - 1))

    model.objective = mip.minimize(weight_expr)

    for row in range(m):
        weight = np.zeros(num_var, dtype=int)
        weight[:n] = stab_tmp[row, :] % qudit
        cnt = 1
        for q in range(num_anc_stab):
            weight[n + row * num_anc_stab + q] = -(qudit ** cnt)
            cnt += 1
        model += mip.xsum(weight[i] * x[i] for i in range(num_var)) == 0

    weight = np.zeros(num_var, dtype=int)
    weight[:n] = logicOp_tmp
    cnt = 1
    for q in range(num_anc_logical):
        weight[n + m * num_anc_stab + q] = -(qudit ** cnt)
        cnt += 1

    model += mip.xsum(weight[i] * x[i] for i in range(num_var)) >= 1
    model += mip.xsum(weight[i] * x[i] for i in range(num_var)) <= qudit - 1

    # heuristic ub from bposd
    if ub is not None:
        # 1) cutoff: solver never explores nodes whose LP bound exceeds ub.
        #    Since we already know a solution of weight `ub` exists, the search
        #    only needs to look for something STRICTLY better; anything not
        #    provably better than ub can be pruned immediately.
        model.cutoff = ub+1

    if ub is not None and heuristic_solution is not None:
        start = [(x[i], int(heuristic_solution[i]) % qudit) for i in range(n)]
        if qudit != 2:
            for i in range(n):
                val = int(heuristic_solution[i]) % qudit
                for j in range(qudit - 1):
                    start.append((y[i][j], 1 if val == j + 1 else 0))
        model.start = start

    res = model.optimize(max_seconds=max_time)
    end_time = time.time() - start_time
    print(res, "time:", end_time)

    if res == mip.OptimizationStatus.INFEASIBLE:
        print("Apparently infeasible")
        # nothing at all satisfies orthogonality/anti-commutation -- shouldn't
        # happen for a real logical operator, but guard anyway
        # TODO: this actually happens often if the heuristic guess is the correct distance
        # this should be fixed, currently I have juset set model.cutoff = ub+1 to avoid this
        return -1

    if res == mip.OptimizationStatus.NO_SOLUTION_FOUND:
        # cutoff pruned everything and nothing strictly better than ub was found
        # => ub itself is optimal
        return ub

    opt_val = int(sum(x[i].x != 0 for i in range(n)))
    # return opt val and the minimum weight logical:
    return opt_val