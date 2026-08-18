# -*- coding: utf-8 -*-

"""
Searching for Qudit BB codes using graph coverings

"""
# TODO: (in order)
# -> Build search and check how to parallelise
# -> Write MIP implementation with Cuda CuOpt
# -> Fix BP issues

import os
import csv
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import scipy
import networkx as nx
import galois
import argparse

try:
    from .graph import graph_connectivity
    from .ilpdistance import distance_test_qudit
    from .helpers import print_as_polynomial, to_array
    from .logical import qudit_css_code, get_bicycle_code
    from .cudailpdistance import heuristic_distance_test_qudit
    from .qudit_bposd import quditBPOSD
    import prune
except ImportError:  # pragma: no cover - fallback for running the file directly
    from graph import graph_connectivity
    from ilpdistance import distance_test_qudit
    from helpers import print_as_polynomial, to_array
    from logical import qudit_css_code, get_bicycle_code
    from qudit_bposd import quditBPOSD
    from cudailpdistance import heuristic_distance_test_qudit
    import prune

gamma1, gamma2, delta1, delta2=(1, 1, 1, -1)


def append_code_to_csv(code_dict, csv_path="found_codes.csv"):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        polynomial_A = print_as_polynomial(code_dict["factorA"], code_dict["A"], "A")
        polynomial_B = print_as_polynomial(code_dict["factorB"], code_dict["B"], "B")
        writer.writerow([code_dict["final_depiction"], polynomial_A, polynomial_B])


def evaluate_code_candidate(candidate):
    A_mod, B_mod, ell_mod, m_mod, qudit, factorA, factorB, base_d = candidate
    code_dict = {
        "ell": ell_mod,
        "m": m_mod,
        "A": A_mod,
        "B": B_mod,
        "qudit": qudit,
        "factorA": factorA,
        "factorB": factorB,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "delta1": delta1,
        "delta2": delta2,
        "final_depiction": None,
    }

    Hx, Hz = get_bicycle_code(code_dict)
    n, K, HZ, HX, lz, lx = qudit_css_code(qudit, to_array(Hx), to_array(Hz))

    Tconn = graph_connectivity(Hx, Hz, HX, HZ)
    if not Tconn:
        return None

    if K == 0:
        print("K=0!")
        return None


    lx = np.array(lx, dtype=np.int8)
    lz = np.array(lz, dtype=np.int8)

    distance_list_x = []
    for i in range(K):
        print(f"Getting heuristic distance for i={i}")
        bposd = quditBPOSD(qudit, max_iter=30)
        w1, e = bposd.distance_upper_bound_bposd_qudit(HZ, lx[i, :], qudit)
        if w1 is None:
            print("Heuristic BP-OSD found no valid upper bound for this logical operator; skipping candidate.")
            return None
        print(f"Heuristic distance is w1={w1}")
        print("\n \n \n \n")
        if w1 <= base_d:
            print("Heuristic distance is too small, continuing..")
            return None
    for i in range(K):
        print(f"Getting distance for i={i} for n={n} and k = {K}")
        w = distance_test_qudit(HZ, lz[i, :], qudit)
        if w <= base_d:
            # if h is odd, w >= base_d guaranteed
            # if h is even this is not guaranteed
            print(f"Distance {w} is too small, continuing...")
            return None
        print("Logical qudit=", i, "Distance=", w)
        distance_list_x.append(w)

    distance_x = np.amin(distance_list_x)

    print("Found distance x", distance_x)
    print("[[", n, ",", K, ",", distance_x, "]]_" + str(qudit))
    if K*distance_x**2/n > 1.5:
        print("Reasonably good code found!")
        print_as_polynomial(factorA, A_mod, "A")
        print_as_polynomial(factorB, B_mod, "B")
        print(f"kd^2/n = {K*distance_x**2/n:.3f}, k/2n = {K/(2*n):.3f}")
        print("\n \n \n \n")
    code_dict["final_depiction"] = f"[[{n},{K},{distance_x}]]_{qudit}, kd^2/n = {K*distance_x**2/n:.3f}, k/2n = {K/(2*n):.3f}"
    return (A_mod, B_mod), code_dict, distance_x


def evaluate_candidates_in_parallel(candidates, max_workers=None):
    if not candidates:
        return []

    max_workers = max_workers or min(max(1, os.cpu_count() or 1), len(candidates))
    if len(candidates) == 1:
        return [evaluate_code_candidate(candidates[0])]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f"Doing things! Evaluating {len(candidates)} candidates with {max_workers} workers.", flush=True)
        futures = [executor.submit(evaluate_code_candidate, candidate) for candidate in candidates]
        results = []
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(f"Completed {completed}/{len(futures)} candidates.", flush=True)
        return results





def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the qudit experiment."
    )

    parser.add_argument("--qudit", type=int, default=3)

    parser.add_argument("--ell", type=int, default=4)
    parser.add_argument("--m", type=int, default=3)

    parser.add_argument("--base-d", type=int, default=4, dest="base_d")

    parser.add_argument(
        "--A",
        type=str,
        default="1,0;2,0",
        help='Semicolon-separated tuples, e.g. "1,0;2,0"',
    )

    parser.add_argument(
        "--B",
        type=str,
        default="3,0;0,1;0,2",
        help='Semicolon-separated tuples, e.g. "3,0;0,1;0,2"',
    )

    parser.add_argument(
        "--factorA",
        type=int,
        nargs="+",
        default=[1, 1],
    )

    parser.add_argument(
        "--factorB",
        type=int,
        nargs="+",
        default=[2, 2, 2],
    )

    parser.add_argument("--u", type=int, default=3)
    parser.add_argument("--t", type=int, default=3)

    args = parser.parse_args()

    def parse_poly(s):
        return tuple(
            tuple(map(int, item.split(",")))
            for item in s.split(";")
        )

    args.A = parse_poly(args.A)
    args.B = parse_poly(args.B)

    return args

def random_ifnot_one(lis):
    if len(lis) == 1:
        return lis[0]
    else:
        return random.choice(lis)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_args()

    qudit = args.qudit
    ell = args.ell
    m = args.m
    base_d = args.base_d

    A = args.A
    B = args.B

    factorA = args.factorA
    factorB = args.factorB

    u_max = args.u
    t_max = args.t


    # check base code for connectivity:
    base_code_dict={"ell":ell,"m":m,"A":A,"B":B,"qudit":qudit,"factorA":factorA,"factorB":factorB,"gamma1":gamma1,"gamma2":gamma2,"delta1":delta1,"delta2":delta2, "final_depiction":None}
    Hx,Hz=get_bicycle_code(base_code_dict)
    n,K,HZ,HX,lz,lx=qudit_css_code(qudit,to_array(Hx),to_array(Hz))
    Tconn = graph_connectivity(Hx, Hz, HX, HZ)
    print_as_polynomial(factorA, A, "A")
    print_as_polynomial(factorB, B, "B")
    print(f"For base code: [[{n},{K},{base_d}]]_{qudit}, kd^2/n = {K*base_d**2/n:.3f}, k/2n = {K/(2*n):.3f}")
    if not Tconn:
        print("Not connected")

    number_found = 0
    store_code_dict = {}
    timeout = time.time() +10* 60*60
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    parallel_workers = max(1, min(allocated_cpus, 32))
    print(f"Number of parallel workers {parallel_workers}")
    while number_found < 100:
        if(time.time() > timeout):
            print("Timeout reached, stopping search and printing results to file.")
            break

        candidate_batch = []
        while len(candidate_batch) < parallel_workers:
            u = random.randint(2, u_max)
            t = random.randint(2, t_max)

            ai1 = bi1 = list(range(u - 1))
            ai2 = bi2 = list(range(t - 1))

            ell_mod = ell * u
            m_mod = m * t

            A_mod = [(a[0]+ell*random_ifnot_one(ai1), a[1]+m*random_ifnot_one(ai2)) for a in A]
            B_mod = [(b[0]+ell*random_ifnot_one(bi1), b[1]+m*random_ifnot_one(bi2)) for b in B]

            A_mod = tuple(prune.reduce_polynomial(A_mod))
            B_mod = tuple(prune.reduce_polynomial(B_mod))

            if((A_mod, B_mod) in store_code_dict or (B_mod, A_mod) in store_code_dict):
                print("Duplicate polynomial found, continuing..")
                continue
            print("Candidate polynomials for ell=", ell_mod, "m=", m_mod, "u=", u, "t=", t)
            print_as_polynomial(factorA, A_mod, "A")
            print_as_polynomial(factorB, B_mod, "B")
            print("\n")
            candidate_batch.append((A_mod, B_mod, ell_mod, m_mod, qudit, factorA, factorB, base_d))

        if not candidate_batch:
            continue
        print("Evaluating candidate polynomials: ")

        results = evaluate_candidates_in_parallel(candidate_batch, max_workers=parallel_workers)
        for result in results:
            print(f"Result: {result}")
            if result is None:
                continue
            (A_mod, B_mod), code_dict, _ = result
            store_code_dict[(A_mod, B_mod)] = code_dict
            number_found += 1
            append_code_to_csv(code_dict)




