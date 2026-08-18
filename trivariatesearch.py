import argparse
import csv
import os
import random
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


from logical import get_trivariate_bicycle_code, qudit_css_code
from ilpdistance import distance_test_qudit as get_qudit_distance
from bposddistance import heuristic_distance_test_qudit


def random_monomial(ell, m, s):
    return (random.randrange(ell), random.randrange(m), random.randrange(s))


def random_polynomial(term_count, ell, m, s):
    terms = set()
    while len(terms) < term_count:
        terms.add(random_monomial(ell, m, s))
    return list(terms)


def evaluate_trivariate_candidate(start, code_dict, max_time):
    HX, HZ = get_trivariate_bicycle_code(code_dict)
    n, K, HZ_c, HX_c, lz, lx = qudit_css_code(code_dict["qudit"], HX, HZ)

    if K <= 0:
        return {"code_dict": code_dict, "valid": False, "reason": "K=0"}

    lz = np.array(lz, dtype=np.int8)
    lx = np.array(lx, dtype=np.int8)

    distances_z = []
    distances_x = []
    rng = np.random.default_rng()

    for i in range(K):
        stab_coeffs_z = rng.integers(0, code_dict["qudit"], size=HZ_c.shape[0])
        lz_randomized = (lz[i, :] + stab_coeffs_z @ HZ_c) % code_dict["qudit"]

        try:
            upper_bound, heuristic_solution = heuristic_distance_test_qudit(HZ_c, lz_randomized, code_dict["qudit"])
            if upper_bound is None:
                print("Heuristic BP-OSD produced no valid upper bound; skipping exact distance solve for this logical operator.")
                return {"code_dict": code_dict, "valid": False, "reason": "heuristic_upper_bound_unavailable", "lz": lz[i, :], "HZ_c": HZ_c}
            print(f"Using heuristic upper bound {upper_bound} for exact distance test")
            w_z = get_qudit_distance(HZ_c, lz_randomized, code_dict["qudit"], ub=upper_bound, heuristic_solution=heuristic_solution)
        except Exception as exc:
            return {"code_dict": code_dict, "valid": False, "reason": f"distance_z_failed: {str(exc)}", "lz": lz[i, :], "HZ_c": HZ_c}
        if w_z <= 0:
            return {"code_dict": code_dict, "valid": False, "reason": "distance_z_failed"}
        distances_z.append(w_z)

        stab_coeffs_x = rng.integers(0, code_dict["qudit"], size=HX_c.shape[0])
        lx_randomized = (lx[i, :] + stab_coeffs_x @ HX_c) % code_dict["qudit"]

        try:
            upper_bound, heuristic_solution = heuristic_distance_test_qudit(HX_c, lx_randomized, code_dict["qudit"])
            if upper_bound is None:
                print("Heuristic BP-OSD produced no valid upper bound; skipping exact distance solve for this logical operator.")
                return {"code_dict": code_dict, "valid": False, "reason": "heuristic_upper_bound_unavailable", "lx": lx[i, :], "HX_c": HX_c}
            print(f"Using heuristic upper bound {upper_bound} for exact distance test")
            w_x = get_qudit_distance(HX_c, lx_randomized, code_dict["qudit"], ub=upper_bound, heuristic_solution=heuristic_solution)
        except Exception as exc:
            return {"code_dict": code_dict, "valid": False, "reason": f"distance_x_failed: {str(exc)}", "lx": lx[i, :], "HX_c": HX_c}
        if w_x <= 0:
            return {"code_dict": code_dict, "valid": False, "reason": "distance_x_failed"}
        distances_x.append(w_x)

    return {
        "code_dict": code_dict,
        "valid": True,
        "n": n,
        "K": K,
        "distance_z": min(distances_z),
        "distance_x": min(distances_x),
        "distance": min(min(distances_z), min(distances_x)),
        "distances_z": distances_z,
        "distances_x": distances_x,
    }

def search_random_trivariate_codes(
    qudit,
    ell,
    m,
    s,
    tries,
    max_terms_A,
    max_terms_B,
    min_terms_A,
    min_terms_B,
    max_n,
    max_time,
    seed=None,
):
    random.seed(seed)

    n = 2 * ell * m * s
    if n >= max_n:
        raise ValueError(f"Requested code size n={n} is not less than max_n={max_n}")

    found = []
    seen = set()

    for i in range(tries):
        term_count_A = random.randint(min_terms_A, max_terms_A)
        term_count_B = random.randint(min_terms_B, max_terms_B)
        A = tuple(sorted(random_polynomial(term_count_A, ell, m, s)))
        B = tuple(sorted(random_polynomial(term_count_B, ell, m, s)))
        key = (A, B)
        if key in seen:
            continue
        seen.add(key)

        code_dict = {
            "ell": ell,
            "m": m,
            "s": s,
            "qudit": qudit,
            "A": list(A),
            "B": list(B),
            "factorA": [1] * len(A),
            "factorB": [1] * len(B),
            "gamma1": 1,
            "gamma2": 1,
            "delta1": 1,
            "delta2": -1,
        }

        result = evaluate_trivariate_candidate_randomstab(code_dict, max_time=max_time)
        if result.get("valid"):
            found.append(result)
            print(
                f"FOUND code n={result['n']} K={result['K']} d={result['distance']} "
                f"(dz={result['distance_z']}, dx={result['distance_x']}, kd^2/n = {result['K']*result['distance']**2/n:.3f}), A={code_dict['A']}, B={code_dict['B']}"
            )
            
        else:
            print(f"skip [{i+1}/{tries}] reason={result.get('reason') or result.get('error')}" )

    return found


def save_results(results, csv_path="found_trivariate_codes.csv"):
    header = [
        "n",
        "K",
        "distance",
        "distance_z",
        "distance_x",
        "ell",
        "m",
        "s",
        "qudit",
        "A",
        "B",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for entry in results:
            code = entry["code_dict"]
            writer.writerow(
                [
                    entry["n"],
                    entry["K"],
                    entry["distance"],
                    entry["distance_z"],
                    entry["distance_x"],
                    code["ell"],
                    code["m"],
                    code["s"],
                    code["qudit"],
                    code["A"],
                    code["B"],
                    ";".join(
                        ",".join(map(str, monomial)) for monomial in code["A"]
                    ),
                    ";".join(
                        ",".join(map(str, monomial)) for monomial in code["B"]
                    ),
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Random search for trivariate bicycle qudit codes")
    parser.add_argument("--qudit", type=int, default=2)
    parser.add_argument("--ell", type=int, default=3)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--s", type=int, default=3)
    parser.add_argument("--tries", type=int, default=1000)
    parser.add_argument("--max-terms-A", type=int, default=3)
    parser.add_argument("--max-terms-B", type=int, default=3)
    parser.add_argument("--min-terms-A", type=int, default=1)
    parser.add_argument("--min-terms-B", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=100)
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="found_trivariate_codes_qubit.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"searching trivariate codes q={args.qudit} ell={args.ell} m={args.m} s={args.s} "
        f"n<= {args.max_n}, tries={args.tries}"
    )

    results = search_random_trivariate_codes(
        qudit=args.qudit,
        ell=args.ell,
        m=args.m,
        s=args.s,
        tries=args.tries,
        max_terms_A=args.max_terms_A,
        max_terms_B=args.max_terms_B,
        min_terms_A=args.min_terms_A,
        min_terms_B=args.min_terms_B,
        max_n=args.max_n,
        max_time=args.max_time,
        seed=args.seed,
    )

    if results:
        save_results(results, csv_path=args.output)
        print(f"Saved {len(results)} valid codes to {args.output}")
    else:
        print("No valid codes found")


if __name__ == "__main__":
    main()

