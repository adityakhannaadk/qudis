"""qudit_osd.py
A handwritten description of OSD is given in "bposd.pdf",
however there is essentially no difference between the qubit and qudit versions, except that the qudit version uses galois.FieldArray for finite-field arithmetic.
"""

import itertools
import numpy as np
from sympy import isprime

try:
    import galois
except Exception as e:
    raise ImportError("The 'galois' package is required for this module. Install it with 'pip install galois'.") from e


def _check_prime(q):
    if not isinstance(q, (int, np.integer)) or q < 2 or not isprime(int(q)):
        raise ValueError(f"q={q} is not prime; see qudit_bp.py docstring.")


def most_reliable_basis(H, s, reliability, q):
    """
    Gauss-Jordan elimination on H (over GF(q)) visiting columns in order of
    decreasing reliability. Returns pivot column list, pivot row list and the
    row-reduced augmented matrix as a `galois.FieldArray`.
    """
    _check_prime(q)
    H = np.asarray(H, dtype=int)
    s = np.asarray(s, dtype=int)
    reliability = np.asarray(reliability, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be a two-dimensional matrix")
    m, n = H.shape
    if s.shape != (m,):
        raise ValueError("s must have one entry per check row")
    if reliability.shape != (n,):
        raise ValueError("reliability must have one entry per variable")
    if not np.all(np.isfinite(reliability)):
        raise ValueError("reliability must contain finite values")

    GF = galois.GF(q)
    M = np.concatenate([H % q, (s % q).reshape(-1, 1)], axis=1)
    M = GF(M)
    order = np.argsort(-reliability, kind="stable")

    used_rows = set()
    pivot_cols = []
    pivot_rows = []
    for col in order:
        candidates = [r for r in range(m) if r not in used_rows and M[r, col] != 0]
        if not candidates:
            continue
        r = candidates[0]
        # normalize pivot row.  Both the numerator and denominator must be
        # galois field elements, not plain Python ints, otherwise division
        # raises the TypeError seen in the BP-OSD retry path.
        inv = GF(1) / M[r, col]
        M[r, :] = M[r, :] * inv
        # eliminate other rows
        for r2 in range(m):
            if r2 != r and M[r2, col] != 0:
                M[r2, :] = M[r2, :] - M[r2, col] * M[r, :]
        used_rows.add(r)
        pivot_cols.append(int(col))
        pivot_rows.append(int(r))
        if len(pivot_cols) == m:
            break

    rank = len(pivot_cols)
    if rank < m:
        for row in range(m):
            if np.all(M[row, :n] == 0) and M[row, -1] != 0:
                raise ValueError("syndrome is inconsistent with H")

    return pivot_cols, pivot_rows, M


def _solution_from_pivots(n, q, pivot_cols, pivot_rows, M, nonpivot_values=None):
    """Build a full length-n solution vector over integers in [0, q).

    M is expected to be a `galois.FieldArray` representing the row-reduced
    augmented matrix [H | s].
    """
    e = np.zeros(n, dtype=np.int64)
    GF = galois.GF(q)
    if nonpivot_values is not None:
        for col, val in nonpivot_values.items():
            e[col] = int(val) % q

    rhs = M[:, -1].copy()
    if nonpivot_values is not None:
        for col, val in nonpivot_values.items():
            rhs = rhs - M[:, col] * GF(int(val))

    for col, row in zip(pivot_cols, pivot_rows):
        e[col] = int(rhs[row]) % q

    return e % q


def osd_decode_qudit(H, s, reliability, q, order=0, weight_cost=None):
    """
    OSD-w decoding of H e = s (mod q), q prime. Uses galois.FieldArray for
    finite-field arithmetic; behaviour matches the original implementation.
    """
    _check_prime(q)
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("order must be a non-negative integer")
    H = np.asarray(H, dtype=int) % q
    s = np.asarray(s, dtype=int) % q
    if H.ndim != 2:
        raise ValueError("H must be a two-dimensional matrix")
    m, n = H.shape

    pivot_cols, pivot_rows, M = most_reliable_basis(H, s, reliability, q)

    best_e = _solution_from_pivots(n, q, pivot_cols, pivot_rows, M)
    if weight_cost is None:
        def candidate_cost(candidate):
            return int(np.count_nonzero(candidate))
    else:
        weight_cost = np.asarray(weight_cost, dtype=float)
        if weight_cost.shape != (n,) or not np.all(np.isfinite(weight_cost)):
            raise ValueError("weight_cost must be a finite array of shape (n,)")
        if np.any(weight_cost < 0):
            raise ValueError("weight_cost must be non-negative")

        def candidate_cost(candidate):
            return float(np.sum(weight_cost[candidate != 0]))

    best_cost = candidate_cost(best_e)

    if order > 0:
        pivot_set = set(pivot_cols)
        order_full = np.argsort(-reliability, kind="stable")
        nonpivot_cols_by_reliability = [c for c in order_full if c not in pivot_set]
        reprocess_cols = nonpivot_cols_by_reliability[:order]

        for combo in itertools.product(range(q), repeat=len(reprocess_cols)):
            if all(v == 0 for v in combo):
                continue
            nonpivot_values = dict(zip(reprocess_cols, combo))
            e = _solution_from_pivots(n, q, pivot_cols, pivot_rows, M, nonpivot_values)
            cost = candidate_cost(e)
            if cost < best_cost:
                best_cost = cost
                best_e = e

    return best_e, True