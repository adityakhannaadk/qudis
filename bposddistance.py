"""
bposddistance.py
A handwritten derivation of the BP-OSD distance upper bound is given in "distances.pdf".
This is very simple, the main confusion is why we can set the final value of the augmented 
matrix to be 1.
"""
import time
import numpy as np
from sympy import isprime

try:
    from .qudit_bp import bp_decode_qudit, symmetric_qudit_prior
    from .qudit_osd import osd_decode_qudit
except ImportError:  # pragma: no cover - direct script execution
    from qudit_bp import bp_decode_qudit, symmetric_qudit_prior
    from qudit_osd import osd_decode_qudit


def distance_upper_bound_bposd_qudit(
    stab,
    logicOp,
    qudit,
    verbose=True,
    num_trials=5,
    max_iter=30,
    osd_order=0,
    p_search=0.5,
    max_time=100
):
    """
    BP-OSD based upper bound on the minimum weight of the logical class of
    `logicOp`, for a qudit stabilizer code over GF(qudit), qudit PRIME.

    stab: (m, n) array, stabilizer check matrix (rows mod qudit).
    logicOp: (n,) array, one representative logical operator.
    qudit: prime dimension q.
    num_trials: number of random restarts (random stabilizer offset +
        random BP priors each trial); the returned bound is the minimum
        weight found across all trials.
    max_iter: BP iterations per trial.
    osd_order: OSD reprocessing order (see qudit_osd.osd_decode_qudit;
        cost grows like q^osd_order, keep small).
    p_search: mean "error rate" used to generate random per-trial channel
        priors -- this has no physical meaning here (there's no real
        channel), it just controls how much BP trusts the all-zero
        default vs. exploring larger patterns. Values around 0.3-0.6 tend
        to give the most diverse candidate corrections.
    max_time: optional wall-clock budget in seconds; stops early (still
        returns the best bound found so far) if exceeded.

    Returns: (best_weight, best_e)
        best_weight: int, the smallest weight found (an UPPER BOUND on the
            true minimum weight of this logical class -- never a proof of
            optimality).
        best_e: the corresponding error pattern.
    """
    if not isinstance(qudit, (int, np.integer)) or not isprime(int(qudit)):
        raise ValueError(
            f"qudit={qudit} is not prime; see qudit_bp.py docstring for why "
            f"this implementation (like the original ILP code's mod-qudit "
            f"arithmetic) requires it."
        )

    if not isinstance(num_trials, (int, np.integer)) or num_trials < 0:
        raise ValueError("num_trials must be a non-negative integer")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(osd_order, (int, np.integer)) or osd_order < 0:
        raise ValueError("osd_order must be a non-negative integer")
    if not np.isfinite(p_search) or not 0 <= p_search <= 1:
        raise ValueError("p_search must be between 0 and 1")
    if max_time is not None and (not np.isfinite(max_time) or max_time < 0):
        raise ValueError("max_time must be non-negative or None")

    print("Running BP-OSD distance upper bound search with parameters:")
    print(f"  qudit: {qudit}")
    print(f"  num_trials: {num_trials}")
    print(f"  max_iter: {max_iter}")
    print(f"  osd_order: {osd_order}")
    print(f"  p_search: {p_search}")
    print(f"  max_time: {max_time}")
    qudit = int(qudit)
    stab = np.asarray(stab)
    logicOp = np.asarray(logicOp)
    if stab.ndim != 2:
        raise ValueError("stab must be a two-dimensional matrix")
    if logicOp.shape != (stab.shape[1],):
        raise ValueError("logicOp must have one entry per qudit")
    stab = stab.astype(np.int64) % qudit
    logicOp = logicOp.astype(np.int64) % qudit
    m, n = stab.shape
    assert logicOp.shape[0] == n

    start_time = time.time()
    best_weight = np.inf
    best_e = None

    rng = np.random.default_rng()
    for trial in range(num_trials):
        if max_time is not None and time.time() - start_time >= max_time:
            break

        print(f"[trial {trial}] Starting trial {trial + 1}/{num_trials}")

        # randomise the logical representative by a random stabilizer
        # applied per trial to diversify which representative of the
        # (large, degenerate) logical coset BP-OSD is asked to find.
        stab_coeffs = rng.integers(0, qudit, size=m)
        ell = (logicOp + stab_coeffs @ stab) % qudit

        # Build the augmented system [stab ; ell] with syndrome [0 ... 0, 1].
        # This makes the decoder target a representative of the logical class
        # rather than a stabilizer.
        H_aug = np.vstack([stab, ell])
        s_aug = np.zeros(m + 1, dtype=int)
        s_aug[-1] = 1

        # random per-trial priors: diversifies which low-weight
        # representative BP's message passing tends to settle into.
        p_trial = rng.uniform(max(0.05, p_search - 0.25), min(0.9, p_search + 0.25))
        priors = symmetric_qudit_prior(n, qudit, p_trial)

        try:
            decoding, marginals, converged = bp_decode_qudit(
                H_aug, s_aug, priors, qudit, max_iter=max_iter
            )
        except Exception as exc:
            if verbose:
                print(f"[trial {trial}] BP decode raised {type(exc).__name__}: {exc}; skipping trial.")
            continue

        if converged:
            e = decoding
            if verbose:
                print(f"[trial {trial}] BP converged to a valid solution")
        else:
            reliability = marginals.max(axis=1)
            if verbose:
                print(f"[trial {trial}] BP did not converge, running OSD order {osd_order}")
            try:
                e, _ = osd_decode_qudit(
                    H_aug, s_aug, reliability, qudit, order=osd_order
                )
                if verbose:
                    print(f"[trial {trial}] OSD found a valid solution")
            except Exception as exc:
                # The chosen logical representative may be dependent on the
                # stabilizers, so this augmented syndrome can be impossible.
                if verbose:
                    print(f"[trial {trial}] OSD failed: {type(exc).__name__}: {exc}; skipping trial.")
                continue

        # sanity: e must actually satisfy the augmented system (guaranteed
        # for the OSD branch by construction; check anyway for the BP branch
        # and as a general safety net)
        if not np.array_equal((H_aug @ e) % qudit, s_aug):
            if verbose:
                print(f"[trial {trial}] Decode failed the augmented-syndrome sanity check; skipping trial.")
            continue

        w = int(np.count_nonzero(e))
        if verbose:
            print(f"[trial {trial}] p={p_trial:.2f} converged={converged} weight={w}")

        if w < best_weight:
            best_weight = w
            best_e = e.copy()

    if best_weight == np.inf:
        if verbose:
            print("No valid BP-OSD decode found in any trial; returning (None, None)")
        return None, None

    return int(best_weight), best_e