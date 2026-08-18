"""qudit_bposd.py
Compatibility facade for qudit belief-propagation plus OSD decoding.

The implementation lives in :mod:`qudit_bp` and :mod:`qudit_osd`. This
module keeps the original ``quditBPOSD`` API used by older scripts while
ensuring BP, OSD, and the distance estimator share one implementation.
"""

import numpy as np

try:
    from .qudit_bp import (
        _check_prime,
        _exclude_one_convolution,
        _exclude_one_elementwise,
        bp_decode_qudit,
        symmetric_qudit_prior,
    )
    from .qudit_osd import _solution_from_pivots, most_reliable_basis, osd_decode_qudit
except ImportError:  # pragma: no cover - direct script execution
    from qudit_bp import (
        _check_prime,
        _exclude_one_convolution,
        _exclude_one_elementwise,
        bp_decode_qudit,
        symmetric_qudit_prior,
    )
    from qudit_osd import _solution_from_pivots, most_reliable_basis, osd_decode_qudit


def default_prior(n, q, p):
    """Backward-compatible name for the symmetric qudit prior."""
    return symmetric_qudit_prior(n, q, p)


class quditBPOSD:
    """Backward-compatible object interface for the qudit BP-OSD decoder."""

    def __init__(self, q, priors=None, max_iter=50):
        _check_prime(q)
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.q = int(q)
        self.priors = None if priors is None else np.asarray(priors, dtype=float)
        self.max_iter = int(max_iter)

    @staticmethod
    def elementwise(msgs):
        return _exclude_one_elementwise(msgs)

    def convolve(self, msgs, coeffs, paritycheck):
        return _exclude_one_convolution(msgs, coeffs, self.q, paritycheck)

    def bp_decode(self, H, s, priors=None, q=None, max_iter=None):
        if q is not None and int(q) != self.q:
            raise ValueError("q does not match the decoder dimension")
        H = np.asarray(H)
        selected_priors = self.priors if priors is None else np.asarray(priors, dtype=float)
        if selected_priors is None:
            selected_priors = symmetric_qudit_prior(H.shape[1], self.q, 0.1)
        return bp_decode_qudit(
            H,
            s,
            selected_priors,
            self.q,
            max_iter=self.max_iter if max_iter is None else max_iter,
        )

    def MRB(self, H, s, reliability, q=None):
        if q is not None and int(q) != self.q:
            raise ValueError("q does not match the decoder dimension")
        return most_reliable_basis(H, s, reliability, self.q)

    def sol_from_pivots(self, n, q, pivot_cols, pivot_rows, M, nonpivot_values=None):
        if int(q) != self.q:
            raise ValueError("q does not match the decoder dimension")
        return _solution_from_pivots(n, self.q, pivot_cols, pivot_rows, M, nonpivot_values)

    def osd_decode(self, H, s, reliability, q=None, order=0, weight_cost=None):
        if q is not None and int(q) != self.q:
            raise ValueError("q does not match the decoder dimension")
        return osd_decode_qudit(H, s, reliability, self.q, order=order, weight_cost=weight_cost)

    def distance_upper_bound_bposd_qudit(
        self,
        stab,
        logicOp,
        num_trials=200,
        max_iter=None,
        osd_order=0,
        p_search=0.5,
        max_time=None,
        verbose=False,
    ):
        try:
            from .bposddistance import distance_upper_bound_bposd_qudit
        except ImportError:  # pragma: no cover - direct script execution
            from bposddistance import distance_upper_bound_bposd_qudit
        return distance_upper_bound_bposd_qudit(
            stab,
            logicOp,
            self.q,
            num_trials=num_trials,
            max_iter=self.max_iter if max_iter is None else max_iter,
            osd_order=osd_order,
            p_search=p_search,
            max_time=max_time,
            verbose=verbose,
        )
