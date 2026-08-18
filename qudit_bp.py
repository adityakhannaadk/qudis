"""qudit_bp.py
A handwritten description of BP is given in "bposd.pdf" 
There are lots of potential improvements to this algorithm,
and I would eventually like to rewrite it in C++/Rust, but
algorithmic improvements to begin with"""
import numpy as np
from sympy import isprime


def _check_prime(q):
    if not isinstance(q, (int, np.integer)) or q < 2:
        raise ValueError(f"q={q!r} must be a prime integer")
    if not isprime(q):
        raise ValueError(
            f"q={q} is not prime. This decoder (and the original ILP code) "
            f"only implements the case where GF(q) coincides with integers "
            f"mod q. Prime-power qudits (4, 8, 9, ...) need proper GF(p^m) "
            f"field arithmetic, not plain mod-q arithmetic."
        )

# This should be modified for more complicated noise models, but for now we just use a symmetric qudit-flip channel
def symmetric_qudit_prior(n, q, p):
    """
    Prior distribution per qudit under the symmetric qudit-flip channel:
    P(no error) = 1-p, P(error to any of the q-1 nonzero shifts) = p/(q-1) each.
    Returns an (n, q) array, row j = prior distribution for qudit j.
    """
    # np.full creates an array of shape (n, q) filled with the value p/(q-1)
    _check_prime(q)
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    prior = np.full((n, q), p / (q - 1), dtype=float)
    prior[:, 0] = 1 - p
    return prior


def _exclude_one_elementwise(msgs):
    """
    msgs: (d, q) array of d probability vectors (rows sum to 1).
    Returns (d, q) array where row i = elementwise product of all rows
    except row i, renormalised. Uses prefix/suffix running products so
    it costs O(d*q) instead of O(d^2*q), same trick as bp.h.
    """
    msgs = np.asarray(msgs, dtype=float)
    if msgs.ndim != 2:
        raise ValueError("messages must have shape (degree, q)")
    d, q = msgs.shape
    prefix = np.ones((d + 1, q))
    for i in range(d):
        prefix[i + 1] = prefix[i] * msgs[i]
    suffix = np.ones((d + 1, q))
    for i in range(d - 1, -1, -1):
        suffix[i] = suffix[i + 1] * msgs[i]
    out = prefix[:d] * suffix[1:]
    row_sums = out.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] == 0
    row_sums[row_sums == 0] = 1.0
    out /= row_sums
    if np.any(zero_rows):
        out[zero_rows] = 1.0 / q
    return out


def _exclude_one_convolution(msgs, coeffs, q, syndrome_bit):
    """
    The check-node update. msgs: (d,q) incoming variable->check distributions
    for the d neighbours of this check, in the SAME order as coeffs (the
    nonzero H-row entries for those neighbours). Returns (d,q) array: row i
    is the outgoing check->variable message to neighbour i.

    Model: neighbour i contributes f_i = coeffs[i] * e_i (mod q). The check
    enforces sum_i f_i = syndrome_bit (mod q). The message to neighbour i is
    the distribution of e_i implied by "everyone else's f, plus the
    syndrome, mod q".
    """
    msgs = np.asarray(msgs, dtype=float)
    coeffs = np.asarray(coeffs, dtype=int) % q
    if msgs.ndim != 2 or msgs.shape[1] != q:
        raise ValueError("messages must have shape (degree, q)")
    if coeffs.ndim != 1 or coeffs.shape[0] != msgs.shape[0]:
        raise ValueError("coeffs must match the message degree")
    if np.any(coeffs == 0):
        raise ValueError("check-node coefficients must be nonzero")
    d, _ = msgs.shape
    syndrome_bit = int(syndrome_bit) % q
    inv = [pow(int(c), -1, q) for c in coeffs]

    # permute each incoming distribution: g_i[k] = msgs[i][ inv[i]*k mod q ]
    idx = (np.outer(inv, np.arange(q)) % q)
    g = np.take_along_axis(msgs, idx.astype(int), axis=1)

    # circular convolution of all-but-one, via FFT prefix/suffix products
    G = np.fft.fft(g, axis=1)
    prefix = np.ones((d + 1, q), dtype=complex)
    for i in range(d):
        prefix[i + 1] = prefix[i] * G[i]
    suffix = np.ones((d + 1, q), dtype=complex)
    for i in range(d - 1, -1, -1):
        suffix[i] = suffix[i + 1] * G[i]
    combined_F = prefix[:d] * suffix[1:]
    h = np.fft.ifft(combined_F, axis=1).real  # (d,q), h[i] = dist of sum_{j!=i} f_j
    h = np.clip(h, 0, None)
    h_sums = h.sum(axis=1, keepdims=True)
    zero_rows = h_sums[:, 0] == 0
    h_sums[h_sums == 0] = 1.0
    h = h / h_sums
    if np.any(zero_rows):
        h[zero_rows] = 1.0 / q

    # message to neighbour i: mu[e] = h_i[ (syndrome_bit - coeffs[i]*e) mod q ]
    out = np.zeros((d, q))
    for i in range(d):
        e_vals = np.arange(q)
        idx_i = (syndrome_bit - coeffs[i] * e_vals) % q
        out[i] = h[i][idx_i]
        s = out[i].sum()
        if s > 0:
            out[i] /= s
    return out


def bp_decode_qudit(H, s, priors, q, max_iter=50):
    """
    Non-binary sum-product BP decoding of He = s (mod q), q prime.

    H: (m,n) int array, entries in 0..q-1.
    s: (m,) int array, target syndrome, entries in 0..q-1.
    priors: (n,q) array of per-qudit prior distributions (see
        symmetric_qudit_prior).
    Returns: (decoding, marginals, converged)
        decoding: (n,) hard-decision error pattern.
        marginals: (n,q) final posterior distributions (used as OSD
            reliabilities).
        converged: bool, whether H @ decoding == s (mod q).
    """

    # Error messgaes etc.
    # all mod q, so we can check that q is prime and then reduce everything mod q.
    _check_prime(q)
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    H = np.asarray(H)
    s = np.asarray(s)
    priors = np.asarray(priors, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be a two-dimensional matrix")
    if s.ndim != 1 or s.shape[0] != H.shape[0]:
        raise ValueError("s must have one entry per check row")
    if priors.shape != (H.shape[1], q):
        raise ValueError("priors must have shape (H.shape[1], q)")
    if not np.all(np.isfinite(priors)) or np.any(priors < 0):
        raise ValueError("priors must be finite and non-negative")
    prior_sums = priors.sum(axis=1, keepdims=True)
    if np.any(prior_sums <= 0):
        raise ValueError("each prior row must have positive mass")


    priors = priors / prior_sums
    H = H.astype(np.int64) % q
    s = s.astype(np.int64) % q
    m, n = H.shape

    # Tanner-graph adjacency: for each check, its nonzero (col, coeff) list;
    # for each variable, its nonzero (row, coeff) list.
    check_nbrs = [np.nonzero(H[c])[0] for c in range(m)]
    var_nbrs = [np.nonzero(H[:, v])[0] for v in range(n)]

    # bit_to_check[v] holds, for each check c in var_nbrs[v] (same order),
    # the current outgoing variable->check message (a length-q vector).
    bit_to_check = [np.tile(priors[v], (len(var_nbrs[v]), 1)) for v in range(n)]
    # check_to_bit[c] holds, for each var v in check_nbrs[c], the current
    # outgoing check->variable message.
    check_to_bit = [np.full((len(check_nbrs[c]), q), 1.0 / q) for c in range(m)]

    decoding = np.zeros(n, dtype=int)
    marginals = priors.copy()
    converged = False

    for _ in range(max_iter):
        # --- check-to-variable update ---
        new_check_to_bit = []
        for c in range(m):
            nbrs = check_nbrs[c]
            coeffs = H[c, nbrs]
            # gather the current variable->check messages arriving at this check,
            # in nbrs order
            msgs = np.array([
                bit_to_check[v][np.where(var_nbrs[v] == c)[0][0]] for v in nbrs
            ])
            new_check_to_bit.append(_exclude_one_convolution(msgs, coeffs, q, s[c]))
        check_to_bit = new_check_to_bit

        # --- posterior marginals + hard decision ---
        for v in range(n):
            nbrs = var_nbrs[v]
            incoming = np.array([
                check_to_bit[c][np.where(check_nbrs[c] == v)[0][0]] for c in nbrs
            ]) if len(nbrs) else np.zeros((0, q))
            post = priors[v].copy()
            for row in incoming:
                post = post * row
            s_sum = post.sum()
            if s_sum > 0:
                post = post / s_sum
            marginals[v] = post
            decoding[v] = int(np.argmax(post))

        # H @ decoding == s (mod q) check: if satisfied, we can stop early
        # What does H @ decoding mean? What does @ do? It is matrix multiplication. 
        # So H @ decoding is the matrix multiplication of H and decoding. 
        # The result is a vector of length m, 
        # where each entry is the sum of the products of the corresponding row of H and the decoding vector. 
        # This result is then taken modulo q to check if it equals the syndrome vector s.
        # If they are equal, it means that the current decoding satisfies the parity check equations defined by H and s, 
        # and we can stop the belief propagation algorithm early since we have found a valid solution.
        if np.array_equal((H @ decoding) % q, s):
            converged = True
            break

        # --- variable-to-check update (exclude recipient) ---
        for v in range(n):
            nbrs = var_nbrs[v]
            if len(nbrs) == 0:
                continue
            incoming = np.array([
                check_to_bit[c][np.where(check_nbrs[c] == v)[0][0]] for c in nbrs
            ])
            excl = _exclude_one_elementwise(incoming)
            out = priors[v][None, :] * excl
            row_sums = out.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            bit_to_check[v] = out / row_sums

    return decoding, marginals, converged