from header import *
from helpers import *
from line_info import *
from plot import *
from plot import _collect_voigt_components, _find_lsf_component, _collect_continuum_components
import time
import itertools
import numpy as np
import re
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Student-t likelihood helpers
# ---------------------------------------------------------------------------

def student_t_logpdf_sum(resid: np.ndarray, sigma: np.ndarray, nu: float) -> float:
    r"""Sum of Student-t log-pdfs for an array of residuals.

    .. math::
        \ln t_\nu\!\bigl(\frac{d_i - m_i}{\sigma_i}\bigr)
        = \ln\Gamma\!\bigl(\tfrac{\nu+1}{2}\bigr)
          - \ln\Gamma\!\bigl(\tfrac{\nu}{2}\bigr)
          - \tfrac{1}{2}\ln(\nu\pi)
          - \ln\sigma_i
          - \tfrac{\nu+1}{2}\ln\!\bigl(1 + r_i^2/\nu\bigr)

    Parameters
    ----------
    resid : array  (data - model)
    sigma : array  (measurement uncertainties, same length)
    nu : float  degrees of freedom (>0). As nu -> inf, converges to Gaussian.

    Returns
    -------
    float  total log-likelihood
    """
    r = resid / sigma
    norm = (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
    )
    return float(np.sum(norm - np.log(sigma) - 0.5 * (nu + 1.0) * np.log(1.0 + r * r / nu)))


def gaussian_logpdf_sum(resid: np.ndarray, inv_sigma2: np.ndarray, log_norm: np.ndarray) -> float:
    """Standard Gaussian log-likelihood (precomputed normalisations)."""
    return float(-0.5 * np.sum(resid * resid * inv_sigma2 + log_norm))


def build_student_t_mask(
    spectrum_wave: np.ndarray,
    regions: "Mapping[str, Mapping]",
    search_lines,
    z: float,
) -> np.ndarray:
    r"""Return a boolean mask over *spectrum_wave* that is ``True`` for
    pixels falling inside any Student-t wavelength region.

    Parameters
    ----------
    spectrum_wave : 1-D array (Angstrom, observed frame)
    regions : dict
        Mapping from SEARCH_LINES ``tempname`` to a dict with at least
        ``"vrange"`` (tuple of Quantity, e.g. ``(-600*u.km/u.s, 600*u.km/u.s)``).
        Example::

            {"OVI_1": {"vrange": (-600*u.km/u.s, 600*u.km/u.s)},
             "OVI_2": {"vrange": (-600*u.km/u.s, 600*u.km/u.s)}}
    search_lines : astropy.table.Table
        The SEARCH_LINES table.
    z : float
        Absorber redshift.

    Returns
    -------
    1-D bool array, same length as *spectrum_wave*.
    """
    from astropy import units as u
    mask = np.zeros(len(spectrum_wave), dtype=bool)
    for line_key, cfg in regions.items():
        vr = cfg["vrange"]
        # Accept either:
        #   1) a single interval tuple/list: (vlo, vhi)
        #   2) a list of interval tuples/lists: [(vlo1, vhi1), (vlo2, vhi2), ...]
        if isinstance(vr, (list, tuple)) and len(vr) == 2 and not isinstance(vr[0], (list, tuple)):
            vranges = [tuple(vr)]
        elif isinstance(vr, (list, tuple)):
            vranges = list(vr)
        else:
            raise TypeError(
                f"Invalid vrange for line '{line_key}': expected (vlo, vhi) or list of such pairs, got {type(vr)}"
            )
        info = search_lines[search_lines["tempname"] == line_key]
        if len(info) == 0:
            continue
        lam0 = info["wave"][0]
        lam0_val = lam0.to_value(u.Angstrom) if hasattr(lam0, "to_value") else float(lam0)
        lam_obs = lam0_val * (1.0 + z)
        c_kms = 299792.458
        for interval in vranges:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise TypeError(
                    f"Invalid vrange interval for line '{line_key}': expected pair (vlo, vhi), got {interval!r}"
                )
            vlo, vhi = interval
            vlo_val = vlo.to_value(u.km / u.s) if hasattr(vlo, "to_value") else float(vlo)
            vhi_val = vhi.to_value(u.km / u.s) if hasattr(vhi, "to_value") else float(vhi)
            wlo = lam_obs * (1.0 + vlo_val / c_kms)
            whi = lam_obs * (1.0 + vhi_val / c_kms)
            mask |= (spectrum_wave >= wlo) & (spectrum_wave <= whi)
    return mask

def _as_flat_chain(sampler_or_chain, burnin=0, verbose=False):
    """
    Normalize input into a flat MCMC chain of shape (n_samples, n_params).

    Accepts either:
      - an emcee.EnsembleSampler (uses get_chain(discard=burnin, flat=True)), or
      - a numpy-like array. If 3D, it's flattened; if 2D, returned as-is.

    Notes:
      - If an array is provided, 'burnin' is ignored (assumed already applied).
    """
    # emcee sampler case
    if hasattr(sampler_or_chain, "get_chain"):
        arr = sampler_or_chain.get_chain(discard=burnin, flat=True)
        if verbose:
            print("_as_flat_chain: extracted flat chain from sampler:", np.asarray(arr).shape)
        return np.asarray(arr)

    # array-like case
    arr = np.asarray(sampler_or_chain)
    if arr.ndim == 3:
        # flatten steps and walkers
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim != 2:
        raise ValueError("Expected a 2D or 3D array for chain, or an emcee sampler object")
    if burnin and verbose:
        print("_as_flat_chain: 'burnin' ignored for provided array input")
    if verbose:
        print("_as_flat_chain: using provided flat chain:", arr.shape)
    return arr


def _prepare_chain_and_log_prob(sampler_or_chain, log_probs=None, burnin=0, verbose=False):
    """Return flattened chain and matching log-probability array."""

    chain = _as_flat_chain(sampler_or_chain, burnin=burnin, verbose=verbose)

    if log_probs is None:
        if hasattr(sampler_or_chain, "get_log_prob"):
            log_arr = sampler_or_chain.get_log_prob(discard=burnin, flat=True)
        else:
            raise ValueError("log_probs must be provided when sampler lacks get_log_prob")
    else:
        log_arr = np.asarray(log_probs).ravel()

    if chain.shape[0] != log_arr.shape[0]:
        raise ValueError(
            f"Chain length {chain.shape[0]} does not match log_probs length {log_arr.shape[0]}"
        )

    mask = np.isfinite(log_arr)
    if verbose and not np.all(mask):
        print(f"_prepare_chain_and_log_prob: dropping {mask.size - np.count_nonzero(mask)} non-finite log_prob entries")

    return chain[mask], log_arr[mask]

def genSamples(fit, num, flat_params=None, triangle_params=None,
               param_constraints=None, verbose=False, max_iter=50,
               order_alpha=3.0):
    """
    Generate Monte Carlo samples from a fitted model.

    Parameters:
    - fit (FittedModel): Fitted model.
    - num (int): Number of samples to generate.
    - flat_params (list of str, optional): parameter names to sample from flat prior only.
    - triangle_params (list of str, optional): parameter names to draw from triangular priors.
    - param_constraints (list of str, optional): List of constraints like 'paramA>paramB' or 'paramA < paramB' to enforce ordering.
    - verbose (bool): If True, print debug information.
    - order_alpha (float): Dirichlet concentration parameter for ordered velocity blocks.

    Returns:
    - samples (array): Monte Carlo samples from the fitted model.
    """
    if verbose:
        print("Entering genSamples")
        print(f"Requested samples: {num}")
        print(f"Flat params: {flat_params}")
        print(f"Param constraints: {param_constraints}")

    # Ensure flat_params and param_constraints lists
    if flat_params is None:
        flat_params = []
    if param_constraints is None:
        param_constraints = []
    if triangle_params is None:
        triangle_params = []

    # Get parameter names and means
    param_names = list(fit.cov_matrix.param_names)
    if verbose:
        print("Parameter names:", param_names)

    means = np.array([
        fit.parameters[np.where(np.array(fit.param_names, dtype=str) == p)[0]][0]
        for p in param_names
    ])
    if verbose:
        print("Parameter means:", means)

    # Prepare samples array
    n_params = len(param_names)
    samples = np.zeros((num, n_params))

    # Identify covariance vs flat-prior parameters
    cov_indices = [i for i, p in enumerate(param_names) if p not in flat_params and p not in triangle_params]
    flat_indices = [i for i, p in enumerate(param_names) if p in flat_params]
    triangle_indices = [i for i, p in enumerate(param_names) if p in triangle_params]
    if verbose:
        print("Covariance-sampled indices:", cov_indices)
        print("Flat-prior indices:", flat_indices)

    # Covariance sampling with per-dimension bound enforcement
    if cov_indices:
        cov_mean = means[cov_indices]
        cov_cov = fit.cov_matrix.cov_matrix[np.ix_(cov_indices, cov_indices)]
        if verbose:
            print("Cov-sample mean:", cov_mean)
            print("Cov-sample covariance matrix shape:", cov_cov.shape)

        # Initial multivariate normal draw
        cov_samples = stats.multivariate_normal.rvs(mean=cov_mean, cov=cov_cov, size=num)
        cov_samples = np.atleast_2d(cov_samples)
        if verbose:
            print("Initial cov_samples[0:5]:", cov_samples[:5])

        # For each covariate dimension, replace OOB values with flat-prior samples
        for j, idx in enumerate(cov_indices):
            pname = param_names[idx]
            lo, hi = fit.bounds[pname]
            mask_j = (cov_samples[:, j] < lo) | (cov_samples[:, j] > hi)
            if verbose:
                print(f"Checking bounds for '{pname}' (lo={lo}, hi={hi}): {mask_j.sum()} out-of-bounds")
                print(f"Before replacement, cov_samples[:5, {j}]:", cov_samples[:5, j])

            if np.any(mask_j):
                count = np.sum(mask_j)
                if pname.startswith('b_') or pname.startswith('v_'):
                    cov_samples[mask_j, j] = np.random.uniform(lo, hi, size=count)
                else:
                    cov_samples[mask_j, j] = log_uniform(lo, hi, size=count)
                if verbose:
                    print(f"After replacement, cov_samples[:5, {j}]:", cov_samples[:5, j])

        samples[:, cov_indices] = cov_samples
        if verbose:
            print("Covariance samples assigned to samples[:, cov_indices]")
            print("samples[:, cov_indices][0:5]:", samples[:5, cov_indices])

    # Flat-prior parameters: uniform for b_/v_, log-uniform otherwise
    for i in flat_indices:
        lower, upper = fit.bounds[param_names[i]]
        name = param_names[i]
        if verbose:
            print(f"Sampling flat-prior for '{name}' in [{lower}, {upper}]")
        if name.startswith('b_') or name.startswith('v_'):
            samples[:, i] = np.random.uniform(lower, upper, size=num)
        else:
            samples[:, i] = log_uniform(lower, upper, size=num)
        if verbose:
            print(f"samples[:5, {i}] for '{name}':", samples[:5, i])

    # Apply triangular sampling if specified
    for i in triangle_indices:
        lower, upper = fit.bounds[param_names[i]]
        if verbose:
            print(f"Sampling triangular-prior for '{param_names[i]}' in [{lower}, {upper}]")
        samples[:, i] = np.random.triangular(lower, upper, upper, size=num)
    # Parse constraint strings into tuples. Parameter names may include
    # characters beyond \w (e.g. dots, dashes), so accept any token that
    # isn't whitespace or an angle bracket.
    parsed_constraints = []
    canonical_constraints = []
    constraint_info = []
    for c in param_constraints:
        m = re.match(r"\s*([^\s<>]+)\s*([<>])\s*([^\s<>]+)\s*", c)
        if not m:
            continue
        constraint = (m.group(1), m.group(3), m.group(2))
        parsed_constraints.append(constraint)
        canonical = (constraint[0], constraint[1]) if constraint[2] == '<' else (constraint[1], constraint[0])
        canonical_constraints.append(canonical)
        constraint_info.append((constraint, canonical))
    if verbose:
        print("Parsed constraints:", parsed_constraints)

    def _find_velocity_chains(edges):
        """Derive linear ordered groups from canonical velocity edges."""
        if not edges:
            return []
        # build undirected connectivity for component discovery
        neighbors = {}
        for lo, hi in edges:
            neighbors.setdefault(lo, set()).add(hi)
            neighbors.setdefault(hi, set()).add(lo)

        groups = []
        visited_nodes = set()
        for node in neighbors:
            if node in visited_nodes:
                continue
            stack = [node]
            component = []
            while stack:
                current = stack.pop()
                if current in visited_nodes:
                    continue
                visited_nodes.add(current)
                component.append(current)
                for nb in neighbors.get(current, ()):  # neighbors may be empty
                    if nb not in visited_nodes:
                        stack.append(nb)

            if len(component) < 2:
                continue  # single node, no ordering

            comp_edges = [(a, b) for (a, b) in edges if a in component and b in component]
            indegree = {n: 0 for n in component}
            outdegree = {n: 0 for n in component}
            for a, b in comp_edges:
                outdegree[a] += 1
                indegree[b] += 1

            starts = [n for n in component if indegree[n] == 0]
            ends = [n for n in component if outdegree[n] == 0]
            if len(starts) != 1 or len(ends) != 1:
                continue  # not a simple chain

            start = starts[0]
            end = ends[0]

            chain_ok = True
            for n in component:
                if n in (start, end):
                    continue
                if indegree[n] != 1 or outdegree[n] != 1:
                    chain_ok = False
                    break
            if not chain_ok:
                continue

            order = [start]
            current = start
            while current != end:
                next_nodes = [b for (a, b) in comp_edges if a == current]
                if len(next_nodes) != 1:
                    chain_ok = False
                    break
                nxt = next_nodes[0]
                if nxt in order:
                    chain_ok = False
                    break
                order.append(nxt)
                current = nxt

            if not chain_ok or len(order) != len(component):
                continue

            edge_chain = {(order[i], order[i + 1]) for i in range(len(order) - 1)}
            groups.append({"order": order, "edges": edge_chain})

        return groups

    def _sample_dirichlet_chain(bounds, size, alpha):
        """Sample ordered positions within bounds using Dirichlet spacings."""
        raw_lo = [lo for lo, _ in bounds]
        raw_hi = [hi for _, hi in bounds]
        total_span = max(raw_hi) - min(raw_lo)
        if total_span <= 0:
            center = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
            return np.tile(center, (size, 1))

        eps = max(total_span * 1e-12, 1e-12)
        lower = raw_lo[:]
        upper = raw_hi[:]
        for i in range(1, len(bounds)):
            lower[i] = max(lower[i], lower[i - 1] + eps)
        for i in range(len(bounds) - 2, -1, -1):
            upper[i] = min(upper[i], upper[i + 1] - eps)
        for lo, hi in zip(lower, upper):
            if lo > hi:
                raise RuntimeError("Infeasible ordered bounds for Dirichlet sampling.")

        total_lo = lower[0]
        total_hi = upper[-1]
        total_range = total_hi - total_lo
        if total_range <= 0:
            return np.tile(np.array(lower), (size, 1))

        alpha_vec = np.full(len(bounds) + 1, alpha, dtype=float)
        feasible_bounds = list(zip(lower, upper))
        draws = np.empty((size, len(bounds)))
        pending = np.arange(size)
        attempts = 0
        max_attempts = 1000

        while pending.size > 0 and attempts < max_attempts:
            attempts += 1
            gaps = np.random.dirichlet(alpha_vec, size=pending.size) * total_range
            vals = np.cumsum(gaps[:, : len(bounds)], axis=1) + total_lo
            valid = np.ones(pending.size, dtype=bool)
            for j, (lo, hi) in enumerate(feasible_bounds):
                valid &= (vals[:, j] >= lo) & (vals[:, j] <= hi)
            if np.any(valid):
                draws[pending[valid]] = vals[valid]
            pending = pending[~valid]

        if pending.size > 0:
            raise RuntimeError("Dirichlet sampling unable to satisfy bounds after many attempts.")

        return draws

    constraint_pairs = list(zip(parsed_constraints, canonical_constraints))

    velocity_edges = []
    for (orig, canon) in constraint_pairs:
        pA, pB, _ = orig
        if pA in param_names and pB in param_names and pA.startswith('v_') and pB.startswith('v_'):
            velocity_edges.append(canon)
    # preserve order while removing duplicates
    velocity_edges = list(dict.fromkeys(velocity_edges))

    used_velocity_edges = set()
    velocity_groups = _find_velocity_chains(velocity_edges)

    for group in velocity_groups:
        order = group["order"]
        if not all(name in param_names for name in order):
            continue
        indices = [param_names.index(name) for name in order]
        bounds = [fit.bounds[name] for name in order]
        try:
            dir_samples = _sample_dirichlet_chain(bounds, num, order_alpha)
        except RuntimeError as err:
            if verbose:
                print(f"Dirichlet sampling failed for group {order}: {err}")
            continue

        for col, idx in enumerate(indices):
            samples[:, idx] = dir_samples[:, col]
        used_velocity_edges.update(group["edges"])
        if verbose:
            print(f"Applied Dirichlet ordering to velocities {order} (alpha={order_alpha})")

    leftover_constraints = [orig for (orig, canon) in constraint_pairs if canon not in used_velocity_edges]
    if verbose and leftover_constraints:
        print("Constraints remaining for direct enforcement:", leftover_constraints)

    if leftover_constraints:
        for pA, pB, rel in leftover_constraints:
            if pA not in param_names or pB not in param_names:
                continue
            iA, iB = param_names.index(pA), param_names.index(pB)
            loA, hiA = fit.bounds[pA]

            span = hiA - loA
            eps = max(span * 1e-12, 1e-12) if span > 0 else 1e-12

            if rel == '>':
                upper = np.minimum(hiA, samples[:, iA])
                lower = np.maximum(loA, samples[:, iB] + eps)
                invalid = lower >= upper
                if np.any(invalid):
                    if verbose:
                        print(f"Warning: {pA}>{pB} constraint leaves no volume for {np.sum(invalid)} walkers; clamping to bounds.")
                    lower = np.minimum(lower, upper - eps)
                mask = lower < upper
                if np.any(mask):
                    if pA.startswith('b_') or pA.startswith('v_'):
                        samples[mask, iA] = np.random.uniform(lower[mask], upper[mask])
                    else:
                        samples[mask, iA] = np.array([log_uniform(lo, hi) for lo, hi in zip(lower[mask], upper[mask])])

            elif rel == '<':
                lower = np.maximum(loA, samples[:, iA])
                upper = np.minimum(hiA, samples[:, iB] - eps)
                invalid = lower >= upper
                if np.any(invalid):
                    if verbose:
                        print(f"Warning: {pA}<{pB} constraint leaves no volume for {np.sum(invalid)} walkers; clamping to bounds.")
                    upper = np.maximum(lower + eps, upper)
                mask = lower < upper
                if np.any(mask):
                    if pA.startswith('b_') or pA.startswith('v_'):
                        samples[mask, iA] = np.random.uniform(lower[mask], upper[mask])
                    else:
                        samples[mask, iA] = np.array([log_uniform(lo, hi) for lo, hi in zip(lower[mask], upper[mask])])

        if verbose:
            print("Order constraints enforced via direct sampling.")

    if verbose:
        print("genSamples completed")
        print("Final samples[0:5]:", samples[:5])

    jitter_floor = 1e-2
    param_sigmas = np.std(samples, axis=0, ddof=1)
    low_sigma_mask = param_sigmas < jitter_floor
    if np.any(low_sigma_mask):
            rng = np.random.default_rng(42)
            jitter = np.zeros_like(samples)
            jitter[:, low_sigma_mask] = rng.normal(scale=jitter_floor, size=(np.sum(low_sigma_mask), samples.shape[0])).T
            samples = samples + jitter
            print("Injected jitter into parameters:", np.array(param_names)[low_sigma_mask])

    return samples

def genSamplesSimple(
    fit,
    n_walkers,
    scale=1e-3,
    seed=None,
    column_boxcar_dex=None,
    velocity_boxcar_kms=None,
):
    """
    Generate simple walker positions around the best-fit parameters.

    Parameters
    ----------
    fit : FittedModel
        Model with ``param_names``/``parameters``/``cov_matrix``/``bounds`` attributes.
    n_walkers : int
        Number of walker initial positions to generate.
    scale : float or array-like, optional
        Gaussian fractional scatter (per parameter) used only for parameters that are
        not handled by the column/velocity boxcar options.
    seed : int, optional
        RNG seed for reproducibility.
    column_boxcar_dex : float, optional
        Half-width of a uniform interval in log10 space for parameters whose names
        start with ``"N_"``. Walkers are drawn uniformly in
        :math:`\log_{10}`(param) within ``± column_boxcar_dex`` about the best-fit value
        (subject to parameter bounds). If omitted, the parameter falls back to the
        Gaussian ``scale`` behaviour.
    velocity_boxcar_kms : float, optional
        Half-width (in km/s) of a uniform interval around the best-fit value used for
        parameters whose names start with ``"v_"`` or ``"b_"``. Values are clipped to
        parameter bounds. If omitted, those parameters fall back to the Gaussian
        ``scale`` behaviour.

    Returns
    -------
    init_pos : ndarray, shape (n_walkers, n_params)
        Initial walker positions ordered as ``fit.cov_matrix.param_names``.
    """

    rng = np.random.default_rng(seed)
    # extract best-fit parameter vector in the same order as fit.cov_matrix.param_names
    ordered_names = list(fit.cov_matrix.param_names)
    best = np.array([
        fit.parameters[np.where(np.array(fit.param_names, dtype=str) == p)[0]][0]
        for p in ordered_names
    ])

    if np.isscalar(scale):
        scale_arr = np.full_like(best, fill_value=scale, dtype=float)
    else:
        scale_arr = np.array(scale, dtype=float)
        if scale_arr.shape != best.shape:
            raise ValueError("scale must be scalar or same length as parameters")

    init_pos = np.empty((n_walkers, best.size), dtype=float)

    for idx, name in enumerate(ordered_names):
        best_val = best[idx]
        lower_bound, upper_bound = fit.bounds[name]
        center_val = np.clip(best_val, lower_bound, upper_bound)

        def _uniform(lo, hi):
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                return np.full(n_walkers, center_val)
            return rng.uniform(lo, hi, size=n_walkers)

        if column_boxcar_dex is not None and name.startswith("N_"):
            if best_val > 0 and lower_bound > 0 and upper_bound > 0:
                log_center = np.log10(best_val)
                log_lo = log_center - column_boxcar_dex
                log_hi = log_center + column_boxcar_dex
                log_lo = max(log_lo, np.log10(lower_bound))
                log_hi = min(log_hi, np.log10(upper_bound))
                if log_lo >= log_hi:
                    samples = _uniform(lower_bound, upper_bound)
                else:
                    samples = 10 ** rng.uniform(log_lo, log_hi, size=n_walkers)
            else:
                samples = _uniform(lower_bound, upper_bound)
        elif velocity_boxcar_kms is not None and (name.startswith("v_") or name.startswith("b_")):
            span_lo = max(lower_bound, best_val - velocity_boxcar_kms)
            span_hi = min(upper_bound, best_val + velocity_boxcar_kms)
            if span_lo >= span_hi:
                samples = np.full(n_walkers, center_val)
            else:
                samples = rng.uniform(span_lo, span_hi, size=n_walkers)
        else:
            sigma = scale_arr[idx] * max(abs(best_val), 1.0)
            samples = rng.normal(loc=best_val, scale=sigma, size=n_walkers)

        samples = np.clip(samples, lower_bound, upper_bound)
        init_pos[:, idx] = samples

    return init_pos


def genSamplesSimple2(
    fit,
    n_walkers,
    scale=1e-3,
    seed=None,
    column_boxcar_dex=None,
    T_boxcar_dex=None,
    velocity_boxcar_kms=None,
    continuum_loc_velocity_boxcar_kms=None,
    continuum_coef_boxcar=None,
    continuum_veto_intervals=None,
    continuum_veto_mask=None,
):
    """
    Variant of ``genSamplesSimple`` with explicit controls for continuum knots.

    Parameters
    ----------
    fit : FittedModel
        Model with ``param_names``/``parameters``/``cov_matrix``/``bounds`` attributes.
    n_walkers : int
        Number of walker initial positions to generate.
    scale : float or array-like, optional
        Fractional Gaussian scatter for parameters not handled by special cases.
    seed : int, optional
        RNG seed for reproducibility.
    column_boxcar_dex : float, optional
        Half-width in dex for ``N_`` parameters (identical to ``genSamplesSimple``).
    T_boxcar_dex : float, optional
        Half-width in dex for ``T_`` parameters (log-uniform around the best fit).
    velocity_boxcar_kms : float, optional
        Half-width in km/s for ``v_``/``b_`` parameters (identical to ``genSamplesSimple``).
    continuum_loc_velocity_boxcar_kms : float, dict, or callable, optional
        Half-width (km/s) of a uniform draw around each continuum knot-location
        parameter. Supports scalar, ``{'*': width}``, or callable returning a width
        per parameter. Converted internally to wavelength units using each knot's
        best-fit wavelength.
    continuum_coef_boxcar : float, dict, or callable, optional
        Half-width (linear units) of a uniform draw for continuum knot coefficients.
        Lookup semantics mirror ``continuum_loc_velocity_boxcar_kms``.
    continuum_veto_intervals : sequence of (low, high), optional
        Forbidden wavelength intervals for knot locations. Provide directly in the
        same units as the model parameters (floats or ``Quantity`` convertible to
        length). When a knot's best-fit value lies inside a forbidden interval, the
        walkers are drawn from mirrored half-normal distributions anchored at the
        interval edges using the boxcar width.
    continuum_veto_mask : dict, optional
        Full mask configuration (matching the structure expected by
        ``make_continuum_node_veto``) from which velocity-based veto ranges will be
        converted into wavelength intervals and merged with
        ``continuum_veto_intervals``.

    Returns
    -------
    init_pos : ndarray, shape (n_walkers, n_params)
        Initial walker positions ordered as ``fit.cov_matrix.param_names``.
    """

    rng = np.random.default_rng(seed)

    c_kms_local = c.to(u.km / u.s).value

    ordered_names = list(fit.cov_matrix.param_names)
    best = np.array([
        fit.parameters[np.where(np.array(fit.param_names, dtype=str) == p)[0]][0]
        for p in ordered_names
    ])

    if np.isscalar(scale):
        scale_arr = np.full_like(best, fill_value=scale, dtype=float)
    else:
        scale_arr = np.array(scale, dtype=float)
        if scale_arr.shape != best.shape:
            raise ValueError("scale must be scalar or same length as parameters")

    init_pos = np.empty((n_walkers, best.size), dtype=float)

    def _resolve(option, pname):
        if option is None:
            return None
        if callable(option):
            return option(pname)
        if isinstance(option, dict):
            if pname in option:
                return option[pname]
            if '*' in option:
                return option['*']
            return None
        return option

    def _convert_edge(value, ref_lambda):
        if hasattr(value, 'to'):
            try:
                if value.unit.is_equivalent(u.km / u.s):
                    return ref_lambda * (1.0 + value.to(u.km / u.s).value / c_kms_local)
                return value.to(u.AA).value
            except Exception:
                return float(value)
        return float(value)

    def _convert_interval_pair(lo, hi, lambda_center):
        lo_val = _convert_edge(lo, lambda_center)
        hi_val = _convert_edge(hi, lambda_center)
        return (min(lo_val, hi_val), max(lo_val, hi_val))

    def _build_veto_intervals():
        collected = []

        if continuum_veto_mask is not None:
            entries = continuum_veto_mask.get('redshifts', [])
            table = globals().get('SEARCH_LINES')
            for entry in entries:
                z_entry = entry.get('z')
                cont_cfg = entry.get('continuum_node_vrange')
                if z_entry is None or not cont_cfg:
                    continue
                for line_name, ranges in cont_cfg.items():
                    if table is None:
                        continue
                    rows = table[table['tempname'] == line_name]
                    if len(rows) == 0:
                        continue
                    rest_wave = rows['wave'][0]
                    lambda_center = rest_wave * (1.0 + float(z_entry))
                    for vr in ranges:
                        if not isinstance(vr, (list, tuple)) or len(vr) != 2:
                            continue
                        low, high = vr
                        try:
                            pair = _convert_interval_pair(low, high, lambda_center)
                        except Exception:
                            continue
                        collected.append(pair)

        if continuum_veto_intervals:
            for interval in continuum_veto_intervals:
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    continue
                lo, hi = interval
                try:
                    lo_val = lo.to(u.AA).value if hasattr(lo, 'to') else float(lo)
                    hi_val = hi.to(u.AA).value if hasattr(hi, 'to') else float(hi)
                except Exception:
                    continue
                collected.append((min(lo_val, hi_val), max(lo_val, hi_val)))

        if not collected:
            return None
        return np.asarray(collected, dtype=float)

    def _in_intervals(values, intervals):
        if intervals is None:
            return np.zeros(values.shape, dtype=bool)
        mask = np.zeros(values.shape, dtype=bool)
        for lo, hi in intervals:
            mask |= (values >= lo) & (values <= hi)
        return mask

    def _reject_into_allowed(samples, intervals, draw_fn):
        if intervals is None:
            return samples
        mask = _in_intervals(samples, intervals)
        attempts = 0
        max_attempts = 50
        while np.any(mask) and attempts < max_attempts:
            count = np.count_nonzero(mask)
            samples[mask] = draw_fn(count)
            mask = _in_intervals(samples, intervals)
            attempts += 1
        if np.any(mask):
            idx_bad = np.where(mask)[0]
            for b in idx_bad:
                val = samples[b]
                snapped = val
                for lo, hi in intervals:
                    if lo <= val <= hi:
                        snapped = lo if abs(val - lo) <= abs(val - hi) else hi
                        break
                samples[b] = snapped
        return samples

    veto_intervals = _build_veto_intervals()

    for idx, name in enumerate(ordered_names):
        best_val = best[idx]
        lower_bound, upper_bound = fit.bounds[name]
        center_val = np.clip(best_val, lower_bound, upper_bound)

        def _uniform(lo, hi):
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                return np.full(n_walkers, center_val)
            return rng.uniform(lo, hi, size=n_walkers)

        if column_boxcar_dex is not None and name.startswith("N_"):
            if best_val > 0 and lower_bound > 0 and upper_bound > 0:
                log_center = np.log10(best_val)
                log_lo = log_center - float(column_boxcar_dex)
                log_hi = log_center + float(column_boxcar_dex)
                log_lo = max(log_lo, np.log10(lower_bound))
                log_hi = min(log_hi, np.log10(upper_bound))
                if log_lo >= log_hi:
                    samples = _uniform(lower_bound, upper_bound)
                else:
                    samples = 10 ** rng.uniform(log_lo, log_hi, size=n_walkers)
            else:
                samples = _uniform(lower_bound, upper_bound)
        elif T_boxcar_dex is not None and name.startswith("T_"):
            if best_val > 0 and lower_bound > 0 and upper_bound > 0:
                log_center = np.log10(best_val)
                log_lo = log_center - float(T_boxcar_dex)
                log_hi = log_center + float(T_boxcar_dex)
                log_lo = max(log_lo, np.log10(lower_bound))
                log_hi = min(log_hi, np.log10(upper_bound))
                if log_lo >= log_hi:
                    samples = _uniform(lower_bound, upper_bound)
                else:
                    samples = 10 ** rng.uniform(log_lo, log_hi, size=n_walkers)
            else:
                samples = _uniform(lower_bound, upper_bound)
        elif velocity_boxcar_kms is not None and (name.startswith("v_") or name.startswith("b_")):
            span_lo = max(lower_bound, best_val - float(velocity_boxcar_kms))
            span_hi = min(upper_bound, best_val + float(velocity_boxcar_kms))
            if span_lo >= span_hi:
                samples = np.full(n_walkers, center_val)
            else:
                samples = rng.uniform(span_lo, span_hi, size=n_walkers)
        elif "knot_loc" in name:
            boxcar_v = _resolve(continuum_loc_velocity_boxcar_kms, name)
            intervals = veto_intervals
            if boxcar_v is None:
                sigma = scale_arr[idx] * max(abs(best_val), 1.0)
                draw = lambda size: rng.normal(loc=best_val, scale=sigma, size=size)
                samples = draw(n_walkers)
                samples = _reject_into_allowed(samples, intervals, draw)
            else:
                if hasattr(boxcar_v, 'to'):
                    boxcar_v = boxcar_v.to(u.km / u.s).value
                else:
                    boxcar_v = float(boxcar_v)
                ref_lambda = max(abs(best_val), 1.0)
                half_span = ref_lambda * (boxcar_v / c_kms_local)
                if half_span <= 0:
                    samples = np.full(n_walkers, center_val)
                else:
                    span_lo = max(lower_bound, best_val - half_span)
                    span_hi = min(upper_bound, best_val + half_span)
                    if span_lo >= span_hi:
                        samples = np.full(n_walkers, center_val)
                    else:
                        active_interval = None
                        if intervals is not None:
                            for lo, hi in intervals:
                                if lo <= best_val <= hi:
                                    active_interval = (lo, hi)
                                    break
                        if active_interval is not None:
                            left_edge, right_edge = active_interval
                            choose_left = rng.random(n_walkers) < 0.5
                            left_count = np.count_nonzero(choose_left)
                            right_count = n_walkers - left_count
                            samples = np.empty(n_walkers, dtype=float)
                            if left_count:
                                draws_left = stats.halfnorm.rvs(
                                    scale=half_span,
                                    size=left_count,
                                    random_state=rng,
                                )
                                draws_left = np.maximum(draws_left, 1e-12)
                                samples[choose_left] = left_edge - draws_left
                            if right_count:
                                draws_right = stats.halfnorm.rvs(
                                    scale=half_span,
                                    size=right_count,
                                    random_state=rng,
                                )
                                draws_right = np.maximum(draws_right, 1e-12)
                                samples[~choose_left] = right_edge + draws_right

                            def draw_half(size):
                                choice = rng.random(size) < 0.5
                                res = np.empty(size, dtype=float)
                                if np.any(choice):
                                    d_left = stats.halfnorm.rvs(
                                        scale=half_span,
                                        size=np.count_nonzero(choice),
                                        random_state=rng,
                                    )
                                    d_left = np.maximum(d_left, 1e-12)
                                    res[choice] = left_edge - d_left
                                if np.any(~choice):
                                    d_right = stats.halfnorm.rvs(
                                        scale=half_span,
                                        size=np.count_nonzero(~choice),
                                        random_state=rng,
                                    )
                                    d_right = np.maximum(d_right, 1e-12)
                                    res[~choice] = right_edge + d_right
                                return res

                            samples = _reject_into_allowed(samples, intervals, draw_half)
                        else:
                            draw = lambda size: rng.uniform(span_lo, span_hi, size=size)
                            samples = draw(n_walkers)
                            samples = _reject_into_allowed(samples, intervals, draw)
            samples = np.clip(samples, lower_bound, upper_bound)
        elif "knot_coef" in name:
            coef_box = _resolve(continuum_coef_boxcar, name)
            if coef_box is None:
                sigma = scale_arr[idx] * max(abs(best_val), 1.0)
                samples = rng.normal(loc=best_val, scale=sigma, size=n_walkers)
            else:
                if hasattr(coef_box, 'to'):
                    width = coef_box.to_value()
                else:
                    width = float(coef_box)
                span_lo = max(lower_bound, best_val - width)
                span_hi = min(upper_bound, best_val + width)
                if span_lo >= span_hi:
                    samples = np.full(n_walkers, center_val)
                else:
                    samples = rng.uniform(span_lo, span_hi, size=n_walkers)
            samples = np.clip(samples, lower_bound, upper_bound)
        else:
            sigma = scale_arr[idx] * max(abs(best_val), 1.0)
            samples = rng.normal(loc=best_val, scale=sigma, size=n_walkers)

        samples = np.clip(samples, lower_bound, upper_bound)
        init_pos[:, idx] = samples

    return init_pos

def generate_order_priors(names, model, verbose=False):
    """
    Given a list of velocity parameter names and the fitted model,
    look up their starting values, sort them, and return a list
    of comparison strings preserving that order.
    """
    nv = []
    for n in names:
        if n not in model.param_names:
            raise ValueError(f"'{n}' is not a model parameter")
        idx = model.param_names.index(n)
        nv.append((n, model.parameters[idx]))
    sorted_names = [n for n, v in sorted(nv, key=lambda x: x[1])]
    comparisons = [f"{sorted_names[i]} < {sorted_names[i+1]}"
                   for i in range(len(sorted_names) - 1)]
    if verbose:
        print("generate_order_priors: sorted =", sorted_names)
        print("  comparisons =", comparisons)
    return comparisons

def analyze_emcee_sampler(sampler, param_names=None, verbose=False,
                          save_path='emcee_diagnostics.txt', burnin=0):
    """
    Run convergence diagnostics on an emcee.EnsembleSampler output.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The fitted sampler object.
    param_names : list of str, optional
        Names for parameters (default: param0, param1, ...).

    Interpretation Guidelines
    -------------------------
    Core Diagnostics:
      - R-hat: should be <= 1.01 (<= 1.05 sometimes acceptable). Higher -> lack of convergence.
      - ESS_bulk: should be >= 400 for reliable mean estimates.
      - ESS_tail: should be >= 100-400 for reliable tail quantiles.
      - MCSE: should be small relative to posterior SD (e.g. MCSE/SD <= 0.1).

    Split-half Consistency:
      - Compare mean of first vs second half of samples.
      - Difference should be small relative to posterior SD.
      - Large differences suggest non-stationarity or poor mixing.

    Geweke Diagnostic:
      - Z-scores comparing early vs late segments.
      - |z| <= 2 for most parameters is consistent with convergence.
    """
    # prepare logging if verbose
    logs = []
    def log(msg):
        print(msg)
        logs.append(str(msg))

    # get raw chain and discard burn-in
    chain = sampler.get_chain(discard=burnin)  # shape (nsteps, nwalkers, ndim)
    if verbose:
        log(f"Discarding first {burnin} steps for burn-in, new chain.shape = {chain.shape}")

    nsteps, nwalkers, ndim = chain.shape
    draws = chain.reshape(nsteps * nwalkers, ndim)

    if verbose:
        log(f"analyze_emcee_sampler: chain.shape = {chain.shape}")

    # acceptance fractions
    acc_frac = sampler.acceptance_fraction
    if verbose:
        log("=== Acceptance Fractions ===")
        log(f"  Mean acceptance fraction: {acc_frac.mean():.3f}")

    if param_names is None:
        param_names = [f"param{i}" for i in range(ndim)]
    if verbose:
        log("Parameter names: " + ", ".join(param_names))

    # wrap into arviz and compute diagnostics
    # Note: arviz.from_emcee no longer supports burnin; discard burnin manually
    idata = az.from_emcee(sampler, var_names=param_names)
    if burnin and burnin > 0:
        # remove initial burnin draws by indexing along the 'draw' dimension
        idata = idata.isel(draw=slice(burnin, None))
    rhat = az.rhat(idata)
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")
    mcse_all = az.mcse(idata)

    if verbose:
        log("=== Core Diagnostics ===")
        for name in param_names:
            v_rhat = rhat[name].values
            v_bulk = ess_bulk[name].values
            v_tail = ess_tail[name].values
            log(f"  {name}: R-hat={v_rhat:.3f}, ESS_bulk={v_bulk:.1f}, ESS_tail={v_tail:.1f}")
        log("=== MCSE / SD ===")
        for i, name in enumerate(param_names):
            mcse = mcse_all[name].values
            sd = np.std(draws[:, i])
            ratio = mcse / sd if sd != 0 else np.nan
            log(f"  {name}: MCSE={mcse:.4f}, MCSE/SD={ratio:.3f}")

    if verbose:
        log("=== Split-half Consistency ===")
        half = nsteps // 2
        for i, name in enumerate(param_names):
            mean_first = np.mean(chain[:half, :, i])
            mean_second = np.mean(chain[half:, :, i])
            diff = np.abs(mean_first - mean_second)
            pooled_sd = np.std(draws[:, i])
            log(f"  {name}: mean1={mean_first:.3f}, mean2={mean_second:.3f}, "
                f"diff={diff:.3f}, diff/SD={diff/pooled_sd:.3f}")

        log("=== Geweke Diagnostic (Quick Classical Test) ===")
        frac_first, frac_last = 0.1, 0.5
        for i, name in enumerate(param_names):
            series = draws[:, i]
            n = len(series)
            first = series[:int(frac_first * n)]
            last = series[int((1 - frac_last) * n):]
            z = (np.mean(first) - np.mean(last)) / np.sqrt(
                np.var(first) / len(first) + np.var(last) / len(last)
            )
            log(f"  {name}: Geweke z = {z:.2f}")

    # write logs to file
    if verbose:
        with open(save_path, "w") as f:
            for line in logs:
                f.write(line + "\n")
        print(f"All diagnostics saved to {save_path}")

    return idata


def plot_emcee_diagnostics(sampler, param_names=None, save_path=None,
                            verbose=False, log_x=False, multiple=1.0, burnin=0, xlim=None):
    """
    Produce running statistics for emcee chains using ArviZ.

    Parameters:
        sampler    : emcee.EnsembleSampler
        param_names: list of str, optional names for each parameter
        save_path  : str, optional path to save the figure
        verbose    : bool, if True print debug information
        log_x      : bool, if True plot the x-axis on log scale (skipping step 0)
        multiple   : float, scaling factor to apply to parameters starting with "N_"
        window     : int, optional size of moving window. If None, use all past samples.
    """
    if verbose:
        print("Entering plot_emcee_diagnostics")
        print(f"  log_x={log_x}, multiple={multiple}, burnin={burnin}, xlim={xlim}, save_path={save_path}")


    # load full chain: shape (nsteps, nwalkers, ndim)
    chain = sampler.get_chain(discard=burnin)
    nsteps, nwalkers, ndim = chain.shape
    if verbose:
        print("  Loaded chain with shape:", chain.shape)

    # default param names
    if param_names is None:
        param_names = [f"param{i}" for i in range(ndim)]
    if verbose:
        print("  Using parameter names:", param_names)

    # prepare plotting
    steps = np.arange(nsteps)
    fig, axes = plt.subplots(ndim, 1, figsize=(8, 3 * ndim), squeeze=False)
    if verbose:
        print(f"  Created figure with {ndim} subplots")

    # quantile levels
    qs = [0.5, 2.5, 16, 50, 84, 97.5, 99.5]
    for i, name in enumerate(param_names):
        if verbose:
            print(f"  Processing parameter '{name}' (index {i})")
        ax = axes[i, 0]
        # extract and scale data per parameter
        data = chain[:, :, i]
        if name.startswith("N_"):
            data = data * multiple
            if verbose:
                print(f"    Applied multiple={multiple} to data for '{name}'")

        if verbose:
            print(f"    Data shape for '{name}':", data.shape)

        # compute all quantiles at once: shape (len(qs), nsteps)
        qvals = np.percentile(data, qs, axis=1)
        if verbose:
            print(f"    Quantiles computed for '{name}', qvals.shape =", qvals.shape)

        # unpack quantiles
        q005, q025, q16, q50, q84, q975, q995 = qvals

        # optionally handle log x-axis
        mask = steps > 0 if log_x else None
        xs = steps[mask] if log_x else steps
        if verbose and log_x:
            print(f"    Using log x-axis, skipping step 0, xs[0:5] =", xs[:5])

        # plot widest to narrowest
        ax.fill_between(xs,
                        q005[mask] if mask is not None else q005,
                        q995[mask] if mask is not None else q995,
                        color='C3', alpha=0.2, label='99.5%')
        ax.fill_between(xs,
                        q025[mask] if mask is not None else q025,
                        q975[mask] if mask is not None else q975,
                        color='C2', alpha=0.3, label='95%')
        ax.fill_between(xs,
                        q16[mask] if mask is not None else q16,
                        q84[mask] if mask is not None else q84,
                        color='C1', alpha=0.4, label='68%')
        # plot median
        ax.plot(xs,
                q50[mask] if mask is not None else q50,
                color='C0', label='Median')

        # set styles
        if log_x:
            ax.set_xscale('log')
        if name.startswith('T_') or name.startswith('N_'):
            ax.set_yscale('log')
        ax.set_title(f"Running quantiles: {name}")
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        if xlim is not None:
            ax.set_xlim(xlim)
        if verbose:
            print(f"    Finished plotting for '{name}'")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if verbose:
            print("Saved plot_emcee_diagnostics figure to", save_path)
        plt.close(fig)
    else:
        if verbose:
            print("Displaying plot_emcee_diagnostics interactively")
        plt.show()


def plot_corner_chain(sampler, param_names, best_params=False,
                      multiple=1.0, range_val=0.99, burnin=0,
                      save_path=None, return_fig=False,
                      verbose=False, main_only=False,
                      main_param_names=None, main_param_regex=None,
                      exclude_param_regex=None):
    """
    Generate a corner plot from the flat chain.

    sampler        : emcee.EnsembleSampler instance
    param_names    : list of parameter names (length == n_params)
    best_params    : bool, if True compute the maximum-likelihood sample from chain
    multiple       : scaling factor to apply to N_ parameters before log10
    range_val      : fraction of data to show in each axis of the corner plot
    burnin         : int, number of initial steps to discard from each walker
    save_path      : optional str, path/filename to save the figure
    return_fig     : bool, if True the matplotlib Figure is returned
    verbose        : bool, if True print additional information during processing
    """
    # get the flattened chain with burn-in discarded (or use provided array)
    # sampler may be an emcee sampler or a precomputed flat chain (n_samples, n_params)
    arr = _as_flat_chain(sampler, burnin=burnin, verbose=verbose)
    n_params = len(param_names)
    if verbose:
        print("plot_corner_chain: chain shape after burnin discard =", arr.shape)

    if arr.ndim != 2 or arr.shape[1] != n_params:
        raise ValueError(f"Expected chain shape (n_samples, {n_params}), got {arr.shape}")

    def _select_param_indices(names):
        indices = list(range(len(names)))

        if main_only:
            if main_param_names is None and main_param_regex is None:
                raise ValueError("main_only=True requires main_param_names or main_param_regex")
            keep = set()
            if main_param_names is not None:
                keep.update([n for n in names if n in set(main_param_names)])
            if main_param_regex is not None:
                pattern = re.compile(main_param_regex)
                keep.update([n for n in names if pattern.search(str(n))])
            indices = [i for i, n in enumerate(names) if n in keep]

        if exclude_param_regex is not None:
            pattern = re.compile(exclude_param_regex)
            indices = [i for i in indices if not pattern.search(str(names[i]))]

        if not indices:
            raise ValueError("No parameters selected for corner plot")
        return indices

    selected_indices = _select_param_indices(param_names)
    selected_names = [param_names[i] for i in selected_indices]
    arr = arr[:, selected_indices]
    n_params = len(selected_names)

    # apply log / scale transforms for corner plotting
    chain_plot = transform_chain_plot(arr, selected_names, multiple)
    if verbose:
        mins = np.nanmin(chain_plot, axis=0)
        maxs = np.nanmax(chain_plot, axis=0)
        print("  chain_plot stats: min =", mins, "max =", maxs)

    # optionally compute "best" (max sum-of-params) sample as truths
    truths = None
    if best_params:
        idx_max = np.nanargmax(np.nansum(arr[:, :n_params], axis=1))
        truths = chain_plot[idx_max].copy()
        if verbose:
            print("  best_params index, truths =", idx_max, truths)

    # corner plot
    # Draw contours only (no scatter points) at 1, 2 and 3-sigma; reduce bins
    # to speed rendering on large chains. Levels are expressed as enclosed
    # probabilities: 1σ ~ 0.682689, 2σ ~ 0.954500, 3σ ~ 0.997300.
    fig = corner.corner(
        chain_plot,
        labels=selected_names,
        truths=truths,
        truth_color='red',
        range=[range_val] * n_params,
        plot_datapoints=False,
        plot_contours=True,
        fill_contours=False,
        levels=[0.682689, 0.954500, 0.997300],
        title_quantiles=[0.16, 0.5, 0.84],
        quantiles=[0.02,0.16, 0.5, 0.84, 0.98],
        bins=20,
    )

    if save_path:
        if verbose:
            print(f"  saving corner plot to {save_path}")
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            fig = None

    if return_fig:
        return fig

def plot_corner_chain_with_totals(sampler, param_names, ions=None, best_params=False,
                                    multiple=1.0, range_val=0.99, burnin=0,
                                    save_path=None, return_fig=False, verbose=False,
                                    fit=None, main_only=False,
                                    main_param_names=None, main_param_regex=None,
                                    exclude_param_regex=None):
    """
    Corner plot of the flattened chain augmented with total column densities
    for specified ions. If `ions` is None, all ions inferred from parameters
    named like "N_<ION>..." are included (order preserved by first appearance).

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
    param_names : list of str
        Full list of parameter names corresponding to the chain ordering.
    ions : list of str or None
        Ion names to sum (e.g. ['HI', 'CIII']). If None, all found ions are used.
    best_params : bool
        If True, mark the "best" sample (max sum of original parameters) as truths.
    multiple : float
        Scaling factor applied to N_ parameters before log10 (same behavior as other plotters).
    range_val : float
        Fraction of axis range to show for corner.
    burnin, save_path, return_fig, verbose : as in other corner plot helpers.
    fit : astropy.modeling.fitting.FittedModel or None
        Optional fitted model whose parameter values should be overplotted as
        the "truth" on the corner plot. When provided, the fitted parameter
        vector (and derived totals) take precedence over `best_params`.
    """

    # get flattened chain with burnin discarded (or use provided array)
    arr = _as_flat_chain(sampler, burnin=burnin, verbose=verbose)
    if verbose:
        print("plot_corner_chain_with_totals: flatchain.shape =", arr.shape)

    n_params = len(param_names)
    if arr.ndim != 2 or arr.shape[1] != n_params:
        raise ValueError(f"Expected chain shape (n_samples, {n_params}), got {arr.shape}")

    def _select_param_indices(names):
        indices = list(range(len(names)))

        # Always keep Student-t scale/dof parameters in this corner plot,
        # even when main_only filtering is active.
        tau_like_regex = re.compile(r"^(tau|nu|log10_tau|log10_nu)(?:_|$)")
        tau_keep = {n for n in names if tau_like_regex.search(str(n))}

        if main_only:
            if main_param_names is None and main_param_regex is None:
                raise ValueError("main_only=True requires main_param_names or main_param_regex")
            keep = set()
            if main_param_names is not None:
                keep.update([n for n in names if n in set(main_param_names)])
            if main_param_regex is not None:
                pattern = re.compile(main_param_regex)
                keep.update([n for n in names if pattern.search(str(n))])
            keep.update(tau_keep)
            indices = [i for i, n in enumerate(names) if n in keep]

        if exclude_param_regex is not None:
            pattern = re.compile(exclude_param_regex)
            indices = [i for i in indices if not pattern.search(str(names[i]))]

        if not indices:
            raise ValueError("No parameters selected for corner plot")
        return indices

    selected_indices = _select_param_indices(param_names)
    selected_names = [param_names[i] for i in selected_indices]
    arr = arr[:, selected_indices]
    param_names = list(selected_names)
    n_params = len(param_names)

    # discover N_ parameters and map them to ion tokens in order of first appearance
    ion_map = {}
    ions_seen = []
    for i, nm in enumerate(param_names):
        m = re.match(r"^N_([A-Za-z0-9]+)", nm)
        if m:
            ion = m.group(1)
            ion_map.setdefault(ion, []).append(i)
            if ion not in ions_seen:
                ions_seen.append(ion)

    if len(ion_map) == 0:
        raise ValueError("No parameters starting with 'N_' were found in param_names")

    # decide which ions to include
    if ions is None:
        ions_to_use = ions_seen
    else:
        ions_to_use = list(ions)
        # validate requested ions
        missing = [ion for ion in ions_to_use if ion not in ion_map]
        if missing:
            raise ValueError(f"Requested ions not found among N_ parameters: {missing}")

    if verbose:
        print("Including total column densities for ions:", ions_to_use)

    # compute totals per ion (sum over components) -> shape (n_samples, n_ions)
    totals = np.column_stack([arr[:, ion_map[ion]].sum(axis=1) for ion in ions_to_use])

    # take log10 of totals (and apply multiple) just like other N_ parameters
    # use nan_to_num to guard against zero/negative producing -inf/nan
    totals_logged = np.nan_to_num(np.log10(totals * multiple))

    # build augmented array and names (append logged totals)
    total_names = [f"TOTAL_N_{ion}" for ion in ions_to_use]
    augmented = np.hstack((arr, totals_logged))
    augmented_names = list(param_names) + total_names

    if verbose:
        print("Augmented chain shape:", augmented.shape)
        print("Augmented parameter names (tail):", augmented_names[-len(total_names):])

    # apply transforms (log10 for N_ and T_ as elsewhere) to the augmented array
    chain_plot = transform_chain_plot(augmented, augmented_names, multiple)

    # compute "best" truths if requested (use original arr to pick index)
    truths = None
    if fit is not None:
        if not hasattr(fit, 'param_names') or not hasattr(fit, 'parameters'):
            raise TypeError("`fit` must be an astropy fitted model with 'param_names' and 'parameters' attributes")
        fit_param_map = {name: val for name, val in zip(fit.param_names, fit.parameters)}
        best_values = np.zeros(len(augmented_names))
        missing = []
        for i, name in enumerate(param_names):
            val = fit_param_map.get(name)
            if val is None:
                # Chain can include nuisance parameters not present in the
                # deterministic fit (e.g. log10_nu for Student-t likelihood).
                # Fall back to the chain median so we can still draw truths.
                best_values[i] = np.nanmedian(arr[:, i])
                missing.append(name)
                continue
            if hasattr(val, 'value'):
                val = val.value
            best_values[i] = val
        if missing and verbose:
            print(
                "Fitted model missing parameters; using chain medians for:",
                missing,
            )
        # derive total column densities from the fit
        for j, ion in enumerate(ions_to_use):
            total = 0.0
            for idx in ion_map[ion]:
                total += best_values[idx]
            best_values[n_params + j] = np.nan_to_num(np.log10(total * multiple))
        truths = transform_chain_plot(best_values.reshape(1, -1), augmented_names, multiple)[0]
        if verbose:
            print("Using fitted model parameters as truths on the corner plot")
    elif best_params:
        idx_max = np.nanargmax(np.nansum(arr[:, :n_params], axis=1))
        truths = chain_plot[idx_max].copy()
        if verbose:
            print("best_params index:", idx_max)

    # produce corner plot
    fig = corner.corner(
        chain_plot,
        labels=augmented_names,
        truths=truths,
        truth_color='red',
        range=[range_val] * len(augmented_names),
        plot_datapoints=False,
        plot_contours=True,
        fill_contours=False,
        levels=[0.682689, 0.954500, 0.997300],
        bins=20,
    )

    if save_path:
        if verbose:
            print("Saving augmented corner plot to", save_path)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            plt.close(fig)
            fig = None

    if return_fig:
        return fig

def plot_corner_velocities(sampler, param_names, vel_prefix='v_',
                           burnin=0, save_path=None, return_fig=False,
                           verbose=False, range_val=0.99,
                           main_only=False, main_param_names=None,
                           main_param_regex=None, exclude_param_regex=None):
    """
    Make a corner plot that only includes velocity parameters (by prefix)
    and draws a dotted diagonal line (y = x) on each off-diagonal panel to
    indicate equality between velocities. Helpful for debugging ordering.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The sampler containing the chain.
    param_names : list of str
        Full list of parameter names corresponding to the chain ordering.
    vel_prefix : str
        Prefix identifying velocity parameters (default 'v_').
    burnin : int
        Steps to discard from the start of each chain.
    save_path : str or None
        If provided, save the figure to this path.
    return_fig : bool
        If True, return the matplotlib Figure object.
    verbose : bool
        If True, print debug messages.
    range_val : float
        Fraction of axis range to show for corner library.
    """
    # get flattened chain after burnin (or use provided array)
    arr = _as_flat_chain(sampler, burnin=burnin, verbose=verbose)
    if verbose:
        print("plot_corner_velocities: flatchain.shape =", arr.shape)

    def _select_param_indices(names):
        indices = list(range(len(names)))

        if main_only:
            if main_param_names is None and main_param_regex is None:
                raise ValueError("main_only=True requires main_param_names or main_param_regex")
            keep = set()
            if main_param_names is not None:
                keep.update([n for n in names if n in set(main_param_names)])
            if main_param_regex is not None:
                pattern = re.compile(main_param_regex)
                keep.update([n for n in names if pattern.search(str(n))])
            indices = [i for i, n in enumerate(names) if n in keep]

        if exclude_param_regex is not None:
            pattern = re.compile(exclude_param_regex)
            indices = [i for i in indices if not pattern.search(str(names[i]))]

        if not indices:
            raise ValueError("No parameters selected for corner plot")
        return indices

    selected_indices = _select_param_indices(param_names)
    selected_names = [param_names[i] for i in selected_indices]

    # determine velocity parameter indices and names
    vel_idx = [i for i, n in enumerate(selected_names) if n.startswith(vel_prefix)]
    vel_names = [selected_names[i] for i in vel_idx]
    if len(vel_idx) == 0:
        raise ValueError(f"No parameters starting with '{vel_prefix}' found in param_names")

    data = arr[:, [selected_indices[i] for i in vel_idx]]
    if verbose:
        print(f"Selected {len(vel_names)} velocity parameters:", vel_names)

    # produce corner plot
    fig = corner.corner(
        data,
        labels=vel_names,
        range=[range_val] * len(vel_names),
        plot_datapoints=False,
        plot_contours=True,
        fill_contours=False,
        levels=[0.682689, 0.954500, 0.997300],
        bins=20,
    )

    # Draw dotted diagonal y=x lines on each off-diagonal panel.
    axes = fig.axes
    n = len(vel_names)
    # corner can return either full n*n axes or lower-triangle only.
    try:
        if len(axes) == n * n:
            ax_mat = np.array(axes).reshape((n, n))
        else:
            # lower-triangle mapping: axes are provided row by row for j<=i
            ax_mat = np.empty((n, n), dtype=object)
            k = 0
            for i in range(n):
                for j in range(0, i + 1):
                    ax_mat[i, j] = axes[k]
                    k += 1
    except Exception:
        # fallback: just iterate over axes and try to draw a diagonal on all
        for ax in axes:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            lo = max(min(x0, y0), -1e9)
            hi = min(max(x1, y1), 1e9)
            xs = np.linspace(lo, hi, 2)
            try:
                ax.plot(xs, xs, ':', color='k', linewidth=0.8)
            except Exception:
                pass
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if return_fig:
            return fig
        return

    # iterate over lower-triangle off-diagonal panels and draw y=x
    for i in range(n):
        for j in range(0, i):
            ax = ax_mat[i, j]
            if ax is None:
                continue
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            lo = max(min(x0, y0), -1e9)
            hi = min(max(x1, y1), 1e9)
            xs = np.linspace(lo, hi, 2)
            try:
                ax.plot(xs, xs, ':', color='k', linewidth=0.8)
            except Exception:
                if verbose:
                    print(f"Could not draw diagonal on axes ({i},{j})")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if verbose:
            print("Saved velocity corner plot to", save_path)

    if return_fig:
        return fig


def plot_mcmc_draws(sampler, model, spectrum, line_list, z, save_path,
                    n_draws=100, vrange=300, n=3,
                    skip_submodels=True, alpha=0.3, verbose=False, burnin=0,
                    submodel_colors=None, log_probs=None,
                    plot_submodel_bands=True, plot_continuum_bands=True,
                    continuum_band_color='tab:orange',
                    bestfit_method='max_likelihood', ls_params=None,
                    band_method='pointwise', curvewise_metric='rms',
                    plot_extended_bands=False,
                    band95_alpha=0.18, band995_alpha=0.08):
    """
    Plot posterior bands (68%) for the total model and optional submodels.

    Parameters:
      sampler       : emcee.EnsembleSampler instance or chain array
      model         : compound model with .param_names, .fixed, .bounds
      spectrum      : Spectrum1D object
      line_list     : list of transitions
      z             : redshift
      save_name     : base filename for saving
      n_draws       : number of samples used to estimate bands
      vrange        : velocity range for plotting
      n             : binning of the data
      skip_submodels: if True, only plot total model
      alpha         : transparency for bands
      verbose       : bool, if True print additional information during processing
      log_probs     : optional log-prob array for choosing max-likelihood sample
      plot_submodel_bands: if True, draw bands for submodels
    band_method   : 'pointwise' for per-pixel 16-84% percentiles (default),
                'curvewise' for curve-ranked 68% credible envelope
    curvewise_metric : 'rms' or 'max' — distance metric for curvewise ranking
    plot_extended_bands : if True, also plot 95% and 99.5% bands
    band95_alpha  : alpha for the 95% bands
    band995_alpha : alpha for the 99.5% bands
    """
    arr = _as_flat_chain(sampler, burnin=burnin, verbose=verbose)
    chain = np.asarray(arr)
    if chain.ndim != 2:
        raise ValueError("plot_mcmc_draws expects a 2D chain array")

    if verbose:
        print("plot_mcmc_draws: flatchain.shape =", chain.shape)

    sample_count = min(int(n_draws), chain.shape[0])
    if sample_count <= 0:
        raise ValueError("n_draws must be >= 1")
    idx = np.random.choice(chain.shape[0], size=sample_count, replace=False)
    sample_params = chain[idx]

    best_params = None
    if bestfit_method == 'least_squares' and ls_params is not None:
        best_params = np.asarray(ls_params, dtype=float)
        if verbose:
            print("plot_mcmc_draws: using least-squares parameters for best-fit curve")
    elif bestfit_method == 'max_likelihood' and log_probs is not None:
        log_probs = np.asarray(log_probs).reshape(-1)
        if log_probs.shape[0] == chain.shape[0]:
            best_idx = int(np.nanargmax(log_probs))
            best_params = chain[best_idx]
            if verbose:
                print(f"plot_mcmc_draws: best_idx={best_idx} from log_probs")
    elif bestfit_method == 'median':
        best_params = np.nanmedian(chain, axis=0)
        if verbose:
            print("plot_mcmc_draws: using median of chain for best-fit curve")

    if best_params is None:
        if log_probs is not None:
            log_probs = np.asarray(log_probs).reshape(-1)
            if log_probs.shape[0] == chain.shape[0]:
                best_idx = int(np.nanargmax(log_probs))
                best_params = chain[best_idx]
                if verbose:
                    print(f"plot_mcmc_draws: fallback to max_likelihood, best_idx={best_idx}")
        if best_params is None:
            best_params = np.nanmedian(sample_params, axis=0)
            if verbose:
                print("plot_mcmc_draws: fallback to median parameters for best-fit curve")

    def _make_velocity_grid(line_wave):
        if hasattr(line_wave, "to_value"):
            line_wave_value = line_wave.to_value(u.AA)
        else:
            line_wave_value = float(line_wave)
        full_velocity = to_velocity(spectrum.spectral_axis, line_wave_value, z).to('km/s').value
        window = (full_velocity >= -vrange) & (full_velocity <= vrange)
        if np.count_nonzero(window) >= 2:
            step = np.median(np.diff(full_velocity[window]))
        elif full_velocity.size >= 2:
            step = np.median(np.diff(full_velocity))
        else:
            step = 1.0
        step = max(abs(step), 0.1)
        return np.arange(-vrange, vrange + step, step)

    def _interp_model_to_velocity(model_values, line_wave, v_grid):
        if hasattr(line_wave, "to_value"):
            line_wave_value = line_wave.to_value(u.AA)
        else:
            line_wave_value = float(line_wave)
        velocity_axis = to_velocity(spectrum.spectral_axis, line_wave_value, z).to('km/s').value
        sort_idx = np.argsort(velocity_axis)
        v_sorted = velocity_axis[sort_idx]
        m_sorted = np.asarray(model_values, dtype=float)[sort_idx]
        return np.interp(v_grid, v_sorted, m_sorted, left=np.nan, right=np.nan)

    def _curvewise_band(samples_2d, credible_frac=0.68):
        """Return (lo, hi) envelope from the central `credible_frac` of curves,
        ranked by scalar distance to the per-pixel median reference."""
        ref = np.nanmedian(samples_2d, axis=0)
        diffs = samples_2d - ref[np.newaxis, :]
        if curvewise_metric == 'max':
            distances = np.nanmax(np.abs(diffs), axis=1)
        else:  # 'rms'
            distances = np.sqrt(np.nanmean(diffs**2, axis=1))
        n_keep = max(1, int(np.ceil(credible_frac * samples_2d.shape[0])))
        keep_idx = np.argsort(distances)[:n_keep]
        kept = samples_2d[keep_idx]
        return np.nanmin(kept, axis=0), np.nanmax(kept, axis=0)

    def _compute_band(samples_2d, credible_frac=0.68):
        """Dispatch to pointwise or curvewise band."""
        if band_method == 'curvewise':
            return _curvewise_band(samples_2d, credible_frac=credible_frac)
        if credible_frac == 0.995:
            return np.nanpercentile(samples_2d, [0.25, 99.75], axis=0)
        if credible_frac == 0.95:
            return np.nanpercentile(samples_2d, [2.5, 97.5], axis=0)
        return np.nanpercentile(samples_2d, [16, 84], axis=0)

    def _draw_bands(ax, v_grid, bands, color, alpha68):
        if plot_extended_bands:
            lo995, hi995 = bands['99.5']
            lo95, hi95 = bands['95']
            ax.fill_between(v_grid, lo995, hi995, color=color, alpha=band995_alpha, linewidth=0)
            ax.fill_between(v_grid, lo95, hi95, color=color, alpha=band95_alpha, linewidth=0)
        lo68, hi68 = bands['68']
        ax.fill_between(v_grid, lo68, hi68, color=color, alpha=alpha68, linewidth=0)

    fig, axes = plt.subplots(len(line_list), 1,
                             figsize=(8, 2 * len(line_list)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    if axes.ndim > 1:
        axes = axes.reshape(-1)
    plot_absorber(fig, axes, spectrum, line_list, z,
                  vrange=vrange, n=n)

    local_skip_submodels = skip_submodels
    if (submodel_colors is not None) and local_skip_submodels:
        if verbose:
            print("plot_mcmc_draws: submodel_colors provided -> enabling submodel plotting")
        local_skip_submodels = False

    voigt_components = _collect_voigt_components(model) if not local_skip_submodels else []
    lsf_component = None if local_skip_submodels else _find_lsf_component(model)

    continuum_components = _collect_continuum_components(model) if plot_continuum_bands else []
    if verbose and plot_continuum_bands:
        print(f"plot_mcmc_draws: found {len(continuum_components)} continuum components")

    for row, line in enumerate(line_list):
        line_wave = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0] * u.AA
        v_grid = _make_velocity_grid(line_wave)

        model_samples = np.empty((sample_params.shape[0], v_grid.size), dtype=float)
        for i, pars in enumerate(sample_params):
            fitter_to_model_params(model, pars, use_min_max_bounds=True)
            full_vals = np.asarray(model(spectrum.spectral_axis.value))
            model_samples[i] = _interp_model_to_velocity(full_vals, line_wave, v_grid)

        fitter_to_model_params(model, best_params, use_min_max_bounds=True)
        best_full = np.asarray(model(spectrum.spectral_axis.value))
        best_vals = _interp_model_to_velocity(best_full, line_wave, v_grid)

        bands = {
            '68': _compute_band(model_samples, credible_frac=0.68),
            '95': _compute_band(model_samples, credible_frac=0.95),
            '99.5': _compute_band(model_samples, credible_frac=0.995),
        }

        ax = axes[row]
        _draw_bands(ax, v_grid, bands, color='0.7', alpha68=alpha)
        ax.plot(v_grid, best_vals, color='green', linewidth=1.5)

        if plot_submodel_bands and not local_skip_submodels:
            for j, voigt in enumerate(voigt_components):
                voigt_line_z = float(getattr(voigt, '_z', z))
                voigt_lines = getattr(voigt, '_line_list', ()) or ()
                is_contaminant = not np.isclose(voigt_line_z, z, atol=1e-6)
                if not is_contaminant and line not in voigt_lines:
                    continue

                lsf_voigt = generic_exp(voigt)
                if lsf_component is not None:
                    lsf_voigt = lsf_voigt | lsf_component

                sub_samples = np.empty((sample_params.shape[0], v_grid.size), dtype=float)
                for i, pars in enumerate(sample_params):
                    fitter_to_model_params(model, pars, use_min_max_bounds=True)
                    sub_full = np.asarray(lsf_voigt(spectrum.spectral_axis.value))
                    sub_samples[i] = _interp_model_to_velocity(sub_full, line_wave, v_grid)
                fitter_to_model_params(model, best_params, use_min_max_bounds=True)
                sub_best_full = np.asarray(lsf_voigt(spectrum.spectral_axis.value))
                sub_best_vals = _interp_model_to_velocity(sub_best_full, line_wave, v_grid)
                sub_bands = {
                    '68': _compute_band(sub_samples, credible_frac=0.68),
                    '95': _compute_band(sub_samples, credible_frac=0.95),
                    '99.5': _compute_band(sub_samples, credible_frac=0.995),
                }

                chosen_color = None
                name_key = getattr(voigt, 'name', None) or f'comp_{j}'
                if isinstance(submodel_colors, dict):
                    if name_key in submodel_colors:
                        chosen_color = submodel_colors[name_key]
                    else:
                        base = name_key.split('__')[0]
                        if base in submodel_colors:
                            chosen_color = submodel_colors[base]
                if chosen_color is None:
                    chosen_color = 'tab:red' if is_contaminant else 'tab:green'

                _draw_bands(ax, v_grid, sub_bands, color=chosen_color, alpha68=0.12)
                ax.plot(v_grid, sub_best_vals, color=chosen_color, linewidth=1.0, linestyle=':')

        # Plot continuum bands
        if plot_continuum_bands and continuum_components:
            cont_samples = np.empty((sample_params.shape[0], v_grid.size), dtype=float)
            for i, pars in enumerate(sample_params):
                fitter_to_model_params(model, pars, use_min_max_bounds=True)
                cont_vals = np.ones_like(spectrum.spectral_axis.value, dtype=float)
                for comp in continuum_components:
                    cont_vals += np.asarray(comp(spectrum.spectral_axis.value), dtype=float) - 1.0
                cont_samples[i] = _interp_model_to_velocity(cont_vals, line_wave, v_grid)
            fitter_to_model_params(model, best_params, use_min_max_bounds=True)
            cont_best = np.ones_like(spectrum.spectral_axis.value, dtype=float)
            for comp in continuum_components:
                cont_best += np.asarray(comp(spectrum.spectral_axis.value), dtype=float) - 1.0
            cont_best_v = _interp_model_to_velocity(cont_best, line_wave, v_grid)
            cont_lo, cont_hi = _compute_band(cont_samples)
            ax.fill_between(v_grid, cont_lo, cont_hi, color=continuum_band_color, alpha=0.15, linewidth=0)
            ax.plot(v_grid, cont_best_v, color=continuum_band_color, linewidth=1.0, linestyle='--')

        ax.set_xlim(-vrange, vrange)
        ax.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    if verbose:
        print("  saved mcmc bands to", save_path)
    plt.close(fig)


def plot_bestfit_grid_from_chains(
    models,
    spectra,
    line_list,
    z_list,
    chain_paths,
    save_path,
    log_prob_paths=None,
    param_name_paths=None,
    n_samples=200,
    vrange=300,
    n=3,
    oversample=4,
    seed=None,
    bestfit_color='green',
    band_color='0.7',
    band_alpha=0.35,
    skip_submodels=True,
    submodel_colors=None,
    plot_submodel_bands=True,
    verbose=False,
):
    """
    Plot a grid of best-fit spectra from saved MCMC chains with 68% bands.

    Parameters
    ----------
    models : list
        List of compound models (one per column).
    spectra : list
        List of Spectrum1D objects (one per column).
    line_list : list
        List of line names to plot (rows).
    z_list : list
        List of redshifts (one per column).
    chain_paths : list
        Paths to saved chains (.npy) for each model.
    save_path : str
        Path to save the output figure.
    log_prob_paths : list, optional
        Paths to saved log-probability arrays (.npy). Required to select
        maximum-likelihood samples.
    param_name_paths : list, optional
        Paths to saved parameter name arrays (.npy) for chain ordering.
    n_samples : int
        Number of chain samples to draw for uncertainty bands.
    vrange : float
        Velocity range (km/s) for plotting.
    n : int
        Binning for observed data.
    oversample : int
        Oversampling factor for smooth model/band curves.
    seed : int, optional
        RNG seed for reproducibility.
    """
    if not isinstance(models, (list, tuple)):
        raise ValueError("models must be a list or tuple")
    n_models = len(models)
    if n_models == 0:
        raise ValueError("models list is empty")
    if len(spectra) != n_models or len(z_list) != n_models or len(chain_paths) != n_models:
        raise ValueError("spectra, z_list, and chain_paths must match models length")

    if log_prob_paths is None:
        raise ValueError("log_prob_paths is required to select the maximum-likelihood sample")
    if len(log_prob_paths) != n_models:
        raise ValueError("log_prob_paths must match models length")

    if param_name_paths is not None and len(param_name_paths) != n_models:
        raise ValueError("param_name_paths must match models length when provided")

    rng = np.random.default_rng(seed)
    timer_total_start = time.perf_counter()

    def _free_param_names(model):
        return [
            pname for pname in model.param_names
            if not model.fixed[pname] and not model.tied[pname]
        ]

    def _load_param_names(path):
        if path is None:
            return None
        names = np.load(path, allow_pickle=True)
        if isinstance(names, np.ndarray):
            names = names.tolist()
        return [str(n) for n in names]

    def _maybe_reorder_chain(chain, model, param_names):
        free_names = _free_param_names(model)
        if chain.shape[1] == len(free_names):
            if param_names is None:
                return chain
            # reorder only if names differ
            if list(map(str, param_names)) == list(map(str, free_names)):
                return chain
        if param_names is None:
            raise ValueError("Chain shape does not match free parameters and no param names were provided")
        name_to_idx = {str(name): i for i, name in enumerate(param_names)}
        missing = [name for name in free_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Param names missing free parameters: {missing}")
        order = [name_to_idx[name] for name in free_names]
        return chain[:, order]

    def _make_velocity_grid(spectrum, line_wave, z, vrange, oversample):
        if hasattr(line_wave, "to_value"):
            line_wave_value = line_wave.to_value(u.AA)
        else:
            line_wave_value = float(line_wave)
        full_velocity = to_velocity(spectrum.spectral_axis, line_wave_value, z).to('km/s').value
        window = (full_velocity >= -vrange) & (full_velocity <= vrange)
        if np.count_nonzero(window) >= 2:
            step = np.median(np.diff(full_velocity[window]))
        elif full_velocity.size >= 2:
            step = np.median(np.diff(full_velocity))
        else:
            step = 1.0
        step = max(abs(step) / max(int(oversample), 1), 0.1)
        v_grid = np.arange(-vrange, vrange + step, step)
        return v_grid

    def _velocity_to_wavelength(v_grid, line_wave, z):
        c_kms = c.to(u.km / u.s).value
        if hasattr(line_wave, "to_value"):
            lambda0 = (line_wave * (1 + z)).to_value(u.AA)
        else:
            lambda0 = float(line_wave) * (1 + z)
        return lambda0 * (1.0 + v_grid / c_kms)

    def _interpolate_model_to_velocity(model_values, spectrum_axis, line_wave, z, v_grid):
        if hasattr(line_wave, "to_value"):
            line_wave_value = line_wave.to_value(u.AA)
        else:
            line_wave_value = float(line_wave)
        velocity_axis = to_velocity(spectrum_axis, line_wave_value, z).to('km/s').value
        sort_idx = np.argsort(velocity_axis)
        v_sorted = velocity_axis[sort_idx]
        m_sorted = np.asarray(model_values, dtype=float)[sort_idx]
        return np.interp(v_grid, v_sorted, m_sorted, left=np.nan, right=np.nan)

    n_rows = len(line_list)
    if n_rows == 0:
        raise ValueError("line_list is empty")

    fig, axes = plt.subplots(
        n_rows,
        n_models,
        figsize=(4.5 * n_models, 2.2 * n_rows),
        sharex=False,
        sharey=False,
    )

    axes = np.atleast_2d(axes)

    for col in range(n_models):
        timer_model_start = time.perf_counter()
        model = models[col]
        spectrum = spectra[col]
        z = z_list[col]

        if verbose:
            print(f"[bestfit_grid] Loading chain/log_prob for column {col + 1}/{n_models}")
        chain = np.load(chain_paths[col])
        log_prob = np.load(log_prob_paths[col])
        param_names = _load_param_names(param_name_paths[col]) if param_name_paths is not None else None
        chain = _as_flat_chain(chain, burnin=0, verbose=False)
        log_prob = np.asarray(log_prob).reshape(-1)

        chain = _maybe_reorder_chain(chain, model, param_names)

        finite = np.isfinite(log_prob)
        if not np.any(finite):
            raise ValueError(f"No finite log_prob values for model index {col}")
        log_prob_finite = log_prob[finite]
        chain_finite = chain[finite]

        best_idx = int(np.argmax(log_prob_finite))
        best_params = chain_finite[best_idx]

        sample_count = min(int(n_samples), chain_finite.shape[0])
        sample_idx = rng.choice(chain_finite.shape[0], size=sample_count, replace=False)
        if best_idx not in sample_idx:
            sample_idx[0] = best_idx
        sample_params = chain_finite[sample_idx]
        if verbose:
            print(
                f"[bestfit_grid] Column {col + 1}: chain={chain.shape}, finite={chain_finite.shape}, "
                f"samples={sample_count}, best_idx={best_idx}"
            )

        axes_col = axes[:, col]
        plot_absorber(
            fig,
            axes_col,
            spectrum,
            line_list,
            z,
            vrange=vrange,
            n=n,
        )

        voigt_components = _collect_voigt_components(model) if not skip_submodels else []
        lsf_component = None if skip_submodels else _find_lsf_component(model)

        for row, line in enumerate(line_list):
            line_wave = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0] * u.AA
            v_grid = _make_velocity_grid(spectrum, line_wave, z, vrange, oversample)
            model_samples = np.empty((sample_params.shape[0], v_grid.size), dtype=float)
            for i, pars in enumerate(sample_params):
                fitter_to_model_params(model, pars, use_min_max_bounds=True)
                model_full = np.asarray(model(spectrum.spectral_axis.value))
                model_samples[i] = _interpolate_model_to_velocity(
                    model_full,
                    spectrum.spectral_axis,
                    line_wave,
                    z,
                    v_grid,
                )

            fitter_to_model_params(model, best_params, use_min_max_bounds=True)
            best_full = np.asarray(model(spectrum.spectral_axis.value))
            best_vals = _interpolate_model_to_velocity(
                best_full,
                spectrum.spectral_axis,
                line_wave,
                z,
                v_grid,
            )

            lo, hi = np.nanpercentile(model_samples, [16, 84], axis=0)

            ax = axes_col[row]
            ax.fill_between(v_grid, lo, hi, color=band_color, alpha=band_alpha, linewidth=0)
            ax.plot(v_grid, best_vals, color=bestfit_color, linewidth=1.5)
            ax.set_xlim(-vrange, vrange)

            if plot_submodel_bands and not skip_submodels:
                for j, voigt in enumerate(voigt_components):
                    voigt_line_z = float(getattr(voigt, '_z', z))
                    voigt_lines = getattr(voigt, '_line_list', ()) or ()
                    is_contaminant = not np.isclose(voigt_line_z, z, atol=1e-6)
                    if not is_contaminant and line not in voigt_lines:
                        continue

                    lsf_voigt = generic_exp(voigt)
                    if lsf_component is not None:
                        lsf_voigt = lsf_voigt | lsf_component

                    sub_samples = np.empty((sample_params.shape[0], v_grid.size), dtype=float)
                    for i, pars in enumerate(sample_params):
                        fitter_to_model_params(model, pars, use_min_max_bounds=True)
                        sub_full = np.asarray(lsf_voigt(spectrum.spectral_axis.value))
                        sub_samples[i] = _interpolate_model_to_velocity(
                            sub_full,
                            spectrum.spectral_axis,
                            line_wave,
                            z,
                            v_grid,
                        )
                    fitter_to_model_params(model, best_params, use_min_max_bounds=True)
                    sub_best_full = np.asarray(lsf_voigt(spectrum.spectral_axis.value))
                    sub_best_vals = _interpolate_model_to_velocity(
                        sub_best_full,
                        spectrum.spectral_axis,
                        line_wave,
                        z,
                        v_grid,
                    )
                    sub_lo, sub_hi = np.nanpercentile(sub_samples, [16, 84], axis=0)

                    # choose color
                    chosen_color = None
                    name_key = getattr(voigt, 'name', None)
                    if name_key is None:
                        name_key = f'comp_{j}'
                    if isinstance(submodel_colors, dict):
                        if name_key in submodel_colors:
                            chosen_color = submodel_colors[name_key]
                        else:
                            base = name_key.split('__')[0]
                            if base in submodel_colors:
                                chosen_color = submodel_colors[base]
                    if chosen_color is None:
                        chosen_color = 'tab:red' if is_contaminant else 'tab:green'

                    ax.fill_between(v_grid, sub_lo, sub_hi, color=chosen_color, alpha=0.12, linewidth=0)
                    ax.plot(v_grid, sub_best_vals, color=chosen_color, linewidth=1.0, linestyle=':')

        local_skip_submodels = skip_submodels
        if submodel_colors is not None and local_skip_submodels:
            local_skip_submodels = False
        if not local_skip_submodels:
            plot_model(
                fig,
                axes_col,
                spectrum,
                line_list,
                model,
                z,
                vrange=vrange,
                skip_submodels=False,
                alpha=0.5,
                submodel_colors=submodel_colors,
            )

        timer_model_end = time.perf_counter()
        if verbose:
            print(f"[bestfit_grid] Column {col + 1} done in {timer_model_end - timer_model_start:.2f}s")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if verbose:
        print("Saved best-fit grid to", save_path)
        timer_total_end = time.perf_counter()
        print(f"[bestfit_grid] Total time: {timer_total_end - timer_total_start:.2f}s")
    plt.close(fig)


def plot_log_prob_histogram_with_cdf(log_probs, ls_log_prob=None, bins=50,
                                     save_path=None, return_fig=False,
                                     verbose=False, xlim=None):
    """Histogram of log-probabilities with cumulative fraction overlay."""

    values = np.ravel(np.asarray(log_probs))
    mask = np.isfinite(values)
    if not np.all(mask):
        if verbose:
            print(f"plot_log_prob_histogram_with_cdf: dropping {values.size - np.count_nonzero(mask)} non-finite entries")
        values = values[mask]

    if values.size == 0:
        raise ValueError("No finite log-probability values available")

    hist_kwargs = {}
    hist_range = None
    if xlim is not None:
        if len(xlim) != 2:
            raise ValueError("xlim must be a (min, max) tuple")
        xmin, xmax = xlim
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            raise ValueError("xlim must contain finite values with xmin < xmax")
        hist_range = (xmin, xmax)
        hist_kwargs["range"] = hist_range

    counts, edges = np.histogram(values, bins=bins, **hist_kwargs)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0
    cdf = np.cumsum(counts) / np.sum(counts)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(centers, counts, width=widths, color='steelblue', alpha=0.7, edgecolor='black', label='histogram')

    if ls_log_prob is not None and np.isfinite(ls_log_prob):
        ax.axvline(ls_log_prob, color='red', linestyle='--', linewidth=1.5, label='least-squares value')

    ax.set_xlabel('log probability')
    ax.set_ylabel('count')
    ax.set_title('Log-probability volume diagnostic')

    ax2 = ax.twinx()
    ax2.plot(centers, cdf, color='darkorange', linewidth=1.5, label='cumulative fraction')
    ax2.set_ylabel('cumulative fraction')
    ax2.set_ylim(0.0, 1.0)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines or lines2:
        ax.legend(lines + lines2, labels + labels2, loc='lower right')

    if hist_range is not None:
        ax.set_xlim(hist_range)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            plt.close(fig)
            return None

    if return_fig:
        return fig


def plot_log_prob_vs_distance(sampler_or_chain, log_probs=None, reference=None,
                              burnin=0, save_path=None, return_fig=False,
                              verbose=False, max_samples=100000,
                              ls_log_prob=None):
    """Hexbin of log-probability against Mahalanobis distance from reference."""

    chain, values = _prepare_chain_and_log_prob(sampler_or_chain, log_probs=log_probs,
                                                burnin=burnin, verbose=verbose)

    if reference is None:
        reference_vec = np.nanmean(chain, axis=0)
    else:
        reference_vec = np.asarray(reference, dtype=float)
        if reference_vec.shape[0] != chain.shape[1]:
            raise ValueError("reference vector length does not match number of parameters")

    deltas = chain - reference_vec
    cov = np.cov(chain, rowvar=False)
    reg = max(np.nanmean(np.diag(cov)), 1.0) * 1e-9
    inv_cov = np.linalg.pinv(cov + np.eye(cov.shape[0]) * reg)
    mahal_sq = np.einsum('ij,jk,ik->i', deltas, inv_cov, deltas)
    distances = np.sqrt(np.clip(mahal_sq, 0.0, None))

    idx = np.arange(distances.size)
    if distances.size > max_samples:
        rng = np.random.default_rng(12345)
        idx = rng.choice(distances.size, size=max_samples, replace=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    hb = ax.hexbin(distances[idx], values[idx], gridsize=60, cmap='viridis')
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('sample count')
    ax.set_xlabel('Mahalanobis distance from reference')
    ax.set_ylabel('log probability')

    if ls_log_prob is not None and np.isfinite(ls_log_prob):
        ax.axhline(ls_log_prob, color='red', linestyle='--', linewidth=1.2, label='least-squares value')
        ax.legend(loc='lower left')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            plt.close(fig)
            return None

    if return_fig:
        return fig


def plot_log_prob_pairwise_scatter(sampler_or_chain, param_names, log_probs=None,
                                   burnin=0, pairs=None, max_points=20000,
                                   reference=None, save_path=None,
                                   return_fig=False, verbose=False):
    """Scatter subplots coloured by log-probability for selected parameter pairs."""

    chain, values = _prepare_chain_and_log_prob(sampler_or_chain, log_probs=log_probs,
                                                burnin=burnin, verbose=verbose)

    name_to_index = {name: idx for idx, name in enumerate(param_names)}

    if pairs is None:
        combos = list(itertools.combinations(param_names[:min(5, len(param_names))], 2))
        pairs = combos[:min(4, len(combos))]
    else:
        normalized = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("pairs must be an iterable of 2-tuples")
            normalized.append(tuple(pair))
        pairs = normalized

    if not pairs:
        raise ValueError("No parameter pairs provided for plotting")

    indices = []
    for x_name, y_name in pairs:
        if x_name not in name_to_index or y_name not in name_to_index:
            raise ValueError(f"Parameters {x_name} and/or {y_name} not found in param_names")
        indices.append((name_to_index[x_name], name_to_index[y_name]))

    idx = np.arange(values.size)
    if values.size > max_points:
        rng = np.random.default_rng(67890)
        idx = rng.choice(values.size, size=max_points, replace=False)

    chain_sub = chain[idx]
    values_sub = values[idx]

    n_panels = len(indices)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    reference_vec = None
    if reference is not None:
        reference_vec = np.asarray(reference, dtype=float)
        if reference_vec.shape[0] != chain.shape[1]:
            raise ValueError("reference vector length does not match number of parameters")

    scatter_artist = None
    for ax, (pair, (ix, iy)) in zip(axes_flat, zip(pairs, indices)):
        scatter_artist = ax.scatter(chain_sub[:, ix], chain_sub[:, iy], c=values_sub,
                                    s=8, cmap='viridis', alpha=0.6, linewidths=0)
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
        if reference_vec is not None:
            ax.scatter(reference_vec[ix], reference_vec[iy], marker='*', color='red',
                       s=80, edgecolors='black', linewidths=0.5, zorder=5)

    for leftover_ax in axes_flat[n_panels:]:
        leftover_ax.remove()

    if scatter_artist is not None:
        fig.colorbar(scatter_artist, ax=axes_flat[:n_panels], label='log probability', shrink=0.9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            plt.close(fig)
            return None

    if return_fig:
        return fig


def plot_log_prob_running_stats(sampler, ls_log_prob=None, burnin=0,
                                save_path=None, return_fig=False,
                                verbose=False):
    """Plot per-step mean, per-step max, and running max of sampler log-probabilities."""

    if not hasattr(sampler, "get_log_prob"):
        raise TypeError("Sampler must provide get_log_prob for running statistics")

    logp = sampler.get_log_prob(discard=burnin)
    if logp.ndim != 2:
        raise ValueError("Expected log-prob array with shape (nsteps, nwalkers)")

    steps = np.arange(logp.shape[0]) + 1
    step_mean = np.mean(logp, axis=1)
    step_max = np.max(logp, axis=1)
    running_max = np.maximum.accumulate(step_max)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, step_mean, color='steelblue', label='per-step mean')
    ax.plot(steps, step_max, color='seagreen', linestyle='--', label='per-step max')
    ax.plot(steps, running_max, color='darkorange', label='running max')

    if ls_log_prob is not None and np.isfinite(ls_log_prob):
        ax.axhline(ls_log_prob, color='red', linestyle='--', linewidth=1.2, label='least-squares value')

    ax.set_xlabel('step')
    ax.set_ylabel('log probability')
    ax.set_title('Sampler log-probability trajectory')
    ax.legend(loc='lower right')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            plt.close(fig)
            return None

    if return_fig:
        return fig

def transform_chain_plot(chain, names, multiple):
    """
    Apply log10 (and optional scaling) to N_ and T_ parameters in the chain.
    Works for arrays of shape (..., n_params): always loops over the last axis.
    """

    arr = np.array(chain, copy=True)
    if arr.ndim < 2:
        raise ValueError("transform_chain_plot: chain must have at least 2 dimensions")
    n_params = arr.shape[-1]
    if n_params != len(names):
        raise ValueError(f"transform_chain_plot: expected last dimension size {len(names)}, got {n_params}")

    print(arr.shape)
    for i, pname in enumerate(names):
        slice_i = (..., i)
        data = arr[slice_i]
        if pname.startswith('N_'):
            arr[slice_i] = np.nan_to_num(np.log10(data * np.round(multiple)))
        elif pname.startswith('T_'):
            arr[slice_i] = np.nan_to_num(np.log10(data))
        # else: leave data unchanged

    return arr

def plot_prior_corner(samples, param_names, range_val=0.99,
                      multiple=1.0, save_path=None, return_fig=False, verbose=False):
    """
    Produce a corner plot of samples drawn from the prior.

    Parameters
    ----------
    samples : array-like, shape (n_samples, n_params)
        Output from genSamples.
    param_names : list of str
        Names of the parameters (length must match samples.shape[1]).
    range_val : float, optional
        Fraction of each axis range to display in the corner plot.
    multiple : float, optional
        Scaling factor to apply to N_ parameters before log10 transform.
    save_path : str, optional
        If given, figure is saved to this path.
    return_fig : bool, optional
        If True, returns the matplotlib Figure object.
    verbose : bool, optional
        If True, prints status messages.
    """
    arr = np.array(samples, copy=True)
    if arr.ndim != 2 or arr.shape[1] != len(param_names):
        raise ValueError("samples must have shape (n_samples, len(param_names))")
    
    # Apply log transforms
    for i, pname in enumerate(param_names):
        data = arr[:, i]
        if pname.startswith("N_"):
            arr[:, i] = np.nan_to_num(np.log10(data * multiple))
        elif pname.startswith("T_"):
            arr[:, i] = np.nan_to_num(np.log10(data))

    if verbose:
        mins = np.nanmin(arr, axis=0)
        maxs = np.nanmax(arr, axis=0)
        print("plot_prior_corner: transformed data ranges")
        for nm, lo, hi in zip(param_names, mins, maxs):
            print(f"  {nm}: [{lo:.3f}, {hi:.3f}]")

    fig = corner.corner(
        arr,
        labels=param_names,
        range=[range_val] * len(param_names),
        plot_datapoints=False,            # no individual markers
        fill_contours=False,              # no filled contours
        plot_contours=True,
        levels=[0.68, 0.95, 99],              # 1σ, 2σ, and 3σ contours
        bins=30,                          # moderate bin count
    )

    if save_path:
        if verbose:
            print(f"Saving prior corner plot to {save_path}")
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            fig = None

    if return_fig:
        return fig


def plot_prior_corner_velocities(samples, param_names, vel_prefix='v_', range_val=0.99,
                                    save_path=None, return_fig=False, verbose=False):
    """
    Corner plot of prior samples restricted to velocity parameters (by prefix).
    Draws dotted diagonal y=x on each off-diagonal panel to indicate equality.

    Parameters
    ----------
    samples : array-like, shape (n_samples, n_params)
        Output from genSamples.
    param_names : list of str
        Names of all parameters corresponding to columns of samples.
    vel_prefix : str
        Prefix identifying velocity parameters (default 'v_').
    range_val : float
        Fraction of axis range to show for corner.
    save_path : str or None
        If provided, save the figure.
    return_fig : bool
        If True, return the Figure object.
    verbose : bool
        If True, print debug messages.
    """
    arr = np.asarray(samples)
    if arr.ndim != 2 or arr.shape[1] != len(param_names):
        raise ValueError("samples must have shape (n_samples, len(param_names))")

    # select velocity columns
    vel_idx = [i for i, nm in enumerate(param_names) if nm.startswith(vel_prefix)]
    vel_names = [param_names[i] for i in vel_idx]
    if len(vel_idx) == 0:
        raise ValueError(f"No parameters starting with '{vel_prefix}' found in param_names")

    data = arr[:, vel_idx]
    if verbose:
        print("plot_prior_corner_velocities: selected velocity parameters:", vel_names)
        print("  data.shape =", data.shape)

    # produce corner plot for velocities only
    fig = corner.corner(
        data,
        labels=vel_names,
        range=[range_val] * len(vel_names),
        plot_datapoints=False,
        plot_contours=True,
        fill_contours=False,
        levels=[0.682689, 0.954500, 0.997300],
        bins=20,
    )

    # draw dotted diagonal y=x on off-diagonal panels
    axes = fig.axes
    n = len(vel_names)

    try:
        if len(axes) == n * n:
            ax_mat = np.array(axes).reshape((n, n))
        else:
            ax_mat = np.empty((n, n), dtype=object)
            k = 0
            for i in range(n):
                for j in range(0, i + 1):
                    ax_mat[i, j] = axes[k]
                    k += 1
    except Exception:
        if verbose:
            print("plot_prior_corner_velocities: fallback diagonal drawing across all axes")
        for ax in axes:
            try:
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                lo = max(min(x0, y0), -1e9)
                hi = min(max(x1, y1), 1e9)
                xs = np.linspace(lo, hi, 2)
                ax.plot(xs, xs, ':', color='k', linewidth=0.8)
            except Exception:
                if verbose:
                    print("  could not draw diagonal on an axis")
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if return_fig:
            return fig
        return

    for i in range(n):
        for j in range(0, i):
            ax = ax_mat[i, j]
            if ax is None:
                continue
            try:
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                lo = max(min(x0, y0), -1e9)
                hi = min(max(x1, y1), 1e9)
                xs = np.linspace(lo, hi, 2)
                ax.plot(xs, xs, ':', color='k', linewidth=0.8)
            except Exception:
                if verbose:
                    print(f"Could not draw diagonal on axes ({i},{j})")

    if save_path:
        if verbose:
            print("Saving prior velocity corner plot to", save_path)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            plt.close(fig)
            fig = None

    if return_fig:
        return fig