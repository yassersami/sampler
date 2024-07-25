import numpy as np
from numba import njit
from scipy.integrate._ivp.common import (norm)

import simulator_0d.src.py0D.global_data as gd

# from .common import (validate_max_step, validate_tol, select_initial_step,
#                      norm, EPS, num_jac, validate_first_step,
#                      warn_extraneous)

EPS = np.finfo(float).eps
NUM_JAC_DIFF_REJECT = EPS ** 0.875
NUM_JAC_DIFF_SMALL = EPS ** 0.75
NUM_JAC_DIFF_BIG = EPS ** 0.25
NUM_JAC_MIN_FACTOR = 1e3 * EPS
NUM_JAC_FACTOR_INCREASE = 10
NUM_JAC_FACTOR_DECREASE = 0.1

MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10

@njit
def add_dim(x):
    # y[:, None]
    x_n = np.zeros(x.shape + (1,))
    x_n[:,0] = x
    return x_n

@njit
def norm(x):
    """Compute RMS norm."""
    return np.linalg.norm(x) / x.size ** 0.5

# @njit
def compute_R(order, factor):
    """Compute the matrix for changing the differences array."""
    I = np.arange(1, order + 1)[:, None]
    J = np.arange(1, order + 1)
    M = np.zeros((order + 1, order + 1))
    M[1:, 1:] = (I - 1 - factor * J) / I
    M[0] = 1
    if (np.cumprod(M, axis=0)!=0).any():
        print("test")
    return np.cumprod(M, axis=0)

@njit
def compute_R_nb(order, factor):
    """Compute the matrix for changing the differences array."""
    I = add_dim(np.arange(1, order + 1))
    # I = np.arange(1, order + 1)[:, None]
    J = np.arange(1, order + 1)
    M = np.zeros((order + 1, order + 1))
    M[1:, 1:] = (I - 1 - factor * J) / I
    M[0] = 1
    R = np.zeros(M.shape)
    for j in range(R.shape[1]):
        R[:,j]=(np.cumprod(M[:,j]))
    return R


def change_D(D, order, factor):
    """Change differences array in-place when step size is changed."""
    R = compute_R_nb(order, factor)
    U = compute_R_nb(order, 1)
    RU = R.dot(U)
    D[:order + 1] = np.dot(RU.T, D[:order + 1])


@njit
def change_D_nb(D, order, factor):
    """Change differences array in-place when step size is changed."""
    R = compute_R_nb(order, factor)
    U = compute_R_nb(order, 1)
    RU = R.dot(U)
    D[:order + 1] = np.dot(RU.T, D[:order + 1])


def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol, fargs):
    """Solve the algebraic system resulting from BDF method."""
    d = 0
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        f = fun(t_new, y, *fargs)
        if not np.all(np.isfinite(f)):
            break

        dy = solve_lu(LU, c * f - psi - d)
        dy_norm = norm(dy / scale)

        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old

        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            break

        y += dy
        d += dy

        if (dy_norm == 0 or
                rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break

        dy_norm_old = dy_norm

    return converged, k + 1, y, d


@njit
def solve_bdf_system_nb(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol, fargs):
    """Solve the algebraic system resulting from BDF method."""
    d = 0
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        f = fun(t_new, y, *fargs)
        if not np.all(np.isfinite(f)):
            break

        dy = solve_lu(LU, c * f - psi - d)
        dy_norm = norm(dy / scale)

        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old

        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            break

        y += dy
        d += dy

        if (dy_norm == 0 or
                rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break

        dy_norm_old = dy_norm

    return converged, k + 1, y, d


# @njit
def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol, fargs):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    if y0.size == 0:
        return np.inf

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1, *fargs)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)


@njit
def num_jac_nb(fun, fun_unwrapped, t, y, f, threshold, factor, sparsity=None, args = tuple()):
    """Finite differences Jacobian approximation tailored for ODE solvers.

    This function computes finite difference approximation to the Jacobian
    matrix of `fun` with respect to `y` using forward differences.
    The Jacobian matrix has shape (n, n) and its element (i, j) is equal to
    ``d f_i / d y_j``.

    A special feature of this function is the ability to correct the step
    size from iteration to iteration. The main idea is to keep the finite
    difference significantly separated from its round-off error which
    approximately equals ``EPS * np.abs(f)``. It reduces a possibility of a
    huge error and assures that the estimated derivative are reasonably close
    to the true values (i.e., the finite difference approximation is at least
    qualitatively reflects the structure of the true Jacobian).

    Parameters
    ----------
    fun : callable
        Right-hand side of the system implemented in a vectorized fashion.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Value of the right hand side at (t, y).
    threshold : float
        Threshold for `y` value used for computing the step size as
        ``factor * np.maximum(np.abs(y), threshold)``. Typically, the value of
        absolute tolerance (atol) for a solver should be passed as `threshold`.
    factor : ndarray with shape (n,) or None
        Factor to use for computing the step size. Pass None for the very
        evaluation, then use the value returned from this function.
    sparsity : tuple (structure, groups) or None
        Sparsity structure of the Jacobian, `structure` must be csc_matrix.

    Returns
    -------
    J : ndarray or csc_matrix, shape (n, n)
        Jacobian matrix.
    factor : ndarray, shape (n,)
        Suggested `factor` for the next evaluation.
    """
    y = np.asarray(y)
    n = y.shape[0]
    if n == 0:
        return np.empty((0, 0)), factor

    if factor is None:
        factor = np.full(n, EPS ** 0.5)
    else:
        factor = factor
   #     factor = factor.copy()

    # Direct the step as ODE dictates, hoping that such a step won't lead to
    # a problematic region. For complex ODEs it makes sense to use the real
    # part of f as we use steps along real axis.
    f_sign = np.zeros(f.size)
    for i in range(f.size):
        if f[i]>=0:
            f_sign[i] = 1
        else:
            f_sign[i] = -1
    y_scale = f_sign * np.maximum(threshold, np.abs(y))
    h = (y + factor * y_scale) - y

    # Make sure that the step is not 0 to start with. Not likely it will be
    # executed often.
    for i in np.nonzero(h == 0)[0]:
        while h[i] == 0:
            factor[i] *= 10
            h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]
    # dense_num_jac_new_nb(fun,fun_unwrapped, t, y, f, h, factor, y_scale, n)
    # print(np.max(np.abs(A-B)))
    # exit()

    # chron1 = time.time()
    A = dense_num_jac_new_nb(fun,fun_unwrapped, t, y, f, h, factor, y_scale, n, args)

    # t1 = (time.time()-chron1)
    return A
    # return A+(t1,)




@njit
def f_abs_r(f, max_ind):
    """equiv: np.abs(f[max_ind, r])"""
    max_diff2 = np.zeros(max_ind.shape)
    for i in range(len(max_diff2)):
        max_diff2[i] = np.abs(f[max_ind[i], i])
    return max_diff2



@njit
def dense_num_jac_new_nb(fun, fun_unwrapped, t, y, f, h, factor, y_scale, n, args = tuple()):
    check_go = True
    cpt = 0
    while check_go:
        h_vecs = np.diag(h)
        f_new = fun(fun_unwrapped, t, add_dim(y) + h_vecs, args)
        check_go = np.isnan(f_new).any()
        cpt = cpt + 1
        h = h/(2**cpt)
        if (h==0).any():
            raise ValueError("jacobian not found")
        # if cpt >=2:
        #     print("tests")
    diff = f_new - add_dim(f)
    max_ind = np.argmax(np.abs(diff), axis=0)
    r = np.arange(n)
    # max_diff2 = np.zeros(y.shape)
    # for i in range(len(max_diff2)):
    #     max_diff2[i] = np.abs(diff[max_ind[i], i])
    max_diff = f_abs_r(diff,max_ind)
    # max_diff = np.abs(diff[max_ind, r])
    scale = np.maximum(np.abs(f[max_ind]), f_abs_r(f_new, max_ind))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if np.any(diff_too_small):
        # print("\n\ntest\n\n")
        ind, = np.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]

        # h_vecs[ind, ind] = h_new
        for j in range(len(h_new)):
            for i in range(len(ind)):
                if i == ind[j]:
                    h_vecs[i,i] = h_new[j]
        f_new = fun(fun_unwrapped, t, add_dim(y) + h_vecs[:, ind], args)
        diff_new = f_new - add_dim(f)
        max_ind = np.argmax(np.abs(diff_new), axis=0)
        r = np.arange(ind.shape[0])

    #     max_diff_new = np.abs(diff_new[max_ind, r])
        max_diff_new = f_abs_r(diff_new,max_ind)

        scale_new = np.maximum(np.abs(f[max_ind]), f_abs_r(f_new, max_ind))
        # print()
        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            # print("\n\ntest\n\n")
            update, = np.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    diff /= h

    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor



def _step_impl(int_BDF):
    t = int_BDF.t
    D = int_BDF.D

    max_step = int_BDF.max_step
    min_step = 10 * np.abs(np.nextafter(t, int_BDF.direction * np.inf) - t)
    if int_BDF.h_abs > max_step:
        h_abs = max_step
        change_D_nb(D, int_BDF.order, max_step / int_BDF.h_abs)
        int_BDF.n_equal_steps = 0
    elif int_BDF.h_abs < min_step:
        h_abs = min_step
        change_D_nb(D, int_BDF.order, min_step / int_BDF.h_abs)
        int_BDF.n_equal_steps = 0
    else:
        h_abs = int_BDF.h_abs

    atol = int_BDF.atol
    rtol = int_BDF.rtol
    order = int_BDF.order

    alpha = int_BDF.alpha
    gamma = int_BDF.gamma
    error_const = int_BDF.error_const

    J = int_BDF.J
    LU = int_BDF.LU
    current_jac = int_BDF.jac is None

    step_accepted = False
    while not step_accepted:
        if h_abs < min_step:
            return False, int_BDF.TOO_SMALL_STEP

        h = h_abs * int_BDF.direction
        t_new = t + h

        if int_BDF.direction * (t_new - int_BDF.t_bound) > 0:
            t_new = int_BDF.t_bound
            change_D_nb(D, order, np.abs(t_new - t) / h_abs)
            int_BDF.n_equal_steps = 0
            LU = None

        h = t_new - t
        h_abs = np.abs(h)

        y_predict = np.sum(D[:order + 1], axis=0)
        scale = atol + rtol * np.abs(y_predict)
        psi = np.dot(D[1: order + 1].T, gamma[1: order + 1]) / alpha[order]
        # print("y_predict", y_predict)
        # print(order)
        converged = False
        if not (y_predict[(gd.i_Y + 2) :]<0).any():
            c = h / alpha[order]
            while not converged:
                if LU is None:
                    LU = int_BDF.lu(int_BDF.I - c * J)

                converged, n_iter, y_new, d = solve_bdf_system(
                    int_BDF.fun_single_nb, t_new, y_predict, c, psi, LU, int_BDF.solve_lu,
                    scale, int_BDF.newton_tol, int_BDF.fargs)

                if not converged:
                    if current_jac:
                        break
                    J = int_BDF.jac(t_new, y_predict)
                    LU = None
                    current_jac = True

        if not converged:
            factor = 0.5
            h_abs *= factor
            change_D_nb(D, order, factor)
            int_BDF.n_equal_steps = 0
            LU = None
            continue

        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                    + n_iter)

        scale = atol + rtol * np.abs(y_new)
        error = error_const[order] * d
        error_norm = norm(error / scale)

        if error_norm > 1:
            factor = max(MIN_FACTOR,
                            safety * error_norm ** (-1 / (order + 1)))
            h_abs *= factor
            change_D_nb(D, order, factor)
            int_BDF.n_equal_steps = 0
            # As we didn't have problems with convergence, we don't
            # reset LU here.
        else:
            step_accepted = True

    int_BDF.n_equal_steps += 1

    int_BDF.t = t_new
    int_BDF.y = y_new

    int_BDF.h_abs = h_abs
    int_BDF.J = J
    int_BDF.LU = LU

    # Update differences. The principal relation here is
    # D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D
    # contained difference for previous interpolating polynomial and
    # d = D^{k + 1} y_n. Thus this elegant code follows.
    D[order + 2] = d - D[order + 1]
    D[order + 1] = d
    for i in reversed(range(order + 1)):
        D[i] += D[i + 1]

    if int_BDF.n_equal_steps < order + 1:
        return True, None

    if order > 1:
        error_m = error_const[order - 1] * D[order]
        error_m_norm = norm(error_m / scale)
    else:
        error_m_norm = np.inf

    if order < MAX_ORDER:
        error_p = error_const[order + 1] * D[order + 2]
        error_p_norm = norm(error_p / scale)
    else:
        error_p_norm = np.inf

    error_norms = np.array([error_m_norm, error_norm, error_p_norm])
    with np.errstate(divide='ignore'):
        factors = error_norms ** (-1 / np.arange(order, order + 3))

    delta_order = np.argmax(factors) - 1
    order += delta_order
    int_BDF.order = order

    factor = min(MAX_FACTOR, safety * np.max(factors))
    int_BDF.h_abs *= factor
    change_D_nb(D, order, factor)
    int_BDF.n_equal_steps = 0
    int_BDF.LU = None

    return True, None


def step(int_BDF):
    """
    from _ivp/base
    """
    """Perform one integration step.

    Returns
    -------
    message : string or None
        Report from the solver. Typically a reason for a failure if
        `self.status` is 'failed' after the step was taken or None
        otherwise.
    """
    if int_BDF.status != 'running':
        raise RuntimeError("Attempt to step on a failed or finished "
                            "solver.")

    if int_BDF.n == 0 or int_BDF.t == int_BDF.t_bound:
        # Handle corner cases of empty solver or no integration.
        int_BDF.t_old = int_BDF.t
        int_BDF.t = int_BDF.t_bound
        message = None
        int_BDF.status = 'finished'
    else:
        t = int_BDF.t
        success, message = int_BDF._step_impl(int_BDF)

        if not success:
            int_BDF.status = 'failed'
        else:
            int_BDF.t_old = t
            if int_BDF.direction * (int_BDF.t - int_BDF.t_bound) >= 0:
                int_BDF.status = 'finished'

    return message



##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################




def num_jac(fun, t, y, f, threshold, factor, sparsity=None):
    """Finite differences Jacobian approximation tailored for ODE solvers.

    This function computes finite difference approximation to the Jacobian
    matrix of `fun` with respect to `y` using forward differences.
    The Jacobian matrix has shape (n, n) and its element (i, j) is equal to
    ``d f_i / d y_j``.

    A special feature of this function is the ability to correct the step
    size from iteration to iteration. The main idea is to keep the finite
    difference significantly separated from its round-off error which
    approximately equals ``EPS * np.abs(f)``. It reduces a possibility of a
    huge error and assures that the estimated derivative are reasonably close
    to the true values (i.e., the finite difference approximation is at least
    qualitatively reflects the structure of the true Jacobian).

    Parameters
    ----------
    fun : callable
        Right-hand side of the system implemented in a vectorized fashion.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Value of the right hand side at (t, y).
    threshold : float
        Threshold for `y` value used for computing the step size as
        ``factor * np.maximum(np.abs(y), threshold)``. Typically, the value of
        absolute tolerance (atol) for a solver should be passed as `threshold`.
    factor : ndarray with shape (n,) or None
        Factor to use for computing the step size. Pass None for the very
        evaluation, then use the value returned from this function.
    sparsity : tuple (structure, groups) or None
        Sparsity structure of the Jacobian, `structure` must be csc_matrix.

    Returns
    -------
    J : ndarray or csc_matrix, shape (n, n)
        Jacobian matrix.
    factor : ndarray, shape (n,)
        Suggested `factor` for the next evaluation.
    """
    y = np.asarray(y)
    n = y.shape[0]
    if n == 0:
        return np.empty((0, 0)), factor

    if factor is None:
        factor = np.full(n, EPS ** 0.5)
    else:
        factor = factor.copy()

    # Direct the step as ODE dictates, hoping that such a step won't lead to
    # a problematic region. For complex ODEs it makes sense to use the real
    # part of f as we use steps along real axis.
    f_sign = 2 * (np.real(f) >= 0).astype(float) - 1
    y_scale = f_sign * np.maximum(threshold, np.abs(y))
    h = (y + factor * y_scale) - y

    # Make sure that the step is not 0 to start with. Not likely it will be
    # executed often.
    for i in np.nonzero(h == 0)[0]:
        while h[i] == 0:
            factor[i] *= 10
            h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]

    if sparsity is None:
        return _dense_num_jac(fun, t, y, f, h, factor, y_scale)
    else:
        structure, groups = sparsity
        return _sparse_num_jac(fun, t, y, f, h, factor, y_scale,
                               structure, groups)


def _dense_num_jac(fun, t, y, f, h, factor, y_scale):
    n = y.shape[0]
    h_vecs = np.diag(h)
    f_new = fun(t, y[:, None] + h_vecs)
    diff = f_new - f[:, None]
    max_ind = np.argmax(np.abs(diff), axis=0)
    r = np.arange(n)
    max_diff = np.abs(diff[max_ind, r])
    scale = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if np.any(diff_too_small):
        ind, = np.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        h_vecs[ind, ind] = h_new
        f_new = fun(t, y[:, None] + h_vecs[:, ind])
        diff_new = f_new - f[:, None]
        max_ind = np.argmax(np.abs(diff_new), axis=0)
        r = np.arange(ind.shape[0])
        max_diff_new = np.abs(diff_new[max_ind, r])
        scale_new = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            update, = np.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    diff /= h

    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor


def _sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups):
    n = y.shape[0]
    n_groups = np.max(groups) + 1
    h_vecs = np.empty((n_groups, n))
    for group in range(n_groups):
        e = np.equal(group, groups)
        h_vecs[group] = h * e
    h_vecs = h_vecs.T

    f_new = fun(t, y[:, None] + h_vecs)
    df = f_new - f[:, None]

    i, j, _ = find(structure)
    diff = coo_matrix((df[i, groups[j]], (i, j)), shape=(n, n)).tocsc()
    max_ind = np.array(abs(diff).argmax(axis=0)).ravel()
    r = np.arange(n)
    max_diff = np.asarray(np.abs(diff[max_ind, r])).ravel()
    scale = np.maximum(np.abs(f[max_ind]),
                       np.abs(f_new[max_ind, groups[r]]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if np.any(diff_too_small):
        ind, = np.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        h_new_all = np.zeros(n)
        h_new_all[ind] = h_new

        groups_unique = np.unique(groups[ind])
        groups_map = np.empty(n_groups, dtype=int)
        h_vecs = np.empty((groups_unique.shape[0], n))
        for k, group in enumerate(groups_unique):
            e = np.equal(group, groups)
            h_vecs[k] = h_new_all * e
            groups_map[group] = k
        h_vecs = h_vecs.T

        f_new = fun(t, y[:, None] + h_vecs)
        df = f_new - f[:, None]
        i, j, _ = find(structure[:, ind])
        diff_new = coo_matrix((df[i, groups_map[groups[ind[j]]]],
                               (i, j)), shape=(n, ind.shape[0])).tocsc()

        max_ind_new = np.array(abs(diff_new).argmax(axis=0)).ravel()
        r = np.arange(ind.shape[0])
        max_diff_new = np.asarray(np.abs(diff_new[max_ind_new, r])).ravel()
        scale_new = np.maximum(
            np.abs(f[max_ind_new]),
            np.abs(f_new[max_ind_new, groups_map[groups[ind]]]))

        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            update, = np.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    diff.data /= np.repeat(h, np.diff(diff.indptr))

    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor
