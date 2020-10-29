"""
The MIT License (MIT)

Copyright (c) 2018 Alex H. Williams and N. Benjamin Erichson

Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
  so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams

Reorganized by Guangyu Robert Yang
"""

import numpy as np
from scipy import linalg
import numba
import timeit
from .tda_tensors import randn_ktensor, rand_ktensor, KTensor, unfold, khatri_rao


def _check_cpd_inputs(X, rank):
    """Checks that inputs to optimization function are appropriate.

    Parameters
    ----------
    X : ndarray
        Tensor used for fitting CP decomposition.
    rank : int
        Rank of low rank decomposition.

    Raises
    ------
    ValueError: If inputs are not suited for CP decomposition.
    """
    if X.ndim < 3:
        raise ValueError("Array with X.ndim > 2 expected.")
    if rank <= 0 or not isinstance(rank, int):
        raise ValueError("Rank is invalid.")


def _get_initial_ktensor(init, X, rank, random_state, scale_norm=True):
    """
    Parameters
    ----------
    init : str
        Specifies type of initializations ('randn', 'rand')
    X : ndarray
        Tensor that the decomposition is fit to.
    rank : int
        Rank of decomposition
    random_state : RandomState or int
        Specifies seed for random number generator
    scale_norm : bool
        If True, norm is scaled to match X (default: True)

    Returns
    -------
    U : KTensor
        Initial factor matrices used optimization.
    normX : float
        Frobenious norm of tensor data.
    """
    normX = linalg.norm(X) if scale_norm else None

    if init == 'randn':
        # TODO - match the norm of the initialization to the norm of X.
        U = randn_ktensor(X.shape, rank, norm=normX, random_state=random_state)

    elif init == 'rand':
        # TODO - match the norm of the initialization to the norm of X.
        U = rand_ktensor(X.shape, rank, norm=normX, random_state=random_state)

    elif isinstance(init, KTensor):
        U = init.copy()

    else:
        raise ValueError("Expected 'init' to either be a KTensor or a string "
                         "specifying how to initialize optimization. Valid "
                         "strings are ('randn', 'rand').")

    return U, normX


class FitResult(object):
    """
    Holds result of optimization.

    Attributes
    ----------
    total_time: float
        Number of seconds spent before stopping optimization.
    obj : float
        Objective value of optimization (at current parameters).
    obj_hist : list of floats
        Objective values at each iteration.
    """

    def __init__(self, factors, method, tol=1e-5, verbose=True, max_iter=500,
                 min_iter=1, max_time=np.inf):
        """Initializes FitResult.

        Parameters
        ----------
        factors : KTensor
            Initial guess for tensor decomposition.
        method : str
            Name of optimization method (used for printing).
        tol : float
            Stopping criterion.
        verbose : bool
            Whether to print progress of optimization.
        max_iter : int
            Maximum number of iterations before quitting early.
        min_iter : int
            Minimum number of iterations before stopping due to convergence.
        max_time : float
            Maximum number of seconds before quitting early.
        """
        self.factors = factors
        self.obj = np.inf
        self.obj_hist = []
        self.method = method

        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.max_time = max_time

        self.iterations = 0
        self.converged = False
        self.t0 = timeit.default_timer()
        self.total_time = None

    @property
    def still_optimizing(self):
        """True unless converged or maximum iterations/time exceeded."""

        # Check if we need to give up on optimizing.
        if (self.iterations > self.max_iter) or (self.time_elapsed() > self.max_time):
            return False

        # Always optimize for at least 'min_iter' iterations.
        elif not hasattr(self, 'improvement') or (self.iterations < self.min_iter):
            return True

        # Check convergence.
        else:
            self.converged = self.improvement < self.tol
            return False if self.converged else True

    def time_elapsed(self):
        return timeit.default_timer() - self.t0

    def update(self, obj):

        # Keep track of iterations.
        self.iterations += 1

        # Compute improvement in objective.
        self.improvement = self.obj - obj
        self.obj = obj
        self.obj_hist.append(obj)

        # If desired, print progress.
        if self.verbose:
            p_args = self.method, self.iterations, self.obj, self.improvement
            s = '{}: iteration {}, objective {}, improvement {}.'
            print(s.format(*p_args))

    def finalize(self):

        # Set final time, final print statement
        self.total_time = self.time_elapsed()

        if self.verbose:
            s = 'Converged after {} iterations, {} seconds. Objective: {}.'
            print(s.format(self.iterations, self.total_time, self.obj))

        return self


def cp_als(X, rank, random_state=None, init='randn', skip_modes=[], **options):
    """Fits CP Decomposition using Alternating Least Squares (ALS).

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A tensor with ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    random_state : integer, ``RandomState``, or ``None``, optional (default ``None``)
        If integer, sets the seed of the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, use the RandomState instance used by ``numpy.random``.

    init : str, or KTensor, optional (default ``'randn'``).
        Specifies initial guess for KTensor factor matrices.
        If ``'randn'``, Gaussian random numbers are used to initialize.
        If ``'rand'``, uniform random numbers are used to initialize.
        If KTensor instance, a copy is made to initialize the optimization.

    skip_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor that are not fit. This can be
        used to fix certain factor matrices that have been previously
        fit.

    options : dict, specifying fitting options.

        tol : float, optional (default ``tol=1E-5``)
            Stopping tolerance for reconstruction error.

        max_iter : integer, optional (default ``max_iter = 500``)
            Maximum number of iterations to perform before exiting.

        min_iter : integer, optional (default ``min_iter = 1``)
            Minimum number of iterations to perform before exiting.

        max_time : integer, optional (default ``max_time = np.inf``)
            Maximum computational time before exiting.

        verbose : bool ``{'True', 'False'}``, optional (default ``verbose=True``)
            Display progress.


    Returns
    -------
    result : FitResult instance
        Object which holds the fitted results. It provides the factor matrices
        in form of a KTensor, ``result.factors``.


    Notes
    -----
    Alternating Least Squares (ALS) is a very old and reliable method for
    fitting CP decompositions. This is likely a good first algorithm to try.


    References
    ----------
    Kolda, T. G. & Bader, B. W.
    "Tensor Decompositions and Applications."
    SIAM Rev. 51 (2009): 455-500
    http://epubs.siam.org/doi/pdf/10.1137/07070111X

    Comon, Pierre & Xavier Luciani & Andre De Almeida.
    "Tensor decompositions, alternating least squares and other tales."
    Journal of chemometrics 23 (2009): 393-405.
    http://onlinelibrary.wiley.com/doi/10.1002/cem.1236/abstract


    Examples
    --------

    ```
    import tensortools as tt
    I, J, K, R = 20, 20, 20, 4
    X = tt.randn_tensor(I, J, K, rank=R)
    tt.cp_als(X, rank=R)
    ```
    """

    # Check inputs.
    _check_cpd_inputs(X, rank)

    # Initialize problem.
    U, normX = _get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, 'CP_ALS', **options)

    # Main optimization loop.
    while result.still_optimizing:

        # Iterate over each tensor mode.
        for n in range(X.ndim):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # i) Normalize factors to prevent singularities.
            U.rebalance()

            # ii) Compute the N-1 gram matrices.
            components = [U[j] for j in range(X.ndim) if j != n]
            grams = np.prod([u.T @ u for u in components], axis=0)

            # iii)  Compute Khatri-Rao product.
            kr = khatri_rao(components)

            # iv) Form normal equations and solve via Cholesky
            c = scipy.linalg.cho_factor(grams, overwrite_a=False)
            p = unfold(X, n).dot(kr)
            U[n] = scipy.linalg.cho_solve(c, p.T, overwrite_b=False).T
            # U[n] = np.linalg.solve(grams, unfold(X, n).dot(kr).T).T

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute objective function
        # grams *= U[-1].T.dot(U[-1])
        # obj = np.sqrt(np.sum(grams) - 2*np.sum(p*U[-1]) + normX**2) / normX
        obj = np.linalg.norm(U.full() - X) / normX

        # Update result
        result.update(obj)

    # Finalize and return the optimization result.
    return result.finalize()


def mcp_als(X, rank, mask, random_state=None, init='randn', skip_modes=[], **options):
    """Fits CP Decomposition with missing data using Alternating Least Squares (ALS).

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A tensor with ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    mask : (I_1, ..., I_N) array_like
        A binary tensor with the same shape as ``X``. All entries equal to zero
        correspond to held out or missing data in ``X``. All entries equal to
        one correspond to observed entries in ``X`` and the decomposition is
        fit to these datapoints.

    random_state : integer, ``RandomState``, or ``None``, optional (default ``None``)
        If integer, sets the seed of the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, use the RandomState instance used by ``numpy.random``.

    init : str, or KTensor, optional (default ``'randn'``).
        Specifies initial guess for KTensor factor matrices.
        If ``'randn'``, Gaussian random numbers are used to initialize.
        If ``'rand'``, uniform random numbers are used to initialize.
        If KTensor instance, a copy is made to initialize the optimization.

    skip_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor that are not fit. This can be
        used to fix certain factor matrices that have been previously
        fit.

    options : dict, specifying fitting options.

        tol : float, optional (default ``tol=1E-5``)
            Stopping tolerance for reconstruction error.

        max_iter : integer, optional (default ``max_iter = 500``)
            Maximum number of iterations to perform before exiting.

        min_iter : integer, optional (default ``min_iter = 1``)
            Minimum number of iterations to perform before exiting.

        max_time : integer, optional (default ``max_time = np.inf``)
            Maximum computational time before exiting.

        verbose : bool ``{'True', 'False'}``, optional (default ``verbose=True``)
            Display progress.


    Returns
    -------
    result : FitResult instance
        Object which holds the fitted results. It provides the factor matrices
        in form of a KTensor, ``result.factors``.


    Notes
    -----
    Fitting CP decompositions with missing data can be exploited to perform
    cross-validation.

    References
    ----------
    Williams, A. H.
    "Solving Least-Squares Regression with Missing Data."
    http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    """

    # Check inputs.
    _check_cpd_inputs(X, rank)

    # Initialize problem.
    U, _ = _get_initial_ktensor(init, X, rank, random_state, scale_norm=False)
    result = FitResult(U, 'MCP_ALS', **options)
    normX = np.linalg.norm((X * mask))

    # Main optimization loop.
    while result.still_optimizing:

        # Iterate over each tensor mode.
        for n in range(X.ndim):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # i) Normalize factors to prevent singularities.
            U.rebalance()

            # ii) Unfold data and mask along the nth mode.
            unf = unfold(X, n)  # i_n x N
            m = unfold(mask, n)  # i_n x N

            # iii) Form Khatri-Rao product of factors matrices.
            components = [U[j] for j in range(X.ndim) if j != n]
            krt = khatri_rao(components).T  # N x r

            # iv) Broadcasted solve of linear systems.
            # Left hand side of equations, R x R x X.shape[n]
            # Right hand side of equations, X.shape[n] x R x 1
            lhs_stack = np.matmul(m[:, None, :] * krt[None, :, :], krt.T[None, :, :])
            rhs_stack = np.dot(unf * m, krt.T)[:, :, None]

            # vi) Update factor.
            U[n] = np.linalg.solve(lhs_stack, rhs_stack).reshape(X.shape[n], rank)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        obj = linalg.norm(mask * (U.full() - X)) / normX

        # Update result
        result.update(obj)

    # Finalize and return the optimization result.
    return result.finalize()


def ncp_bcd(
        X, rank, random_state=None, init='rand',
        skip_modes=[], negative_modes=[], **options):
    """
    Fits nonnegative CP Decomposition using the Block Coordinate Descent (BCD)
    Method.

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A real array with nonnegative entries and ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    init : str, or KTensor, optional (default ``'rand'``).
        Specifies initial guess for KTensor factor matrices.
        If ``'randn'``, Gaussian random numbers are used to initialize.
        If ``'rand'``, uniform random numbers are used to initialize.
        If KTensor instance, a copy is made to initialize the optimization.

    skip_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor that are not fit. This can be
        used to fix certain factor matrices that have been previously
        fit.

    negative_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor whose factors are not constrained
        to be nonnegative.

    options : dict, specifying fitting options.

        tol : float, optional (default ``tol=1E-5``)
            Stopping tolerance for reconstruction error.

        max_iter : integer, optional (default ``max_iter = 500``)
            Maximum number of iterations to perform before exiting.

        min_iter : integer, optional (default ``min_iter = 1``)
            Minimum number of iterations to perform before exiting.

        max_time : integer, optional (default ``max_time = np.inf``)
            Maximum computational time before exiting.

        verbose : bool ``{'True', 'False'}``, optional (default ``verbose=True``)
            Display progress.


    Returns
    -------
    result : FitResult instance
        Object which holds the fitted results. It provides the factor matrices
        in form of a KTensor, ``result.factors``.


    Notes
    -----
    This implemenation is using the Block Coordinate Descent Method.


    References
    ----------
    Xu, Yangyang, and Wotao Yin. "A block coordinate descent method for
    regularized multiconvex optimization with applications to
    negative tensor factorization and completion."
    SIAM Journal on imaging sciences 6.3 (2013): 1758-1789.


    Examples
    --------

    """

    # Check inputs.
    _check_cpd_inputs(X, rank)

    # Store norm of X for computing objective function.
    N = X.ndim

    # Initialize problem.
    U, normX = _get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, 'NCP_BCD', **options)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Block coordinate descent
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Um = U.copy()  # Extrapolations of compoenents
    extraw = 1  # Used for extrapolation weight update
    weights_U = np.ones(N)  # Extrapolation weights
    L = np.ones(N)  # Lipschitz constants
    obj_bcd = 0.5 * normX**2  # Initial objective value

    # Main optimization loop.
    while result.still_optimizing:
        obj_bcd_old = obj_bcd  # Old objective value
        U_old = U.copy()
        extraw_old = extraw

        for n in range(N):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # Select all components, but U_n
            components = [U[j] for j in range(N) if j != n]

            # i) compute the N-1 gram matrices
            grams = np.prod([arr.T.dot(arr) for arr in components], axis=0)

            # Update gradient Lipschnitz constant
            L0 = L  # Lipschitz constants
            L[n] = np.linalg.norm(grams, 2)

            # ii)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            p = unfold(X, n).dot(kr)

            # Compute Gradient.
            grad = Um[n].dot(grams) - p

            # Enforce nonnegativity (project onto nonnegative orthant).
            U[n] = Um[n] - grad / L[n]
            if n not in negative_modes:
                U[n] = np.maximum(0.0, U[n])

        # Compute objective function and update optimization result.
        # grams *= U[X.ndim - 1].T.dot(U[X.ndim - 1])
        # obj = np.sqrt(np.sum(grams) - 2 * np.sum(U[X.ndim - 1] * p) + normX**2) / normX
        obj = np.linalg.norm(X - U.full()) / normX
        result.update(obj)

        # Correction and extrapolation.
        n = np.setdiff1d(np.arange(X.ndim), skip_modes).max()
        grams *= U[n].T.dot(U[n])
        obj_bcd = 0.5 * (np.sum(grams) - 2 * np.sum(U[n] * p) + normX**2)

        extraw = (1 + np.sqrt(1 + 4 * extraw_old**2)) / 2.0

        if obj_bcd >= obj_bcd_old:
            # restore previous A to make the objective nonincreasing
            Um = U_old

        else:
            # apply extrapolation
            w = (extraw_old - 1.0) / extraw  # Extrapolation weight
            for n in range(N):
                if n not in skip_modes:
                    weights_U[n] = min(w, 1.0 * np.sqrt(L0[n] / L[n]))  # choose smaller weights for convergence
                    Um[n] = U[n] + weights_U[n] * (U[n] - U_old[n])  # extrapolation

    # Finalize and return the optimization result.
    return result.finalize()


def ncp_hals(
        X, rank, mask=None, random_state=None, init='rand',
        skip_modes=[], negative_modes=[], **options):
    """
    Fits nonnegtaive CP Decomposition using the Hierarcial Alternating Least
    Squares (HALS) Method.

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A real array with nonnegative entries and ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    mask : (I_1, ..., I_N) array_like
        Binary tensor, same shape as X, specifying censored or missing data values
        at locations where (mask == 0) and observed data where (mask == 1).

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    init : str, or KTensor, optional (default ``'rand'``).
        Specifies initial guess for KTensor factor matrices.
        If ``'randn'``, Gaussian random numbers are used to initialize.
        If ``'rand'``, uniform random numbers are used to initialize.
        If KTensor instance, a copy is made to initialize the optimization.

    skip_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor that are not fit. This can be
        used to fix certain factor matrices that have been previously
        fit.

    negative_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor whose factors are not constrained
        to be nonnegative.

    options : dict, specifying fitting options.

        tol : float, optional (default ``tol=1E-5``)
            Stopping tolerance for reconstruction error.

        max_iter : integer, optional (default ``max_iter = 500``)
            Maximum number of iterations to perform before exiting.

        min_iter : integer, optional (default ``min_iter = 1``)
            Minimum number of iterations to perform before exiting.

        max_time : integer, optional (default ``max_time = np.inf``)
            Maximum computational time before exiting.

        verbose : bool ``{'True', 'False'}``, optional (default ``verbose=True``)
            Display progress.


    Returns
    -------
    result : FitResult instance
        Object which holds the fitted results. It provides the factor matrices
        in form of a KTensor, ``result.factors``.


    Notes
    -----
    This implemenation is using the Hierarcial Alternating Least Squares Method.


    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Examples
    --------


    """

    # Mask missing elements.
    if mask is not None:
        X = np.copy(X)
        X[~mask] = np.mean(X[mask])

    # Check inputs.
    _check_cpd_inputs(X, rank)

    # Initialize problem.
    U, normX = _get_initial_ktensor(init, X, rank, random_state)
    result = FitResult(U, 'NCP_HALS', **options)

    # Store problem dimensions.
    normX = np.linalg.norm(X)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    # i)   compute the N gram matrices and multiply
    # ii)  Compute Khatri-Rao product
    # iii) Update component U_1, U_2, ... U_N
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    while result.still_optimizing:

        for n in range(X.ndim):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # Select all components, but U_n
            components = [U[j] for j in range(X.ndim) if j != n]

            # i) compute the N-1 gram matrices
            grams = np.prod([arr.T @ arr for arr in components], axis=0)

            # ii)  Compute Khatri-Rao product
            kr = khatri_rao(components)
            Xmkr = unfold(X, n).dot(kr)

            # iii) Update component U_n
            _hals_update(U[n], grams, Xmkr, n not in negative_modes)

            # iv) Update masked elements.
            if mask is not None:
                pred = U.full()
                X[~mask] = pred[~mask]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if mask is None:

            # Determine mode that was fit last.
            n = np.setdiff1d(np.arange(X.ndim), skip_modes).max()

            # Add contribution of last fit factors to gram matrix.
            grams *= U[n].T @ U[n]
            residsq = np.sum(grams) - 2 * np.sum(U[n] * Xmkr) + (normX ** 2)
            result.update(np.sqrt(residsq) / normX)

        else:
            result.update(np.linalg.norm(X - pred) / normX)

    # end optimization loop, return result.
    return result.finalize()


@numba.jit(nopython=True)
def _hals_update(factors, grams, Xmkr, nonneg):

    dim = factors.shape[0]
    rank = factors.shape[1]
    indices = np.arange(rank)

    # Handle special case of rank-1 model.
    if rank == 1:
        if nonneg:
            factors[:] = np.maximum(0.0, Xmkr / grams[0, 0])
        else:
            factors[:] = Xmkr / grams[0, 0]

    # Do a few inner iterations.
    else:
        for itr in range(3):
            for p in range(rank):
                idx = (indices != p)
                Cp = factors[:, idx] @ grams[idx][:, p]
                r = (Xmkr[:, p] - Cp) / np.maximum(grams[p, p], 1e-6)

                if nonneg:
                    factors[:, p] = np.maximum(r, 0.0)
                else:
                    factors[:, p] = r