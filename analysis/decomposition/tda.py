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
"""


from tqdm import trange
import collections
import numpy as np

from .munkres import Munkres
import analysis.decomposition.tda_optim as optimize


def kruskal_align(U, V, permute_U=False, permute_V=False):
    """Aligns two KTensors and returns a similarity score.

    Parameters
    ----------
    U : KTensor
        First kruskal tensor to align.
    V : KTensor
        Second kruskal tensor to align.
    permute_U : bool
        If True, modifies 'U' to align the KTensors (default is False).
    permute_V : bool
        If True, modifies 'V' to align the KTensors (default is False).

    Notes
    -----
    If both `permute_U` and `permute_V` are both set to True, then the
    factors are ordered from most to least similar. If only one is
    True then the factors on the modified KTensor are re-ordered to
    match the factors in the un-aligned KTensor.

    Returns
    -------
    similarity : float
        Similarity score between zero and one.
    """

    # Initial model ranks.
    U_init_rank, V_init_rank = U.rank, V.rank

    # Drop any factors with zero magnitude.
    U.prune_()
    V.prune_()

    # Munkres expects V_rank <= U_rank.
    if U.rank > V.rank:
        U.pad_zeros_(U_init_rank - U.rank)
        V.pad_zeros_(V_init_rank - V.rank)
        return kruskal_align(
            V, U, permute_U=permute_V, permute_V=permute_U)

    # Compute similarity matrices.
    unrm = [f / np.linalg.norm(f, axis=0) for f in U.factors]
    vnrm = [f / np.linalg.norm(f, axis=0) for f in V.factors]
    sim_matrices = [np.dot(u.T, v) for u, v in zip(unrm, vnrm)]
    cost = 1 - np.mean(np.abs(sim_matrices), axis=0)

    # Solve matching problem via Hungarian algorithm.
    indices = Munkres().compute(cost.copy())
    prmU, prmV = zip(*indices)

    # Compute mean factor similarity given the optimal matching.
    similarity = np.mean(1 - cost[prmU, prmV])

    # If U and V are of different ranks, identify unmatched factors.
    unmatched_U = list(set(range(U.rank)) - set(prmU))
    unmatched_V = list(set(range(V.rank)) - set(prmV))

    # If permuting both U and V, order factors from most to least similar.
    if permute_U and permute_V:
        idx = np.argsort(cost[prmU, prmV])

    # If permute_U is False, then order the factors such that the ordering
    # for U is unchanged.
    elif permute_V:
        idx = np.argsort(prmU)

    # If permute_V is False, then order the factors such that the ordering
    # for V is unchanged.
    elif permute_U:
        idx = np.argsort(prmV)

    # If permute_U and permute_V are both False, then we are done and can
    # simply return the similarity.
    else:
        return similarity

    # Re-order the factor permutations.
    prmU = [prmU[i] for i in idx]
    prmV = [prmV[i] for i in idx]

    # Permute the factors.
    if permute_U:
        U.permute(prmU + unmatched_U)
    if permute_V:
        V.permute(prmV + unmatched_V)

    # Flip the signs of factors.
    flips = np.sign([F[prmU, prmV] for F in sim_matrices])
    flips[0] *= np.prod(flips, axis=0)  # always flip an even number of factors

    if permute_U:
        for i, f in enumerate(flips):
            U.factors[i][:, :f.size] *= f

    elif permute_V:
        for i, f in enumerate(flips):
            V.factors[i][:, :f.size] *= f

    # Pad zero factors to restore original ranks.
    U.pad_zeros_(U_init_rank - U.rank)
    V.pad_zeros_(V_init_rank - V.rank)

    # Return the similarity score
    return similarity


class TDA(object):
    """
    Represents an ensemble of fitted tensor decompositions.
    """

    def __init__(self, nonneg=False, fit_method=None, fit_options=dict()):
        """Initializes Ensemble.

        Parameters
        ----------
        nonneg : bool
            If True, constrains low-rank factor matrices to be nonnegative.
        fit_method : None, str, callable, optional (default: None)
            Method for fitting a tensor decomposition. If input is callable,
            it is used directly. If input is a string then method is taken
            from tensortools.optimize using ``getattr``. If None, a reasonable
            default method is chosen.
        fit_options : dict
            Holds optional arguments for fitting method.
        """

        # Model parameters
        self._nonneg = nonneg

        # Determinine optimization method. If user input is None, try to use a
        # reasonable default. Otherwise check that it is callable.
        if fit_method is None:
            self._fit_method = optimize.ncp_bcd if nonneg else optimize.cp_als
        elif isinstance(fit_method, str):
            try:
                self._fit_method = getattr(optimize, fit_method)
            except AttributeError:
                raise ValueError("Did not recognize method 'fit_method' "
                                 "{}".format(fit_method))
        elif callable(fit_method):
            self._fit_method = fit_method
        else:
            raise ValueError("Expected 'fit_method' to be a string or "
                             "callable.")

        # Try to pick reasonable defaults for optimization options.
        fit_options.setdefault('tol', 1e-5)
        fit_options.setdefault('max_iter', 500)
        fit_options.setdefault('verbose', False)
        self._fit_options = fit_options

        # TODO - better way to hold all results...
        self.results = dict()

    def fit(self, X, ranks, replicates=1, verbose=True):
        """
        Fits CP tensor decompositions for different choices of rank.

        Parameters
        ----------
        X : (I_1, ..., I_N) array_like
            A real array with nonnegative entries and ``X.ndim >= 3``.
        ranks : int, or iterable
            iterable specifying number of components in each model
        replicates: int
            number of models to fit at each rank
        verbose : bool
            If True, prints summaries and optimization progress.
        """

        # Make ranks iterable if necessary.
        if not isinstance(ranks, collections.Iterable):
            ranks = (ranks,)

        # Iterate over model ranks, optimize multiple replicates at each rank.
        for r in ranks:

            # Initialize storage
            if r not in self.results:
                self.results[r] = []

            # Display fitting progress.
            if verbose:
                itr = trange(replicates,
                             desc='Fitting rank-{} models'.format(r),
                             leave=False)
            else:
                itr = range(replicates)

            # Fit replicates.
            for i in itr:
                model_fit = self._fit_method(X, r, **self._fit_options)
                self.results[r].append(model_fit)

            # Print summary of results.
            if verbose:
                itr.close()
                itr.refresh()
                min_obj = min([res.obj for res in self.results[r]])
                max_obj = max([res.obj for res in self.results[r]])
                elapsed = sum([res.total_time for res in self.results[r]])
                print('Rank-{} models:  min obj, {:.2f};  '
                      'max obj, {:.2f};  time to fit, '
                      '{:.1f}s'.format(r, min_obj, max_obj, elapsed), flush=True)

        # Sort results from lowest to largest loss.
        for r in ranks:
            idx = np.argsort([result.obj for result in self.results[r]])
            self.results[r] = [self.results[r][i] for i in idx]

        # Align best model within each rank to best model of next larger rank.
        # Here r0 is the rank of the lower-dimensional model and r1 is the rank
        # of the high-dimensional model.
        for i in reversed(range(1, len(ranks))):
            r0, r1 = ranks[i-1], ranks[i]
            U = self.results[r0][0].factors
            V = self.results[r1][0].factors
            kruskal_align(U, V, permute_U=True)

        # For each rank, align everything to the best model
        for r in ranks:
            # store best factors
            U = self.results[r][0].factors       # best model factors
            self.results[r][0].similarity = 1.0  # similarity to itself

            # align lesser fit models to best models
            for res in self.results[r][1:]:
                res.similarity = kruskal_align(U, res.factors, permute_V=True)

    def objectives(self, rank):
        """Returns objective values of models with specified rank.
        """
        self._check_rank(rank)
        return [result.obj for result in self.results[rank]]

    def similarities(self, rank):
        """Returns similarity scores for models with specified rank.
        """
        self._check_rank(rank)
        return [result.similarity for result in self.results[rank]]

    def factors(self, rank):
        """Returns KTensor factors for models with specified rank.
        """
        self._check_rank(rank)
        return [result.factors for result in self.results[rank]]

    def _check_rank(self, rank):
        """Checks if specified rank has been fit.

        Parameters
        ----------
        rank : int
            Rank of the models that were queried.

        Raises
        ------
        ValueError: If no models of rank ``rank`` have been fit yet.
        """
        if rank not in self.results:
            raise ValueError('No models of rank-{} have been fit.'
                             'Call Ensemble.fit(tensor, rank={}, ...) '
                             'to fit these models.'.format(rank))
