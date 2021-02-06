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

import scipy as sci
import numpy as np

# TODO - refactor this code to take an arbitrary random generator.
from copy import deepcopy


def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor`.

    Parameters
    ----------
    tensor : ndarray
    mode : int

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``

    Author
    ------
    Jean Kossaifi <https://github.com/tensorly>
    """
    return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))


def khatri_rao(matrices):
    """Khatri-Rao product of a list of matrices.

    Parameters
    ----------
    matrices : list of ndarray

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the
        product.

    Author
    ------
    Jean Kossaifi <https://github.com/tensorly>
    """

    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))


class KTensor(object):
    """Kruskal tensor object.

    Attributes
    ----------
    factors : list of ndarray
        Factor matrices.
    shape : tuple
        Dimensions of full tensor.
    size : int
        Number of elements in full tensor.
    rank : int
        Dimensionality of low-rank factor matrices.
    """

    def __init__(self, factors):
        """Initializes KTensor.

        Parameters
        ----------
        factors : list of ndarray
            Factor matrices.
        """

        self.factors = factors
        self.shape = tuple([f.shape[0] for f in factors])
        self.ndim = len(self.shape)
        self.size = np.prod(self.shape)
        self.rank = factors[0].shape[1]

        for f in factors[1:]:
            if f.shape[1] != self.rank:
                raise ValueError('Tensor factors have inconsistent rank.')

    def full(self):
        """Converts KTensor to a dense ndarray."""

        # Compute tensor unfolding along first mode
        unf = self.factors[0] @ khatri_rao(self.factors[1:]).T

        # Inverse unfolding along first mode
        return np.reshape(unf, self.shape)

    def norm(self):
        """Efficiently computes Frobenius-like norm of the tensor."""
        C = np.prod([F.T @ F for F in self.factors], axis=0)
        return np.sqrt(np.sum(C))

    def rebalance(self):
        """Rescales factors across modes so that all norms match."""

        # Compute norms along columns for each factor matrix
        norms = [np.linalg.norm(f, axis=0) for f in self.factors]

        # Multiply norms across all modes
        lam = np.prod(norms, axis=0) ** (1/self.ndim)

        # Update factors
        self.factors = [f * (lam / fn) for f, fn in zip(self.factors, norms)]
        return self

    def prune_(self):
        """Drops any factors with zero magnitude."""
        idx = self.factor_lams() > 0
        self.factors = [f[:, idx] for f in self.factors]
        self.rank = np.sum(idx)

    def pad_zeros_(self, n):
        """Adds n more factors holding zeros."""
        if n == 0:
            return
        self.factors = [np.column_stack((f, np.zeros((f.shape[0], n))))
                        for f in self.factors]
        self.rank += n

    def permute(self, idx):
        """Permutes the columns of the factor matrices inplace."""

        # Check that input is a true permutation
        if set(idx) != set(range(self.rank)):
            raise ValueError('Invalid permutation specified.')

        # Update factors
        self.factors = [f[:, idx] for f in self.factors]
        return self.factors

    def factor_lams(self):
        return np.prod(
            [np.linalg.norm(f, axis=0) for f in self.factors], axis=0)

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, i):
        return self.factors[i]

    def __setitem__(self, i, factor):
        factor = np.array(factor)
        if factor.shape != (self.shape[i], self.rank):
            raise ValueError('Dimension mismatch in KTensor assignment.')
        self.factors[i] = factor

    def __iter__(self):
        return iter(self.factors)


def _check_random_state(random_state):
    """Checks and processes user input for seeding random numbers.

    Parameters
    ----------
    random_state : int, RandomState instance or None
        If int, a RandomState instance is created with this integer seed.
        If RandomState instance, random_state is returned;
        If None, a RandomState instance is created with arbitrary seed.

    Returns
    -------
    scipy.random.RandomState instance

    Raises
    ------
    TypeError
        If ``random_state`` is not appropriately set.
    """
    if random_state is None or isinstance(random_state, int):
        return sci.random.RandomState(random_state)
    elif isinstance(random_state, sci.random.RandomState):
        return random_state
    else:
        raise TypeError('Seed should be None, int or np.random.RandomState')


def _rescale_tensor(factors, norm):
    # Rescale the tensor to match the specified norm.
    if norm is None:
        return factors.rebalance()
    else:
        # Compute rescaling factor for tensor
        factors[0] *= norm / factors.norm()
        return factors.rebalance()


def randn_ktensor(shape, rank, norm=None, random_state=None):
    """
    Generates a random N-way tensor with rank R, where the factors are
    generated from the standard normal distribution.

    Parameters
    ----------
    shape : tuple
        shape of the tensor

    rank : integer
        rank of the tensor

    norm : float or None, optional (defaults: None)
        If not None, the factor matrices are rescaled so that the Frobenius
        norm of the returned tensor is equal to ``norm``.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    X : (I_1, ..., I_N) array_like
        N-way tensor with rank R.

    Example
    -------
    >>> # Create a rank-2 tensor of dimension 5x5x5:
    >>> import tensortools as tt
    >>> X = tt.randn_tensor((5,5,5), rank=2)

    """

    # Check input.
    rns = _check_random_state(random_state)

    # Draw low-rank factor matrices with i.i.d. Gaussian elements.
    factors = KTensor([rns.standard_normal((i, rank)) for i in shape])
    return _rescale_tensor(factors, norm)


def rand_ktensor(shape, rank, norm=None, random_state=None):
    """
    Generates a random N-way tensor with rank R, where the factors are
    generated from the standard uniform distribution in the interval [0.0,1].

    Parameters
    ----------
    shape : tuple
        shape of the tensor

    rank : integer
        rank of the tensor

    norm : float or None, optional (defaults: None)
        If not None, the factor matrices are rescaled so that the Frobenius
        norm of the returned tensor is equal to ``norm``.

    ktensor : bool
        If true, a KTensor object is returned, i.e., the components are in factored
        form ``[U_1, U_2, ... U_N]``; Otherwise an N-way array is returned.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    X : (I_1, ..., I_N) array_like
        N-way tensor with rank R.

    Example
    -------
    >>> # Create a rank-2 tensor of dimension 5x5x5:
    >>> import tensortools as tt
    >>> X = tt.rand_tensor((5,5,5), rank=2)

    """

    # Check input.
    rns = _check_random_state(random_state)

    # Randomize low-rank factor matrices i.i.d. uniform random elements.
    factors = KTensor([rns.uniform(0.0, 1.0, size=(i, rank)) for i in shape])
    return _rescale_tensor(factors, norm)


def randexp_ktensor(shape, rank, scale=1.0, norm=None, random_state=None):
    """
    Generates a random N-way tensor with rank R, where the entries are
    drawn from an exponential distribution

    Parameters
    ----------
    shape : tuple
        shape of the tensor

    rank : integer
        rank of the tensor

    scale : float
        Scale parameter for the exponential distribution.

    norm : float or None, optional (defaults: None)
        If not None, the factor matrices are rescaled so that the Frobenius
        norm of the returned tensor is equal to ``norm``.

    ktensor : bool
        If true, a KTensor object is returned, i.e., the components are in factored
        form ``[U_1, U_2, ... U_N]``; Otherwise an N-way array is returned.

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.


    Returns
    -------
    X : (I_1, ..., I_N) array_like
        N-way tensor with rank R.

    Example
    -------
    >>> # Create a rank-2 tensor of dimension 5x5x5:
    >>> import tensortools as tt
    >>> X = tt.randexp_tensor((5,5,5), rank=2)

    """

    # Check input.
    rns = _check_random_state(random_state)

    # Randomize low-rank factor matrices i.i.d. uniform random elements.
    factors = KTensor(
        [rns.exponential(scale=scale, size=(i, rank)) for i in shape])
    return _rescale_tensor(factors, norm)
