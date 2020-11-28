"""
The MIT License (MIT)

Copyright (c) 2020 Benjamin Antin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

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


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat

from sklearn.decomposition import PCA

import matplotlib.patheffects as pe


def preprocess(datas,
               times,
               tstart=-50,
               tend=150,
               soft_normalize=5,
               subtract_cc_mean=True,
               pca=True,
               num_pcs=6):
    """
    Preprocess data for jPCA.

    Args
    ----
        datas: List of trials, where each element of the list has shape Times x Neurons.
            As an example, this might be the output of load_churchland_data()

        times: List of times for the experiment. Typically, time zero corresponds
            to a stimulus onset. This list is used for extracting the set of
            data to be analyzed (see tstart and tend args).

        tstart: Integer. Starting time for analysis. For example, if times is [-10, 0 , 10]
                and tstart=0, then the data returned by this function will start
                at index 1.

        tend: Integer. Ending time for analysis.

        soft_normalize: Float or Int. Constant used during soft-normalization preprocessing step.
                        Adapted from original jPCA matlab code. Normalized firing rate is
                        computed by dividing by the range of the unit across all conditions and times
                        plus the soft_normalize constant: Y_{cij} = (max(Y_{:i:}) - min(Y_{:i:}) + C)
                        where Y_{cij} is the cth condition, ith neuron, at the jth time bin.
                        C is the constant provided by soft_normalize. Set C negative to skip the
                        soft-normalizing step.

        subtract_cc_mean: Boolean. Whether or not to subtract the mean across conditions. Default True.

        pca: Boolean. True to perform PCA as a preprocessing step. Defaults to True.

        num_pcs: Int. When pca=True, controls the number of PCs to use. Defaults to 6.

    Returns
    -------
        data_list: List of arrays, each T x N. T will depend on the values
            passed for tstart and tend. N will be equal to num_pcs if pca=True.
            Otherwise the dimension of the data will remain unchanged.
        full_data_variance: Float, variance of original dataset.
        pca_variance_captured: Array, variance captured by each PC.
                               If pca=False, this is set to None.
    """
    datas = np.stack(datas, axis=0)
    num_conditions, num_time_bins, num_units = datas.shape

    if soft_normalize > 0:
        fr_range = np.max(datas, axis=(0, 1)) - np.min(datas, axis=(0, 1))
        datas /= (fr_range + soft_normalize)

    if subtract_cc_mean:
        cc_mean = np.mean(datas, axis=0, keepdims=True)
        datas -= cc_mean

    # For consistency with the original jPCA matlab code,
    # we compute PCA using only the analyzed times.
    idx_start = times.index(tstart)
    idx_end = times.index(tend) + 1  # Add one so idx is inclusive
    num_time_bins = idx_end - idx_start
    datas = datas[:, idx_start:idx_end, :]

    # Reshape to perform PCA on all trials at once.
    X_full = np.concatenate(datas)
    full_data_var = np.sum(np.var(X_full, axis=0))
    pca_variance_captured = None

    if pca:
        pca = PCA(num_pcs)
        datas = pca.fit_transform(X_full)
        datas = datas.reshape(num_conditions, num_time_bins, num_pcs)
        pca_variance_captured = pca.explained_variance_

    data_list = [x for x in datas]
    return data_list, full_data_var, pca_variance_captured


def plot_trajectory(ax, x, y,
                    color="black",
                    outline="black",
                    circle=True,
                    arrow=True,
                    circle_size=0.05,
                    arrow_size=0.05):
    """
    Plot a single neural trajectory in a 2D plane.

    Args
    ----
        ax: Axis used for plotting.

        x: Values of variable on x-axis.

        y: Values of variable on y-axis.

        color: Fill color of line to be plotted. Defaults to "black".

        outline: Outline color of line. Defaults to "black".

        circle: True if the trajectory should have a circle at its starting state.

        arrow: True if the trajectory should have an arrow at its ending state.

    """
    ax.plot(x, y,
            color=color,
            path_effects=[pe.Stroke(linewidth=2, foreground=outline),
                          pe.Normal()])

    if circle:
        circ = plt.Circle((x[0], y[0]),
                          radius=circle_size,
                          facecolor=color,
                          edgecolor="black")
        ax.add_artist(circ)

    if arrow:
        dx = x[-1] - x[-2]
        dy = y[-1] - y[-2]
        px, py = (x[-1], y[-1])
        ax.arrow(px, py, dx, dy,
                 facecolor=color,
                 edgecolor=outline,
                 length_includes_head=True,
                 head_width=arrow_size)


def plot_projections(data_list,
                     x_idx=0,
                     y_idx=1,
                     axis=None,
                     arrows=True,
                     circles=True,
                     arrow_size=0.05,
                     circle_size=0.05):
    """
    Plot trajectories found via jPCA or PCA.

    Args
    ----
        data_list: List of trajectories, where each entry of data_list is an array of size T x D,
                   where T is the number of time-steps and D is the dimension of the projection.
        x_idx: column of data which will be plotted on x axis. Default 0.
        y_idx: column of data which will be plotted on y axis. Default 0.
        arrows: True to add arrows to the trajectory plot.
        circles: True to add circles at the beginning of each trajectory.
        sort_colors: True to color trajectories based on the starting x coordinate. This mimics
                     the jPCA matlab toolbox.
    """
    if axis is None:
        fig = plt.figure(figsize=(5, 5))
        axis = fig.add_axes([1, 1, 1, 1])

    colormap = plt.cm.RdBu
    colors = np.array([colormap(i) for i in np.linspace(0, 1, len(data_list))])
    data_list = [data[:, [x_idx, y_idx]] for data in data_list]

    start_x_list = [data[0, 0] for data in data_list]
    color_indices = np.argsort(start_x_list)

    for i, data in enumerate(np.array(data_list)[color_indices]):
        plot_trajectory(axis,
                        data[:, 0],
                        data[:, 1],
                        color=colors[i],
                        circle=circles,
                        arrow=arrows,
                        arrow_size=arrow_size,
                        circle_size=circle_size)


def load_churchland_data(path):
    """
    Load data from Churchland, Cunningham et al, Nature 2012
    Data available here:
    https://churchland.zuckermaninstitute.columbia.edu/content/code

    Returns a 3D data array, C x T x N. T is the number of time bins,
    and N is the number of neurons (218), and C is the number of conditions.
    Note: Loading a .mat struct in Python is quite messy beause the formatting is messed up,
    which is why we need to do some funky indexing here.
    """
    struct = loadmat(path)
    conditions = struct["Data"][0]

    # For this dataset, times are the same for all conditions,
    # but they are formatted strangely -- each element of the times
    # vector is in a separate list.
    datas = None
    times = [t[0] for t in conditions[0][1]]

    for cond in conditions:
        spikes = cond[0]
        if datas is None:
            datas = spikes
        else:
            datas = np.dstack((datas, spikes))

    datas = np.moveaxis(datas, 2, 0)
    datas = [x for x in datas]
    return datas, times


def ensure_datas_is_list(f):
    def wrapper(self, datas, **kwargs):
        datas = [datas] if not isinstance(datas, list) else datas
        return f(self, datas, **kwargs)

    return wrapper


def skew_sym_regress(X, X_dot, tol=1e-4):
    """
    Original data tensor is C x L x N where N is number of Neurons, L is length of each trial
    and C is number of conditions. We stack this to get L*C x N array.
    Args
    ----
      X_dot: First difference of (reduced dimension) data. Shape is T x N

      X: reduced dimension data. Shape is T x N
    """

    # 1) Initialize h using the odd part of the least-squares solution.
    # 2) call scipy.optimize.minimize and pass in our starting h, and x_dot,
    T, N = X.shape
    M_lstq, _, _, _ = np.linalg.lstsq(X, X_dot, rcond=None)
    M_lstq = M_lstq.T
    M_init = 0.5 * (M_lstq - M_lstq.T)
    h_init = _reshape_mat2vec(M_init, N)

    options = dict(maxiter=10000, gtol=tol)
    result = minimize(lambda h: _objective(h, X, X_dot),
                      h_init,
                      jac=lambda h: _grad_f(h, X, X_dot),
                      method='CG',
                      options=options)
    if not result.success:
        print("Optimization failed.")
        print(result.message)
    M = _reshape_vec2mat(result.x, N)
    assert (np.allclose(M, -M.T))
    return M


def _grad_f(h, X, X_dot):
    _, N = X.shape
    M = _reshape_vec2mat(h, N)
    dM = (X.T @ X @ M.T) - X.T @ X_dot
    return _reshape_mat2vec(dM.T - dM, N)


def _objective(h, X, X_dot):
    _, N = X.shape
    M = _reshape_vec2mat(h, N)
    return 0.5 * np.linalg.norm(X @ M.T - X_dot, ord='fro') ** 2


def _reshape_vec2mat(h, N):
    M = np.zeros((N, N))
    upper_tri_indices = np.triu_indices(N, k=1)
    M[upper_tri_indices] = h
    return M - M.T


def _reshape_mat2vec(M, N):
    upper_tri_indices = np.triu_indices(N, k=1)
    return M[upper_tri_indices]
