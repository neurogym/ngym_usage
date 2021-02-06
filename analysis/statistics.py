import warnings
import numpy as np
import scipy

from elephant.statistics import optimal_kernel_bandwidth
import elephant.kernels as kernels
import quantities as pq


def myrate(spiketrain, sampling_period, kernel='auto',
                       cutoff=5.0, t_start=None, t_stop=None, trim=False,
                       center_kernel=True):
    """
    Estimates instantaneous firing rate by kernel convolution.

    Modified by gryang from elephant.statistics.instantaneous_rate. Much
    faster for many spike trains.

    Parameters
    ----------
    spiketrain : list of lists of spike times
        Neo object(s) that contains spike times, the unit of the time stamps,
        and `t_start` and `t_stop` of the spike train.
    sampling_period : float (s)
        Time stamp resolution of the spike times. The same resolution will
        be assumed for the kernel.

    The rest are the same as elephant.statistics.instantaneous_rate,
    abbreviated here.
    """

    if kernel == 'auto':
        kernel_width_sigma = None
        if len(spiketrain) > 0:
            kernel_width_sigma = optimal_kernel_bandwidth(
                spiketrain.magnitude, times=None, bootstrap=False)['optw']
        if kernel_width_sigma is None:
            raise ValueError(
                "Unable to calculate optimal kernel width for "
                "instantaneous rate from input data.")
        kernel = kernels.GaussianKernel(kernel_width_sigma * spiketrain.units)
    elif not isinstance(kernel, kernels.Kernel):
        raise TypeError(
            "'kernel' must be either instance of class elephant.kernels.Kernel"
            " or the string 'auto'. Found: %s, value %s" % (type(kernel),
                                                            str(kernel)))

    # TODO: do the single spike train case
    n_spiketrain = len(spiketrain)  # Number of spike trains

    # main function:
    nbins = int((t_stop - t_start) / sampling_period) + 1
    time_vectors = np.zeros((n_spiketrain, nbins))
    ranges = (t_start, t_stop+sampling_period)
    times = np.arange(ranges[0], ranges[1], sampling_period)
    for i, st in enumerate(spiketrain):
        # See https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/
        time_vectors[i], _ = np.histogram(st, bins=nbins, range=ranges)
        # c = ((st[(st >= ranges[0]) & (st < ranges[1])] - ranges[0]) /
        #      sampling_period).astype(np.int_)
        # time_vectors[i] = np.bincount(c)

    # This line is necessary to match elephant's original implementation
    time_vectors[:, -1] = 0
    time_vectors = time_vectors.T  # make it (time, units)
    time_vectors = time_vectors.astype(np.float64)  # from elephant

    if cutoff < kernel.min_cutoff:
        cutoff = kernel.min_cutoff
        warnings.warn("The width of the kernel was adjusted to a minimally "
                      "allowed width.")

    sigma = kernel.sigma.rescale(pq.s).magnitude
    t_arr = np.arange(-cutoff * sigma, cutoff * sigma + sampling_period,
                      sampling_period) * pq.s

    if center_kernel:
        # keep the full convolve range and do the trimming afterwards;
        # trimming is performed according to the kernel median index
        fft_mode = 'full'
    elif trim:
        # no median index trimming is involved
        fft_mode = 'valid'
    else:
        # no median index trimming is involved
        fft_mode = 'same'

    _kernel = kernel(t_arr).rescale(pq.Hz).magnitude[:, np.newaxis]
    rates = scipy.signal.fftconvolve(
        time_vectors, _kernel, mode=fft_mode, axes=0)

    median_id = kernel.median_index(t_arr)
    # the size of kernel() output matches the input size
    kernel_array_size = len(t_arr)
    if center_kernel:
        # account for the kernel asymmetry
        if not trim:
            rates = rates[median_id: -kernel_array_size + median_id]
        else:
            rates = rates[2 * median_id: -2 * (kernel_array_size - median_id)]
    else:
        # (to be consistent with center_kernel=True)
        # n points have n-1 intervals;
        # instantaneous rate is a list of intervals;
        # hence, the last element is excluded
        rates = rates[:-1]

    return rates, times[:-1]


if __name__ == '__main__':
    import time
    from elephant.statistics import instantaneous_rate
    from neo.core import SpikeTrain

    sampling_period = 0.01
    t_start = -2
    t_stop = 2
    X = [np.random.uniform(-3, 3, size=(np.random.randint(9000, 11000),)) for i
         in range(100)]
    kernel_sigma = 0.05
    kernel = kernels.GaussianKernel(50 * pq.ms)

    t0 = time.time()
    Rate = list()
    for i in range(len(X)):
        spiketrain = SpikeTrain(X[i] * pq.s, t_start=-3*pq.s, t_stop=3*pq.s)
        rate = instantaneous_rate(spiketrain, sampling_period=0.01 * pq.s,
                                  t_start=-2 * pq.s, t_stop=2 * pq.s,
                                  kernel=kernel)
        Rate.append(rate.magnitude[:, 0])
    Rate = np.array(Rate).T
    time_taken0 = (time.time() - t0)

    t0 = time.time()
    Rate2, times = myrate(X, sampling_period=0.01, t_start=-2, t_stop=2,
                          kernel=kernel)
    time_taken1 = (time.time() - t0)
    print('Original {:0.4f}s, New {:0.4f}s, Speed up {:0.2f}X'.format(
        time_taken0, time_taken1, time_taken0/time_taken1
    ))

    print('Results are the same:', np.allclose(Rate, Rate2))

