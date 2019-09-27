"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential


Functions for evaluation of NILM algorithm performance using various
metrics. Also includes functions for evaluation of unsupervised algorithms
(unknown appliances during algorithm training and prediction).
"""

import numpy as np

# TODO: Possibly also extract
# 1. Percentage of energy identified
# 2. Accuracy of device detection

# TODO:
# 1. Wrapper in the case of mapping between automatically detected appliance and
# known appliance
# 2. Wrapper of the above that also performs the mapping of unknown appliances
# to the known appliances

# TODO: Check documentation format.


def rmse(curve1, curve2):
    """
    This function computes the root mean squared error (RMSE) between two load
    curves. The curves are numpy arrays, so it is required that they have the
    same length and that they correspond to the same time samples.

    Parameters
    ----------
    curve1 : numpy.array. Nx1 numpy array corresponding to the first load curve

    curve2 : numpy.array. Nx1 numpy array corresponding to the second load curve

    Returns
    -------
    out : The computed RMSE value

    """
    assert all(curve1.shape == curve2.shape)
    e = np.sqrt(np.mean((curve1 - curve2) ** 2))
    return e


def mae(curve1, curve2):
    """
    This function computes the mean absolute error (MAE) between two load
    curves. The curves are numpy arrays, so it is required that they have the
    same length and that they correspond to the same time samples.

    Parameters
    ----------
    curve1 : Nx1 numpy array corresponding to the first load curve

    curve2 : Nx1 numpy array corresponding to the second load curve

    Returns
    -------
    out : The computed MAE value

    """
    assert all(curve1.shape == curve2.shape)
    e = np.mean(np.fabs(curve1 - curve2))
    return e


def mape(curve1, curve2):
    """
    This function computes the mean absolute percentage error (MAPE) between two
    load curves. The curves are numpy arrays, so it is required that they have
    the same length and that they correspond to the same time samples. The
    computed MAPE value is given by

    .. math::

    MAPE = \frac{1}{N}\sum_{i=1}^{N}\lVert \frac{c_1(i) - c_2(i)}{c_1(i) +
    1.0}\rVert

    Notice the addition of 1.0 Watt in the denominator to avoid division by
    zero.

    Parameters
    ----------
    curve1 : numpy.array. Nx1 numpy array corresponding to the first load curve

    curve2 : numpy.array. Nx1 numpy array corresponding to the second load curve

    Returns
    -------
    out : The computed MAPE value

    """
    assert all(curve1.shape == curve2.shape)
    e = np.mean(np.fabs((curve1 - curve2) / (curve1 + 1.0)))
    return e


def jaccard_index(intervals1, intervals2):
    """ Compute the Jaccard index (intersection over union) for the estimated
    durations of activation of an appliance. A value close to 1 indicates
    perfect overlap, while a value of zero indicates no overlap. In the context
    of NILM we use it to measure the overlap of appliance "activation" time, as
    measured by the intervals that the device consumes power, between the
    measured power consumption and the ground truth.

    Parameters
    ----------
    intervals1 : Nx2 numpy.array. First set of start/stop timestamp pairs.

    intervals2 : Nx2 numpy.array. Second set of start/stop timestamp pairs.


    Returns
    -------
    out : Number. The value of the Jaccard index (in [0, 1]).

    """
    # Input checks
    if intervals1.shape[1] != 2 or intervals2.shape[1] != 2:
        raise ValueError('Input arrays need to have two columns')

    # Intersection
    def _intersection(a, b):
        if b[1] < a[0] or a[1] < b[0]:
            # No intersection
            return 0.0
        if b[0] < a[0]:
            # a starts after b
            if b[1] < a[1]:
                # partial overlap
                return (b[1] - a[0])
            else:
                # a is fully included in b
                return (a[1] - a[0])
        else:
            # b starts after a
            if a[1] < b[1]:
                # partial overlap
                return (a[1] - b[0])
            else:
                # b is fully included in a
                return (b[1] - b[0])
        # We should never be here
        assert 1 == 0
        return None

    i1_sorted = intervals1[np.argsort(intervals1[:, 0], axis=0), :]
    i2_sorted = intervals2[np.argsort(intervals2[:, 0], axis=0), :]
    for row1 in i1_sorted:
        ovlp_idx = np.logical_and(i2_sorted[:, 1] > row1[0],
                                  i2_sorted[:, 0] < row1[1])
        ovlp = i2_sorted[ovlp_idx, :]
        for row2 in ovlp:
            intersection = _intersection(row1, row2)

    # Union
    intervals = np.concat((i1_sorted, i2_sorted), axis=0)
    intervals = intervals[np.argsort(intervals[:, 0], axis=0), :]
    i_start = intervals[0, 0]
    i_end = intervals[0, 1]
    union = 0.0
    for i in range(1, intervals.shape[0]):
        if intervals[i, 0] < i_end and intervals[i, 1] > i_end:
            i_end = intervals[i, 1]
        elif intervals[i, 0] > i_end:
            union += i_end - i_start
            i_start = intervals[i, 0]
            i_end = intervals[i, 1]
        elif intervals[i, 1] < i_end:
            pass
    union += i_end - i_start

    # Finally, compute the Jaccard index.
    if union <= 10 * np.finfo.eps:
        j_index = 0.0
    else:
        j_index = intersection / union
    return j_index


# def rmse_twostate(matches_d, lcurve_gt):
#     """
#     This function computes the root mean squared error (RMSE) of a set of
#     matching pairs (on and off cycles) of a two-state appliance compared to a
#     ground truth load curve (presumably for the same appliance). The function
#     first reconstructs the estimated load curve of the two-state appliance and
#     then performs the comparison.

#     Parameters
#     ----------
#     matches_d : Nx2 numpy array corresponding to detected matches of appliance
#     switch on/off events.

#     lcurve_gt : A pandas dataframe with measurements. Comparison is made in
#     terms of the active power (column 'active' in the dataframe). It is
#     assumed that the measurements are taken with constant sampling rate.

#     Returns
#     -------
#     out : The computed RMSE value

#     """
#     fs = lcurve_gt.freq
#     if fs is None:
#         raise ValueError('lcurve_gt does not have constant sampling rate. ' +
#                          'pre-process the samples first.')
#     curve1 = lcurve_gt['active'].values
#     curve2 =
