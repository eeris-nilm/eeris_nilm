"""
Copyright 2020 Christos Diou

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import os.path
from sklearn import metrics
from eeris_nilm import appliance
from eeris_nilm.datasets import redd
from eeris_nilm.algorithms import hart
from eeris_nilm import utils

# TODO: Possibly also extract
# 1. Percentage of energy identified
# 2. Accuracy of device detection

# TODO:
# 1. Wrapper in the case of mapping between automatically detected appliance and
# known appliance
# 2. Wrapper of the above that also performs the mapping of unknown appliances
# to the known appliances

# TODO: Replace these metrics with the corresponding metrics of scikit-learn


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


def hart_redd_evaluation(redd_path, house='house_1',
                         date_start='2011-04-18T00:00',
                         date_end='2011-04-25T23:59'):

    p = os.path.join(redd_path, house)
    data, labels = redd.read_redd(p, date_start='2011-04-18T00:00',
                                  date_end='2011-04-25T23:59')
    # Build the model
    model = hart.Hart85eeris(installation_id=1)
    step = 6 * 3600
    for i in range(0, data['mains'].shape[0], step):
        if i % 3600 == 0:
            print("Hour count: %d" % (i / 3600))
        y = data['mains'].iloc[i:min([i+step, data['mains'].shape[0]])]
        model.update(y)
    print('Finished')

    # Create ground truth appliances
    gt_appliances = dict()
    power = dict()
    power_binary = dict()
    for i in labels.index:
        category = labels.loc[i, 'label']
        if category == 'mains':
            continue
        name = category + '_' + str(i)
        g = appliance.Appliance(i, name, category)
        g.signature_from_data(data[i])
        gt_appliances[name] = g
        power[g] = data[i]
        power_binary[g] = power[g] > 15.0

    # Match detected appliances to ground truth
    mapping = dict()
    mapping_g = dict()
    distance = dict()
    est_power = dict()
    for g_k, g in gt_appliances.items():
        mapping_g[g] = []
        matched = False
        for a_k, a in model.appliances.items():
            match, d, index = \
                appliance.Appliance.match_power_state(a, g, t=35.0)
            if match and d < distance[a]:
                mapping[a] = g
                distance[a] = d
                mapping_g[g].append(a)
                matched = True
        if matched:
            # Create one power consumption curve per ground-truth appliance
            est_power[g] = utils.power_curve_from_activations(mapping_g[g])
            est_power_binary[g] = est_power[g] > 0.0
    # TODO: matched variable in following
    # Evaluation section
    jaccard = dict()
    for g_k, g in gt_appliances.items():
        pg = power[g].values
        pa = est_power[g].values
        pg_b = power_binary[g].values
        pa_b = est_power_binary[g].values
        jaccard[g] = metrics.jaccard_score(pg_b, pa_b)
        print('Jaccard %s: %f' % (g.name, jaccard[g]))
        rmse[g] = np.sqrt(metrics.mean_squared_error(pg, pa))
        print('RMSE %s: %f' % (g.name, rmse[g]))
    return jaccard, rmse
    # TODO: Evaluation at segment level
    # jaccard_index
    # rmse
    # precision
    # recall
    # roc
