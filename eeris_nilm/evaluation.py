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


# TODO: Documentation, break this function into a dataset-specific component and
# TODO: Examine the steady-state alternative as well
# a general-purpose component
def hart_redd_evaluation(redd_path, house='house_1',
                         date_start='2011-04-17T00:00',
                         date_end='2011-05-30T23:59'):
    # TODO: Pydoc string
    path = os.path.join(redd_path, house)
    data, labels = redd.read_redd(path, date_start=date_start,
                                  date_end=date_end)
    # Build the model
    model = hart.Hart85eeris(installation_id=1)
    step = 6 * 3600
    for i in range(0, data['mains'].shape[0], step):
        if i % 3600 == 0:
            print("Hour count: %d" % (i / 3600))
        y = data['mains'].iloc[i:min([i+step, data['mains'].shape[0]])]
        model.update(y)
    print('Finished')

    # Create ground truth appliances (based on edges)
    gt_appliances = dict()
    power = dict()
    for i in labels.index:
        # For each appliance
        category = labels.loc[i, 'label']
        if category == 'mains':
            continue
        name = category + '_' + str(i)
        model_g = hart.Hart85eeris(installation_id=1)
        g = appliance.Appliance(i, name, category)
        # Train a Hart model
        for j in range(0, data[i].shape[0], step):
            if j % 3600 == 0:
                print('Appliance %s, hour count %d' % (name, j/3600))
            y = data[i].iloc[j:min([j+step, data[i].shape[0]])]
            # Insert face reactive and voltage columns.
            y.insert(1, 'reactive', 0.0)
            y.insert(1, 'voltage', g.nominal_voltage)
            model_g.update(y)
        # Signatures of detected appliances
        for a in model_g.appliances.values():
            g.append_signature(a.signature)
        # Failed to create signature
        if g.signature is None:
            continue
        gt_appliances[name] = g
        # Perform all evaluations using normalized and resampled data
        p = utils.get_normalized_data(data[i],
                                      nominal_voltage=g.nominal_voltage)
        p = p[['active']]
        p = utils.preprocess_data(p)
        power[g] = p

    # Match detected appliances to ground truth.
    mapping_g = dict()
    mapping = dict()  # Needed for debugging/experiments
    distance = {a: 1e6 for a in model.appliances.values()}
    est_power = dict()
    for g_k, g in gt_appliances.items():
        mapping_g[g] = []
        for a_k, a in model.appliances.items():
            match, d, index = \
                appliance.Appliance.match_power_state(a, g, t=35.0,
                                                      lp=350,
                                                      m=0.1)
            if match and d < distance[a]:
                # Strict evaluation: Each detected appliance matches exactly one
                # ground truth appliance
                if a in mapping.keys():
                    g_old = mapping[a]
                    mapping_g[g_old].remove(a)
                mapping[a] = g
                mapping_g[g].append(a)
        if mapping_g[g]:
            # Create one estimated power consumption curve per ground-truth
            # appliance
            est_power[g] = utils.power_curve_from_activations(mapping_g[g],
                                                              start=date_start,
                                                              end=date_end)
        else:
            est_power[g] = None
    # TODO: matched variable in following
    # Evaluation section
    eval_g = dict()
    eval_est = dict()
    jaccard = dict()
    rmse = dict()
    for g_k, g in gt_appliances.items():
        # Make sure date range is correct
        p = power[g]['active'].copy()
        start_g = p.index[0]
        end_g = p.index[-1]
        # Take only the interval where ground truth and estimate overlap.
        if est_power[g] is not None:
            start_est = est_power[g].index[0]
            end_est = est_power[g].index[-1]
            if start_est > start_g:
                start = start_est
            else:
                start = start_g
            if end_est > end_g:
                end = end_g
            else:
                end = end_est
            pg = p[start:end].values
            pg_b = pg > 15.0
            pa = est_power[g][start:end]['active'].values
            pa_b = pa > 0.0
        else:
            pg = p.values
            pg_b = pg > 15.0  # TODO: Arbitrary threshold
            pa = np.zeros(pg.shape)
            pa_b = np.zeros(pg_b.shape)
        eval_g[g] = pg
        eval_est[g] = pa
        jaccard[g] = metrics.jaccard_score(pg_b, pa_b)
        print('Jaccard %s: %f' % (g.name, jaccard[g]))
        rmse[g] = np.sqrt(metrics.mean_squared_error(pg, pa))
        print('RMSE %s: %f' % (g.name, rmse[g]))
    return gt_appliances, eval_g, eval_est, jaccard, rmse
    # TODO: Evaluation at segment level
    # jaccard_index
    # rmse
    # precision
    # recall
    # roc
