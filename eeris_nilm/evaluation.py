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
import logging
import dill
from sklearn import metrics
from eeris_nilm import appliance
from eeris_nilm.datasets import redd
from eeris_nilm.algorithms import livehart
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


# NOT USED YET
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


def hart_redd_evaluation(redd_path, house='house_1',
                         date_start='2011-04-17T00:00',
                         date_end='2011-05-30T23:59',
                         mode='edges',
                         step=None):
    """
    Evaluate performance of implementation of Hart's algorithm to the REDD
    dataset.

    Parameters
    ----------

    redd_path : str
    Path to the REDD dataset

    house : str
    REDD house name

    date_start : str
    Start date. If earlier than the earliest date, the start time of the data
    is used.

    date_end : str
    End date. If later than the latest date, the end time of the data
    is used.

    mode : str, can be 'edges' or 'steady_states'
    Method to compute the ground truth power states. In 'edges' mode (the
    default), edges and Hart's algorithm are used to determine the transitions
    of the appliance. In 'steady_states' method, the detected steady states of
    the appliance are used instead. This argument is directly passed to the
    'hart_evaluation' function.

    step : int
    Process the data in batches of 'step' seconds at a time. If None, then the
    entire dataset is loaded into memory. WARNING: No memory checks are
    performed, so please make sure that the dataset can fit into memory.  This
    argument is directly passed to the 'hart_evaluation' function.

    Returns
    -------
    gt_appliances : Dictionary of detected ground truth appliances. Appliance
    names are the keys (in the form name_id, where id is an appliance counter).

    eval_g : Dictionary of numpy arrays, one for each ground truth appliance
    The data that is used for evaluation (normalized, resampled, and segmented
    for the duration where estimates from NILM/mains are available)

    eval_est : Dictionary of numpy arrays, one for each detected appliance
    The data that is used for evaluation (normalized, resampled, and segmented
    for the duration where ground truth data are available)

    jaccard : Dictionary of float
    Jaccard index values, one for each ground truth appliance

    rmse : Dictionary of float
    Root mean squared error values, one for each ground truth appliance

    """
    path = os.path.join(redd_path, house)
    data, labels = redd.read_redd(path, date_start=date_start,
                                  date_end=date_end)
    return hart_evaluation(data, labels, mode='edges', step=None)


# TODO: Evaluate using best matching instead of greedy matching
def hart_evaluation(data, labels, mode='edges', step=None):
    """
    Evaluate performance of Hart's algorithm implementation.

    Parameters
    ----------

    data: Dictionary of pandas.DataFrame objects
    One data['mains'] object is expected, corresponding to the installation
    meter. Additional objects correspond to meters of individual appliances and
    are used as ground truth (except those that have a label 'mains', see
    below).

    labels: pandas.Dataframe
    A dataframe with a 'label' column, indicating the appliance category
    corresponding to each of the dataframes in the 'data' dictionary. The
    dataframes with label 'mains' are not used for evaluation.

    mode : str, can be 'edges' or 'steady_states'
    Method to compute the ground truth power states. In 'edges' mode (the
    default), edges and Hart's algorithm are used to determine the transitions
    of the appliance. In 'steady_states' method, the detected steady states of
    the appliance are used instead

    step : int
    Process the data in batches of 'step' seconds at a time. If None, then the
    entire dataset is loaded into memory. WARNING: No memory checks are
    performed, so please make sure that the dataset can fit into memory.

    Returns
    -------
    gt_appliances : Dictionary of detected ground truth appliances. Appliance
    names are the keys (in the form name_id, where id is an appliance counter).

    eval_g : Dictionary of numpy arrays, one for each ground truth appliance
    The data that is used for evaluation (normalized, resampled, and segmented
    for the duration where estimates from NILM/mains are available)

    eval_est : Dictionary of numpy arrays, one for each detected appliance
    The data that is used for evaluation (normalized, resampled, and segmented
    for the duration where ground truth data are available)

    jaccard : Dictionary of float
    Jaccard index values, one for each ground truth appliance

    rmse : Dictionary of float
    Root mean squared error values, one for each ground truth appliance

    """
    # Build the model
    if step is None:
        model = livehart.LiveHart(installation_id=1, batch_mode=True)
        y = data['mains']
        model.update(y)
    else:
        # Perhaps this should be batch as well, if step is larger than a few
        # hours.
        model = livehart.LiveHart(installation_id=1, batch_mode=False)
        for i in range(0, data['mains'].shape[0], step):
            if i % 3600 == 0:
                logging.debug("Hour count: %d" % (i / 3600))
            y = data['mains'].iloc[i:min([i+step, data['mains'].shape[0]])]
            model.update(y)
    print('Finished')

    # Create ground truth appliances (based on edges or steady states)
    gt_appliances = dict()
    power = dict()
    for i in labels.index:
        # For each appliance
        category = labels.loc[i, 'label']
        if category == 'mains':
            continue
        name = category + '_' + str(i)
        g = appliance.Appliance(i, name, category)
        if mode == 'edges':
            # Train a Hart model
            if step is None:
                # Process everything at once
                model_g = livehart.LiveHart(installation_id=1, batch_mode=True)
                y = data[i]
                y.insert(1, 'reactive', 0.0)
                y.insert(1, 'voltage', g.nominal_voltage)
                model_g.update(y)
            else:
                # Step-processing.
                model_g = livehart.LiveHart(installation_id=1, batch_mode=False)
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
        elif mode == 'steady_states':
            g.signature_from_data_steady_states(data[i])
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
            try:
                match, d, index = \
                    appliance.Appliance.match_power_state(a, g, t=35.0,
                                                          lp=350,
                                                          m=0.1)
            except ValueError:
                continue
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
            est_power[g] = utils.power_curve_from_activations(mapping_g[g])
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


def consistency(h, n=15):
    """
    TODO
    """
    # TODO: Option to infer n?
    entropy = -np.sum(np.array([h[i]*np.log2(h[i]) for i in h.keys()]))
    entropy_max = -np.sum(np.array([1/n*np.log2(1/n) for i in range(n)]))
    c = 1.0 - (entropy / entropy_max)
    return c


def live_run(mains, step=3):
    """
    Run the live algorithma and create a live model.
    """
    # Train main model with step-second batches
    model = livehart.LiveHart(installation_id=1, batch_mode=False,
                              store_live=True)
    report_step = 3600 // step
    end = mains.shape[0] - mains.shape[0] % step
    for i in range(0, end, step):
        if i % report_step == 0:
            logging.debug("Processing at second %d" % (i))
        y = mains.iloc[i:i+step]
        model.update(y, start_thread=False)
    return model


def _live_metrics(appliance_h, live_h, data, tol=15):
    """
    Auxiliary function to compute appliance activation matches between live
    display and ground truth
    """
    # Prepare matches
    match_df = appliance_h.sort_values(by=['start'], ignore_index=True)
    match_df_s = appliance_h.sort_values(by=['start'], ignore_index=True)
    l_df = live_h.sort_values(by=['start'], ignore_index=True)
    match_df['matches'] = np.empty((match_df.shape[0], 0)).tolist()
    match_df_s['matches'] = np.empty((match_df_s.shape[0], 0)).tolist()
    hist = dict()
    hist_s = dict()
    hist['NA'] = 0
    hist_s['NA'] = 0
    hist_dur = dict()
    hist_dur['NA'] = 0
    # This code repeats the _activations_overlap_pct funcrtion in
    # appliance.py. Is there an efficient  way to refactor?
    idx1 = 0
    idx2 = 0
    dur1_tot = 0
    dur_match_tot = 0
    events_tot = 0
    while idx1 < match_df.shape[0]:
        start1 = match_df.iloc[idx1]['start']
        end1 = match_df.iloc[idx1]['end']
        dur1 = (end1 - start1).days * 24 * 3600 + (end1 - start1).seconds
        # Ignore very small events
        if dur1 < tol:
            idx1 += 1
            continue
        # Perhaps there are missing data in this segment in the 'mains'
        # recording?
        seg = data.loc[start1:end1]
        if seg['active'].isnull().sum().sum() > 2:
            # Ignore this
            idx1 += 1
            continue
        dur1_tot += dur1
        events_tot += 1
        dur2_match = 0
        while idx2 < live_h.shape[0]:
            start2 = l_df.iloc[idx2]['start']
            end2 = l_df.iloc[idx2]['end']
            dur2 = (end2 - start2).days * 24 * 3600 + (end2 - start2).seconds

            start_d = (start2 - start1).days * 24 * 3600 + \
                (start2 - start1).seconds
            end_d = (end2 - end1).days * 24 * 3600 + \
                (end2 - end1).seconds

            if start_d < -tol:
                idx2 += 1
                continue
            if abs(start_d) <= tol:
                # Match start
                name = l_df.iloc[idx2]['name']
                match_df_s.iloc[idx1]['matches'].append(name)
                if abs(end_d) <= tol:
                    # Match both start and end
                    match_df.iloc[idx1]['matches'].append(name)
                    # duration only for full match
                    dur2_match += dur2
                idx2 += 1
                continue
            if start_d > tol:
                break
            idx2 += 1
        ml = match_df.iloc[idx1]['matches']
        mls = match_df_s.iloc[idx1]['matches']
        if len(ml) > 0:
            if len(ml) > 1:
                # Just for warning
                logging.debug("More than one matches for %d: %s" %
                              (idx1, str(ml)))
            if ml[0] in hist.keys():
                hist[ml[0]] += 1
            else:
                hist[ml[0]] = 1
            if ml[0] in hist_dur.keys():
                hist_dur[ml[0]] += dur2_match
            else:
                hist_dur[ml[0]] = dur2_match
            dur_match_tot += dur2_match
        else:
            hist['NA'] += 1
            hist_dur['NA'] += dur1
        if len(mls) > 0:
            if len(mls) > 1:
                # Just for warning
                logging.debug("More than one matches for %d: %s" %
                              (idx1, str(mls)))
            if mls[0] in hist_s.keys():
                hist_s[mls[0]] += 1
            else:
                hist_s[mls[0]] = 1
        else:
            hist_s['NA'] += 1
        idx1 += 1
    # Compute metrics
    # Number and percentage of unmatched activations
    if events_tot > 0:
        matched_p = 1 - (hist['NA'] / events_tot)
        matched_p_s = 1 - (hist_s['NA'] / events_tot)
    else:
        logging.debug("Zero events for evaluation")
        matched_p = None
        matched_p_s = None
    if dur1_tot > 0:
        # Due to the tolerance, if there's a perfect match then it is possible
        # that matched_dur_tot > dur1_tot.
        if dur_match_tot > dur1_tot:
            dur_match_tot = dur1_tot
        matched_dur_p = dur_match_tot / dur1_tot
    else:
        logging.debug("Zero duration for evaluation")
        matched_dur_p = None
    tot = 0
    hist_n = dict()
    for k in hist.keys():
        if k == 'NA':
            continue
        else:
            hist_n[k] = hist[k]
            tot += hist[k]
    for k in hist_n.keys():
        hist_n[k] /= tot

    tot_s = 0
    hist_s_n = dict()
    for k in hist_s.keys():
        if k == 'NA':
            continue
        else:
            hist_s_n[k] = hist_s[k]
            tot_s += hist_s[k]
    for k in hist_s_n.keys():
        hist_s_n[k] /= tot_s

    metrics = dict()
    metrics['n'] = events_tot
    metrics['d'] = dur1_tot
    metrics['h'] = hist
    metrics['h_n'] = hist_n
    metrics['h_s'] = hist_s
    metrics['h_s_n'] = hist_s_n
    metrics['pm'] = matched_p
    metrics['ps'] = matched_p_s
    metrics['pd'] = matched_dur_p
    metrics['c'] = consistency(hist_n)
    metrics['c_s'] = consistency(hist_s_n)
    return metrics


def live_evaluation(data, labels, step=3, states=True, history_file=None):
    """
    Evaluate performance of Live algorithm.

    Parameters
    ----------

    data: Dictionary of pandas.DataFrame objects
    One data['mains'] object is expected, corresponding to the installation
    meter. Additional objects correspond to meters of individual appliances and
    are used as ground truth (except those that have a label 'mains', see
    below).

    labels: pandas.Dataframe
    A dataframe with a 'label' column, indicating the appliance category
    corresponding to each of the dataframes in the 'data' dictionary. The
    dataframes with label 'mains' are not used for evaluation.

    mode : str, can be 'edges' or 'steady_states'
    Method to compute the ground truth power states. In 'edges' mode (the
    default), edges and Hart's algorithm are used to determine the transitions
    of the appliance. In 'steady_states' method, the detected steady states of
    the appliance are used instead

    step : int
    Process the data in batches of 'step' seconds at a time. If None, then the
    entire dataset is loaded into memory. WARNING: No memory checks are
    performed, so please make sure that the dataset can fit into memory.

    history_file : str or None
    If not None, then all previous variables are ignored and all activation data
    are loaded from the file (a dill.dump() of LiveHart.live_history variable).

    Returns
    -------

    TODO: update


    hist_gt : dict
    Dictionary with ground truth appliance names as keys. Each dictionary
    element is also a dictionary with the live appliance names as keys, as well
    as the extra key 'NA'. Each element of the dictionary is the number of times
    each live appliance was assigned to the ground truth appliance. Thus
    hist[gt_appliance][live_appliance] is the number of times live_appliance
    matched an activation of gt_appliance.

    matched_p_gt: dict
    Dictionary with ground truth appliance names as keys. Each element is the
    percentage of activations that was matched by live appliances.

    matched_dur_p_gt: dict
    Same as matched_p_gt, but the value is the percentage of the total appliance
    activation duration that was matched by live appliances
    """
    # Get activations of appliances
    logging.debug("Computing indivudal appliance activations")
    gt_activations = dict()
    for i in labels.index:
        # For each appliance
        category = labels.loc[i, 'label']
        if category == 'mains':
            continue
        name = category + '_' + str(i)
        # TODO: Hardcoded threshold
        activations = utils.activations_from_power_curve(data[i],
                                                         states=states,
                                                         threshold=35)
        gt_activations[name] = activations

    # Get the model including the live detections
    logging.debug("Computing the model")
    if history_file is None:
        model = live_run(data['mains'], step=step)
        lh = model.live_history
    else:
        with open(history_file, 'rb') as f:
            lh = dill.load(f)
    # Compute metrics
    metrics = dict()
    for name in gt_activations.keys():
        a = gt_activations[name]
        if a is None:
            continue
        metrics[name] = _live_metrics(a, lh, data['mains'], tol=15)
    return metrics


def live_redd_evaluation(redd_path, house='house_1',
                         date_start=None,
                         date_end=None,
                         step=3, history_file=None):
    """Evaluate performance of implementation of Live algorithm to the REDD
    dataset.

    Parameters
    ----------

    redd_path : str
    Path to the REDD dataset

    house : str
    REDD house name

    date_start : str
    Start date. If earlier than the earliest date, the start time of the data
    is used.

    date_end : str
    End date. If later than the latest date, the end time of the data
    is used.

    step : int
    Process the data in batches of 'step' seconds at a time. Default is 3. This
    value should be less than 10 (ideally in the 1-3 range), for the evaluation
    to be meaningful.

    history_file : str or None
    If not None, then all previous variables are ignored and all activation data
    are loaded from the file (a dill.dump() of LiveHart.live_history variable).

    Returns
    -------

    TODO: update

    hist_gt : dict
    Dictionary with ground truth appliance names as keys. Each dictionary
    element is also a dictionary with the live appliance names as keys, as well
    as the extra key 'NA'. Each element of the dictionary is the number of times
    each live appliance was assigned to the ground truth appliance. Thus
    hist[gt_appliance][live_appliance] is the number of times live_appliance
    matched an activation of gt_appliance.

    matched_p_gt: dict
    Dictionary with ground truth appliance names as keys. Each element is the
    percentage of activations that was matched by live appliances.

    matched_dur_p_gt: dict
    Same as matched_p_gt, but the value is the percentage of the total appliance
    activation duration that was matched by live appliances

    """
    path = os.path.join(redd_path, house)
    data, labels = redd.read_redd(path, date_start=date_start,
                                  date_end=date_end)
    metrics = live_evaluation(data, labels, step, states=False,
                              history_file=history_file)
    return metrics
    # TODO: Evaluation by appliance consumption


def live_evaluation_print(metrics):
    """
    TODO
    """
    for name in metrics.keys():
        print("------------------------")
        print("Results for %s" % (name))
        print("Histogram - non normalized")
        print(metrics[name]['h'])
        if metrics[name]['pm'] is not None:
            print("Matched percentage: %f" % (metrics[name]['pm']))
            if metrics[name]['pd'] is not None:
                print("Matched duration percentage: %f" %
                      (metrics[name]['pd']))
        print("Histogram - normalized (excluding NAs)")
        print(metrics[name]['h_n'])
        print("Consistency")
        print(metrics[name]['c'])
