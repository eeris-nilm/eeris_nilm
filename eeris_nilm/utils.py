"""
Copyright 2019 Christos Diou

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
import pandas as pd
import json


def get_segments(signal, mask, only_true=True):
    """
    Get the segments of a signal indicated by consecutive "True" values in a
    binary mask

    Parameters
    ----------
    signal : numpy.array
    1-D numpy array

    mask : numpy.array
    Boolean array with same shape as signal

    only_true: Boolean If it is True, only segments corresponding to True values
    of the original signal are returned. Otherwise, segments corresponding to
    both True and False values are returned.

    Returns
    -------
    out : list
    List of numpy.array elements, each containing a segment of the original
    signal.

    """
    if signal.shape[0] != mask.shape[0]:
        raise ValueError("Signal and mask shape do not match")

    # Vectorized way to identify semgments. Not straightforward, but fast.
    segments = []
    idx = np.where(np.concatenate(([True], mask[:-1] != mask[1:], [True])))[0]
    for i in range(len(idx[:-1])):
        seg = signal[idx[i]:idx[i + 1]]
        segments.append(seg)

    if only_true:
        if mask[0]:
            ret_segments = segments[::2]
        else:
            ret_segments = segments[1::2]
    else:
        ret_segments = segments
    return ret_segments


def get_normalized_data(data, nominal_voltage):
    """
    Normalize power with voltage measurements, if available. See also Hart's
    1985 paper.

    Parameters
    ----------
    data: pandas.DataFrame
    Pandas dataframe with columns 'active', 'voltage' and, optionally,
    'reactive'.

    Returns
    -------
    out: pandads.DataFrame. A normalized dataframe, where
    .. math::
    P_n(i) = P(i)\left\(\frac{V_0}{V(i)}\right\)^{1.5}\\
    Q_n(i) = Q(i)\left\(\frac{V_0}{V(i)}\right\)^{2.5}
    """
    if 'active' not in data.columns:
        raise ValueError("No \'active\' column.")

    r_data = data.copy()

    # Normalization. Raise active power to 1.5 and reactive power to
    # 2.5. See Hart's 1985 paper for an explanation.
    if 'voltage' in data.columns:
        r_data.loc[:, 'active'] = data['active'] * \
            np.power((nominal_voltage / data['voltage']), 1.5)
        if 'reactive' in data.columns:
            r_data.loc[:, 'reactive'] = data['reactive'] * \
                np.power((nominal_voltage / data['voltage']), 2.5)
    return r_data


def preprocess_data(data):
    """
    Drop duplicates and resample all data to 1 second sampling frequency.

    Parameters
    ----------
    data : pandas.DataFrame
    Pandas dataframe with the original data.

    Returns
    -------
    out : pandas.DataFrame
    Preprocessed data
    """
    # Make sure timestamps are in correct order
    out = data.sort_index()
    # Round timestamps to 1s
    out.index = out.index.round('1s')
    out = out.reset_index()
    # Remove possible entries with same timestamp. Keep last entry.
    out = out.drop_duplicates(subset='index', keep='last')
    out = out.set_index('index')
    # TODO: Do we handle NAs? (dropna)
    # Resample to 1s and fill-in the gaps
    out = out.asfreq('1S', method='pad')
    return out


def match_power(p1, p2, active_only=True, t=35.0, lp=1000, m=0.05):
    """
    Match power consumption p1 against p2 according to Hart's algorithm.

    Parameters
    ----------

    p1, p2 : 1x2 Numpy arrays (active and reactive power).

    active_only : Boolean indicating if match should take into account only
    active power or both active and reactive power

    t : Float used to determine whether there is a match or no. If the
    difference is over t, then there is no match

    lp : Large power threshold. If the maximum active power of p1 and p2 is over
    this value, then a percentage is used for matching, instead of t.

    m : The matching percentage used for large active power values.

    Returns
    -------

    match : Boolean for match (True) or no match (False)

    distance : Distance between power consumptions. It is L1 distance in the
    case of active_only=True or L2 distance in case active_only=False.

    """
    # TODO: Check and enforce signature shapes
    if p1[0] < 0.0 or p2[0] < 0.0:
        raise ValueError('Active power must be positive')
    if max((p1[0], p2[0])) >= lp:
        t_active = m * p2[0]
    else:
        t_active = t
    if max((np.fabs(p1[1]), np.fabs(p2[1]))) >= lp:
        t_reactive = m * p2[1]
    else:
        t_reactive = t
    T = np.fabs(np.array([t_active, t_reactive]))
    if active_only:
        d = np.fabs(p2[0] - p1[0])
        if d < T[0]:
            # Match
            return True, d
    else:
        d = np.linalg.norm(p2 - p1)
        if all(np.fabs(p2 - p1) < T):
            # Match
            return True, d
    return False, d


def power_curve_from_activations(appliances, start=None, end=None):
    """
    Create a power curve corresponding to the joint power consumption of a list
    of appliances.

    Parameters
    ----------

    appliances : List
    List containing eeris_nilm.appliance.Appliance instances. Warning: This
    function produces power consumption curves with 1 second period for the
    union of the appliance usage duration without size limitations. It is the
    caller's responsibility to ensure that no memory issues occur.

    start : String or None
    Start time for the power curve. The string must be in
    a format understood by pandas.Timestamp(). Only activations taking place
    after this time are considered. If None, the earliest start time of the data
    is used.

    end : String or None
    End time for the power curve. The string must be in
    a format understood by pandas.Timestamp(). Only activations taking place
    before this time are considered. If None, the earliest start time of the
    data is used.

    Returns
    -------

    curve : pandas.DataFrame
    Dataframe with timestamp index (1 second period) of active and reactive
    power consumption of the appliance.

    """
    # Determine the size of the return dataframe
    if start is None or end is None:
        est_start = None
        est_end = None
        for i in range(len(appliances)):
            app = appliances[i]
            app.activations.sort_values('start')
            ap_start = app.activations.loc[0, 'start']
            ap_end = app.activations['start'].iat[-1]
            if est_start is None or ap_start < est_start:
                est_start = ap_start
            if est_end is None or ap_end > est_end:
                est_end = ap_end
        if start is None and end is not None:
            start = est_start
            end = pd.Timestamp(end)
        elif start is not None and end is None:
            start = pd.Timestamp(start)
            end = est_end
        else:
            start = est_start
            end = est_end
    else:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
    idx = pd.date_range(start=start, end=end, freq='S')
    # ncol = appliances[0].signature.shape[1]
    ncol = 2  # Fixed number of columns.
    data = np.zeros(np.array([idx.shape[0], ncol]))
    power = pd.DataFrame(index=idx, data=data, columns=['active', 'reactive'])
    for i in range(len(appliances)):
        app = appliances[i]
        s = appliances[i].signature[0]
        for _, act in appliances[i].activations.iterrows():
            power.loc[act['start']:act['end']] += s
    return power


def remove_overlapping_matches(matches):
    """
    Remove overlapping matches. First, the function removes matches that
    include other matches of the same appliance (start1 < start2 and end1 >
    end2).

    Parameters
    ----------

    matches : pandas.DataFrame with columns 'start', 'end', as well as other
    columns with match signature.

    Returns
    -------

    out : pandas.DataFrame without matches that are supersets of other
    matches
    """
    out = matches.copy()
    # Find all segments that enclose this one
    for _, row in matches.iterrows():
        start = row['start']
        end = row['end']
        idx = out.index[(out['start'] < start) & (out['end'] > end)]
        out.drop(index=idx, inplace=True)
    return out
    # TODO: Merge partially overlapping segments?


def get_data_from_cenote_response(resp):
    """
    Convert response from cenote system to pandas.DataFrame that can be used by
    eeris_nilm.
    """
    if resp.status_code != 200:
        return None
    rd = json.loads(resp.text)
    data = pd.DataFrame(rd['results'])
    if data.shape[0] == 0:
        return None
    data = data.drop(['installationid', 'uuid', 'cenote$created_at',
                      'cenote$id'], axis=1)
    data['cenote$timestamp'] = pd.to_datetime(data['cenote$timestamp'],
                                              unit='ms', origin='unix')
    data = data.rename(columns={'cenote$timestamp': 'timestamp'})
    data = data.set_index('timestamp')
    data.index.name = None
    return data
