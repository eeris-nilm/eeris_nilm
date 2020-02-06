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
import logging


def get_segments(signal, mask, only_true=True):
    """
    Get the segments of a signal indicated by consecutive "True" values in a binary mask

    Parameters
    ----------
    signal : numpy.array
    1-D numpy array

    mask : numpy.array
    Boolean array with same shape as signal

    only_true: Boolean
    If it is True, only segments corresponding to True values of the original signal are
    returned. Otherwise, segments corresponding to both True and False values are
    returned.

    Returns
    -------
    out : list
    List of numpy.array elements, each containing a segment of the original signal.

    """
    # Exception?
    if signal.shape[0] != mask.shape[0]:
        logging.debug("Signal and mask shape do not match")
        return

    # Vectorized way to identify semgments
    segments = []
    idx = np.where(np.concatenate(([True], mask[:-1] != mask[1:], [True])))[0]
    for i in range(len(idx[:-1])):
        seg = signal[idx[i]:idx[i+1]]
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
    Normalize power with voltage measurements, if available. See also Hart's 1985 paper.

    Parameters
    ----------
    data: pandas.DataFrame
    Pandas dataframe with columns 'active', 'voltage' and, optionally, 'reactive'.

    Returns
    -------
    out: pandads.DataFrame. A normalized dataframe, where
    .. math::
    P_n(i) = P(i)\left\(\frac{V_0}{V(i)}\right\)^{1.5}\\
    Q_n(i) = Q(i)\left\(\frac{V_0}{V(i)}\right\)^{2.5}
    """
    # Normalization. Raise active power to 1.5 and reactive power to
    # 2.5. See Hart's 1985 paper for an explanation.
    #
    # Consider throwing exception here.
    if 'active' not in data.columns:
        logging.debug("No \'active\' column. Doing nothing.")
        return None

    r_data = data.copy()
    if 'voltage' in data.columns:
        r_data.loc[:, 'active'] = data['active'] * \
            np.power((nominal_voltage / data['voltage']), 1.5)
        if 'reactive' in data.columns:
            r_data.loc[:, 'reactive'] = data['reactive'] * \
                np.power((nominal_voltage / data['voltage']), 2.5)
    return r_data
