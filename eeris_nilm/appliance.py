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

from eeris_nilm import utils
import pandas as pd
import numpy as np
# from numpy import linalg as LA
import scipy.signal
import sklearn.cluster
import logging


# ##### Static methods outside class #######

def compare_power(a1, a2, t):
    """
    Helper function to see if two appliances are similar, by comparing all
    power consumption states states.

    Parameters
    ----------
    a1 : eeris_nilm.appliance.Appliance object

    a2 : eeris_nilm.appliance.Appliance object

    t : Float
    Beyond this threshold the devices are considered different (same value
    used for active and reactive power)

    Returns
    -------

    match : bool
    Appliances match (True) or not (False)
    """
    # TODO: This is greedy. Is it OK? Leave it like this for now
    s1 = np.copy(a1.signature)
    s2 = np.copy(a2.signature)
    if s1.shape[0] != s2.shape[0]:
        raise ValueError(("Appliances must have the same number of"
                          "states for full match"))
    matched1 = np.zeros((s1.shape[0], 1), dtype=bool)
    matched2 = np.zeros((s2.shape[0], 1), dtype=bool)
    for i in range(len(s1.shape[0])):
        best_match = -1
        distance = 1e10
        for j in range(len(s2.shape[0])):
            # Greedy approach
            if matched2[j]:
                continue
            p1 = a1.signature[i, :]
            p2 = a2.signature[j, :]
            match, d = utils.match_power(p1, p2, active_only=False)
            if d < distance:
                best_match = j
            if match:
                matched1[i] = True
        if matched1[i]:
            matched2[best_match] = True
    if all(matched1) and all(matched2):
        return True
    else:
        return False


def match_power_state(a1, a2, t=None, lp=None, m=None):
    """
    Helper function to see if the 'on' state of a two-state appliance is
    matched with some state of a multi-state appliance

    Parameters
    ----------
    a1 : eeris_nilm.appliance.Appliance object. Only the first state is
    considered (even if it has more than one)

    a2 : eeris_nilm.appliance.Appliance object

    t, lp, m : Arguments to be passed to utils.match_power (if provided)

    Returns
    -------

    match : bool
    True if some state matches
    distance : float
    Distance of closest state
    index : int
    Index of closest state (row of the signature matrix)
    """
    # TODO: This is greedy. Is it OK? Leave it like this for now
    s1 = np.copy(a1.signature)
    s2 = np.copy(a2.signature)
    matched = False
    distance = 1e10
    index = -1
    for i in range(s2.shape[0]):
        match, d = utils.match_power(s1[0, :], s2[i, :],
                                     active_only=False,
                                     t=t, lp=lp, m=m)
        if d < distance:
            distance = d
            index = i
            # This could be outside the first conditional (should be
            # equivalent)
            if match:
                matched = True
    return matched, distance, index


# TODO: This is deprecated, to be removed
def match_appliances_power(a_from, a_to, t=35.0, copy_activations=True):
    """
    Helper function to match between two dictionaries of appliances, using
    only their power consumption.

    Parameters
    ----------
    a_from : Dictionary of eeris_nilm.appliance.Appliance instances
    Objects that we need to map from

    a_to : Dictionary of eeris_nilm.appliance.Appliance instances
    Objects that we need to map to

    t : float
    Beyond this threshold the devices are definitely different

    copy_activations : bool
    Whether to copy the activations of the 'from' appliance to the 'to'
    appliance.

    Returns
    -------

    out : A dictionary of the form { appliance_id: appliance } where
    appliance is an eeris_nilm.appliance.Appliance object and appliance_id
    is the id of the appliance. This function maps the appliances in a_from
    to a_to i.e., adjusts the appliance_id for the appliances that are
    considered the same in a_from and a_to, keeping the ids of a_to. The
    dictionary also includes appliances that were not mapped (without
    changing their appliance_id).
    """
    # TODO: Works only for two-state appliances, assuming signature encodes
    # only the 'On' state
    #
    # TODO: This is a greedy implementation with many to one mapping. Is
    # this correct? Could an alternative strategy be better instead? To
    # support this, we keep the list of all candidates in the current
    # implementation.
    a = dict()
    mapping = dict()
    for k in a_from.keys():
        # Create the list of candidate matches for the k-th appliance
        candidates = []
        for l in a_to.keys():
            # Works only for two-state appliances. Remember that
            # signature[0] is a 1x2 vector with active and reactive power
            # (1st appliance state)
            match, d = utils.match_power(a_from[k].signature[0],
                                         a_to[l].signature[0],
                                         active_only=False, t=t)
            if match:
                candidates.append((l, d))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            # Simplest approach. Just get the minimum that is below
            # threshold t
            #
            # If we want to avoid mapping to an already mapped appliance,
            # then do this:
            # m = 0
            # while m < len(candidates) and candidates[m][0] in \
            # mapping.keys():
            # m += 1
            # if m < len(candidates):
            #     mapping[k] = candidates[m][0]
            #
            # For now we keep it simple and do this instead:
            mapping[k] = candidates[0][0]
    # Finally, perform the mapping. This loop assumes that keys in both
    # lists are unique (as is the case with appliances created in this
    # class).
    # TODO: Perform uniqueness checks!
    for k in a_from.keys():
        if k in mapping.keys():
            m = mapping[k]
            a[m] = a_to[m]
            a[m]._mapped = True
            if copy_activations:
                a[m].activations = a_from[k].activations.copy()
                a[m].last_returned_end_ts = a_from[k].last_returned_end_ts
        else:
            # Unmapped new appliances
            a[k] = a_from[k]
    return a


def _activations_overlap_pct(a1, a2, tol=5, n_limit=100):
    """
    Given two appliances with their activations, identify the percentage of
    matched activations. For a pair of activations to match, they need to
    have at most

    Parameters
    ----------
    a1: eeris_nilm.appliance.Appliance
    First appliance

    a2: eeris_nilm.appliance.Appliance
    Second appliance

    tol: Tolerance, in seconds for matching of start and end times.

    n_limit : int
    Consider the last n_limit activatations of each appliance for the matching
    (to speed things up). If None, then all values are considered


    Returns
    -------

    pct1: float
    Percentage of a1 activations that overlap with a2 activations

    pct2: float
    Percentage of a2 activations that overlap with a1 activations

    """
    # Make sure activations are sorted by their start time (in POSIX ts)
    df1 = a1.activations.sort_values(by=['start'])
    v1 = df1[['start', 'end']].values.astype('datetime64[s]').astype('int64')
    df2 = a2.activations.sort_values(by=['start'])
    v2 = df2[['start', 'end']].values.astype('datetime64[s]').astype('int64')
    if n_limit is not None:
        v1 = v1[-n_limit:, :]
        v2 = v2[-n_limit:, :]
    # Perform the matching.
    # TODO: Alternative implementation (to be considered, if needed): Create
    # two masks, subtract them and identify segments that are larger than
    # tol seconds as non-matches.
    idx1 = 0
    idx2 = 0
    matched = 0
    while idx1 < v1.shape[0]:
        start1 = v1[idx1, 0]
        end1 = v1[idx1, 1]
        while idx2 < v2.shape[0]:
            start2 = v2[idx2, 0]
            end2 = v2[idx2, 1]

            if start2 - start1 < -tol:
                idx2 += 1
                continue
            if abs(start2 - start1) < tol:
                if abs(end2 - end1) < tol:
                    # match
                    matched += 1
                idx2 += 1
                continue

            if start2 - start1 > tol:
                break
        idx1 += 1
    if v1.shape[0] > 0:
        pct1 = matched / v1.shape[0]
    else:
        pct1 = 0.0
    if v2.shape[0] > 0:
        pct2 = matched / v2.shape[0]
    else:
        pct2 = 0.0
    return (pct1, pct2)


def match_appliances(a_new, a_old, t=100.0, tol=5, copy_activations=True,
                     only_power=False, keep_old=False):
    """
    Helper function to match between two dictionaries of appliances, using
    both their power consumption and their activation timing.

    Parameters
    ----------
    a_new : Dictionary of eeris_nilm.appliance.Appliance instances
    Objects that we need to map from

    a_old : Dictionary of eeris_nilm.appliance.Appliance instances
    Objects that we need to map to

    t : float
    Beyond this threshold the devices are definitely different

    tol: float
    Seconds tolerance for activation matching

    copy_activations: bool
    Whether to copy the activations of the 'from' appliance to the 'to'
    appliance.

    only_power: bool
    Only consider the power consumption of the appliance (i.e., not
    activations) for matching

    keep_old: bool
    Whether to keep all the appliances in a_old (and add only those that have
    not been matched).

    Returns
    -------

    out : A dictionary of the form { appliance_id: appliance } where
    appliance is an eeris_nilm.appliance.Appliance object and appliance_id
    is the id of the appliance. This function maps the appliances in a_new
    to a_old i.e., adjusts the appliance_id for the appliances that are
    considered the same in a_new and a_old, keeping the ids of a_old. The
    dictionary also includes appliances that were not mapped (without
    changing their appliance_id).
    """
    if keep_old:
        a = a_old
    else:
        a = dict()

    mapping = dict()
    for k in a_new.keys():
        # Create the list of candidate matches for the k-th appliance
        candidates = []
        for l in a_old.keys():
            # Works only for two-state appliances. Remember that
            # signature[0] is a 1x2 vector with active and reactive power
            # (1st appliance state)
            match, d = utils.match_power(a_new[k].signature[0],
                                         a_old[l].signature[0],
                                         active_only=False, t=t)
            if match:
                if not only_power:
                    p_new, p_old = _activations_overlap_pct(a_new[k],
                                                            a_old[l],
                                                            tol=tol)
                    # Minus for the sorting afterwards
                    candidates.append((d, -p_old, l))
                else:
                    candidates.append((d, 0.0, l))
        if candidates:
            candidates.sort()
            # Simplistic: Just get the best match overall
            mapping[k] = candidates[0][2]

    # Perform the mapping. This loop assumes that keys in both
    # lists are unique (as is the case with appliances created in this
    # class).
    # TODO: Perform uniqueness checks!
    for k in a_new.keys():
        if k in mapping.keys():
            m = mapping[k]
            if not keep_old:
                a[m] = a_old[m]
            a[m]._mapped = True
            if copy_activations:
                a[m].activations = a_new[k].activations.copy()
                a[m].last_returned_end_ts = a_new[k].last_returned_end_ts
        else:
            # Unmapped new appliances
            a[k] = a_new[k]
    return a


# ##### Class definition #######

# TODO: What happens with variable consumption appliances?
# TODO: Code needs refactoring
# TODO: Make signature a property and enforce restrictions on shape
# TODO: Make unnecessary comparison functions hidden
class Appliance(object):
    """
    Unsupervised appliance model. Includes signatures, usage data and statistics
    as well as other data useful for identification through NILM.
    """
    MATCH_THRESHOLD = 35.0
    MAX_HISTORY_DAYS = 10000

    def __init__(self, appliance_id, name, category, signature=None,
                 nominal_voltage=230.0):
        self.appliance_id = appliance_id
        self.name = name
        self.category = category
        self.nominal_voltage = nominal_voltage
        self.num_states = 2  # Default is two-state appliance
        self.signature = signature  # Should have num_states-1 rows
        self.final = False  # We are allowed to modify signature
        self.verified = False  # Not sure it is actually a new appliance
        self.inactive = False  # Has it not been used for a long time?
        self.p_signature = signature  # Previous signature (for running average)
        # Should we keep data regarding activation of this applicance?
        self.store_activations = True
        # Helper variable to keep track of which activations have been provided
        # to external callers
        self.last_returned_end_ts = None
        # Helper to store when appliance started consuming in the current
        # cycle. Useful for live appliances.
        self.start_ts = None
        # Time segments of specific appliance states (corresponding to rows of
        # the signature matrix). The active power at that state is also
        # recorded.
        columns = ['start', 'end', 'active']
        self.activations = pd.DataFrame([], columns=columns)
        self._mapped = False  # For debugging purposes (see match_appliances)

    def append_activation(self, start, end, active):
        """
        Add an entry to the history of activations of this appliance.

        Parameters
        ----------
        start : pandas.Timestamp
        Start time for the activation

        end : pandas.Timestamp
        End time of the activation

        active : float
        Active power consumption of the state during the activation

        """
        df = pd.DataFrame(data={'start': start,
                                'end': end,
                                'active': active}, index=[0])
        self.activations = self.activations.append(df, ignore_index=True,
                                                   sort=True)

    def return_new_activations(self, update_ts=False):
        """
        Return activations that have been detected and have not been returned
        before.

        Parameters
        ----------
        update_ts : bool
        Whether to set the last update timestamp (if set, then subsequent calls
        will not return the activations that have been already returned)

        Returns
        -------
        out : pandas.DataFrame instance with 'start', 'end',
        'active' columns.
        """
        if self.last_returned_end_ts is None:
            activations = self.activations
        else:
            idx = self.activations['end'] > self.last_returned_end_ts
            activations = self.activations.loc[idx, :]
        activations = activations.sort_values('end', ascending=True,
                                              ignore_index=True)
        if update_ts:
            self.last_returned_end_ts = activations[-1]['end']
        return activations

    def reset_activations_track(self):
        """
        Reset the tracked activations
        """
        self.last_returned_end_ts = None

    def append_signature(self, signature):
        """
        Add one or more [active, reactive] rows to appliance signature
        """
        if signature.shape[1] < 2:
            raise ValueError(("Signature must have at least active and"
                              "reactive power columns"))
        if self.signature is None:
            self.signature = signature
            return
        if self.signature.shape[1] != signature.shape[1]:
            raise ValueError(("Signature shape incompatible"
                              "with appliance signature"))
        self.signature = np.append(self.signature, signature, axis=0)

    def update_appliance_live(self):
        """
        Update live appliance, by updating a running average of its signature.
        """
        if not self.final:
            self.final = True
        # Running average. TODO: Introduce checks for highly abnormal new value?
        if self.p_signature is not None:
            self.signature = 0.9 * self.p_signature + 0.1 * self.signature
        self.p_signature = self.signature

    def signature_from_data_steady_states(self, data):
        """
        Given active (and, possibly, reactive) recorded data from an appliance
        (e.g., data from a smart plug), this function computes the appliance
        signature based on the measured consumption steady states. If a
        'voltage' column is available, it is used for data normalization
        purposes.

        Parameters
        ----------
        data : Pandas dataframe with appliance recordings. Expect columns
        'active' and, optionally, 'reactive' and 'voltage'.

        """
        # TODO: This function works with steady-states. This may not work in
        # variable consumption appliances. Also, Hart's algorithm works on
        # matched edges. If this approach is not effective then we can apply
        # Hart's algorithm in each appliance separately and perform matching on
        # the edges as usual.

        # TODO: Exception?
        if 'active' not in data.columns:
            s = ("Expect \'active\' and, optionally,",
                 "\'reactive\' and \'voltage\' columns")
            raise ValueError(s)

        # Normalize, if voltage is available.
        data_n = utils.get_normalized_data(
            data, nominal_voltage=self.nominal_voltage)
        # Select only active and reactive columns
        if 'reactive' in data.columns:
            data_n = data_n[['active', 'reactive']]
        else:
            data_n = data_n[['active']]
        # Pre-process data to a constant sampling rate, and fill-in missing
        # data.
        data_n = utils.preprocess_data(data_n)
        # Work with numpy data from now on.
        npdata = data_n.values
        # Apply a 5-th order derivative filter to detect edges
        sobel = np.array([-2, -1, 0, 1, 2])
        # Apply an edge threshold
        threshold = 5.0  # TODO: Make this a parameter
        mask = scipy.convolve(npdata[:, 0], sobel, mode='same') < threshold
        segments = utils.get_segments(npdata, mask)
        # Get the average value of each segment
        seg_values = np.array([np.mean(seg) for seg in segments])
        # Make sure shape is appropriate for dbscan
        if len(seg_values.shape) == 1:
            if seg_values.shape[0] > 1:
                # Only this should evaluate to true
                seg_values = seg_values.reshape(-1, 1)
            else:
                seg_values = seg_values.reshape(1, -1)
        # Subsample large sets, otherwise clustering takes forever. Based on how
        # seg_values is calculated, we only need to check number of rows.
        if seg_values.shape[0] > 20000:
            aidx = np.arange(seg_values.shape[0])
            np.random.shuffle(aidx)
            seg_values = seg_values[aidx[:20000]]
        # Cluster the values.
        # TODO: Decide what to do with hardcoded cluster parameters
        d = sklearn.cluster.DBSCAN(eps=35.0, min_samples=3, metric='euclidean',
                                   metric_params=None, algorithm='auto')
        d.fit(seg_values)
        # TODO: Negative values are outliers. Do we need those? What if they are
        # large?
        u_labels = np.unique(d.labels_[d.labels_ >= 0])
        # At least have active and reactive power in signature (even if zero).
        if seg_values.shape[1] == 1:
            centers = np.zeros((u_labels.shape[0], 2))
        else:
            centers = np.zeros((u_labels.shape[0], seg_values.shape[1]))
        idx = 0
        n_skipped = 0
        for l in u_labels:
            c = np.mean(seg_values[d.labels_ == l, :], axis=0)
            # Only active power is available
            if len(c.shape) == 1:
                c = np.array([c[0], 0.0])
            # Low active power (at most 10 watts) is the 'off' state, which is
            # not included.
            if c[0] > 10.0:
                centers[idx, :] = c
                idx += 1
            else:
                n_skipped += 1
        if n_skipped > 1:
            logging.debug(('Skipped %d states during appliance'
                           'signature estimation' % (n_skipped)))
        if idx >= 1:
            self.signature = centers[:idx, :]
        # No longer includes implicit 'off' state
        self.num_states = centers.shape[0]
