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

from eeris_nilm import utils
import pandas as pd
import numpy as np
from numpy import linalg as LA
import scipy.signal
import sklearn.cluster


# TODO: What happens with variable consumption appliances?
class Appliance(object):
    """
    Unsupervised appliance model. Includes signatures, usage data and statistics
    as well as other data useful for identification through NILM.
    """

    def __init__(self, appliance_id, name, category, signature=None,
                 nominal_voltage=230.0):
        self.appliance_id = appliance_id
        self.name = name
        self.category = category
        self.nominal_voltage = nominal_voltage
        self.num_states = 2  # Default is two-state appliance
        self.signature = signature
        self.final = False  # We are allowed to modify signature
        self.verified = False  # Not sure it is actually a new appliance
        self.inactive = False  # Has it not been used for a long time?
        self.p_signature = signature  # Previous signature (for running average)
        # Should we keep data regarding activation of this applicance?
        self.store_activations = True
        self.activations = pd.DataFrame([], columns=['start', 'end'])

    def distance(app1, app2):
        """
        Function defining the distance (or dissimilarity) between two appliances

        Parameters
        ----------
        app1: First Appliance object

        app2: Second Appliance object

        Returns
        -------
        out: Distance between the appliances

        """
        return LA.norm(app1.signature - app2.signature)

    def update_appliance_live(self):
        """
        Update appliance
        """
        if not self.final:
            self.final = True
        # Running average. TODO: Introduce checks for highly abnormal new value?
        if self.p_signature is not None:
            self.signature = 0.9 * self.p_signature + 0.1 * self.signature
        self.p_signature = self.signature

    def signature_from_data(self, data):
        """
        Given active (and, possibly, reactive) recorded data from an appliance (e.g., data
        from a smart plug), this function computes the appliance signature. If a 'voltage'
        column is available, it is used for data normalization purposes. Segmentation to
        determine consumption levels is only done on active power data.

        Parameters
        ----------
        data : Pandas dataframe with appliance recordings. Expect columns
        'active' and, optionally, 'reactive' and 'voltage'.

        """
        # TODO: This function works with steady-states. This may not work in variable
        # consumption appliances. Also, Hart's algorithm works on matched edges. If this
        # approach is not effective then we can apply Hart's algorithm in each appliance
        # separately and perform matching on the edges as usual.

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
        # Pre-process data to a constant sampling rate, and fill-in missing data.
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
        seg_values = np.array([np.mean(s) for s in segments])
        # Make sure shape is appropriate for dbscan
        if len(seg_values.shape) == 1:
            if seg_values.shape[0] > 1:
                seg_values = seg_values.reshape(-1, 1)
            else:
                seg_values = seg_values.reshape(1, -1)

        # Cluster the values.
        # TODO: Decide what to do with hardcoded cluster parameters
        d = sklearn.cluster.DBSCAN(eps=30, min_samples=3, metric='euclidean',
                                   metric_params=None, algorithm='auto')
        d.fit(seg_values)
        # TODO: Negative values are outliers. Do we need those? What if they are
        # large?
        u_labels = np.unique(d.labels_[d.labels_ >= 0])
        centers = np.zeros((u_labels.shape[0], seg_values.shape[1]))
        for l in u_labels:
            centers[l] = np.mean(seg_values[d.labels_ == l, :], axis=0)
        self.signature = centers
        self.num_states = centers.shape[0]

    def match_appliances(a_from, a_to, t=35.0):
        """
        Helper function to match between two dictionaries of appliances.

        Parameters
        ----------
        a_from : Dictionary of eeris_nilm.appliance.Appliance objects that we
        need to map from

        a_to : Dictionary of eeris_nilm.appliance.Appliance objects that we need
        to map to

        t : Beyond this threshold the devices are considered different

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
                match, d = utils.match_power(a_from[k], a_to[l],
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
            else:
                # Unmapped new appliances
                a[k] = a_from[k]
        return a
