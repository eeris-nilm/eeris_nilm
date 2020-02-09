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
import logging

# TODO: What happens with variable consumption appliances?

class Appliance():
    """
    Unsupervised appliance model. Includes signatures, usage data and statistics
    as well as other data useful for identification through NILM.
    """

    def __init__(self, appliance_id, name, category, signature=None, nominal_voltage=230.0):
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
        # TODO: Exception?
        if 'active' not in data.columns:
            logging.debug(
                'Expect \'active\' and, optionally, \'reactive\' columns')
            return

        # Normalize, if voltage is available.
        data_n = utils.get_normalized_data(
            data, nominal_voltage=self.nominal_voltage)
        # Select only active and reactive columns
        if 'reactive' in data.columns:
            data_n = data_n[['active', 'reactive']]
        else:
            data_n = data_n[['active']]
        # Pre-process data to a constant sampling rate, and fill-in missing data.
        data_n = data_n.sort_index()
        data_n = data_n.reset_index()
        data_n = data_n.drop_duplicates(subset='index', keep='last')
        data_n = data_n.set_index('index')
        data_n = data_n.fillna(0)
        data_n = data_n.asfreq('1S', method='pad')

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
        # TODO: Negative values are outliers. Do we need those? What if they are large?
        u_labels = np.unique(d.labels_[d.labels_ >= 0])
        centers = np.zeros((u_labels.shape[0], seg_values.shape[1]))
        for l in u_labels:
            centers[l] = np.mean(seg_values[d.labels_ == l, :], axis=0)
        self.signature = centers
        self.num_states = centers.shape[0]
