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
from eeris_nilm import utils
from eeris_nilm import appliance
import sklearn.cluster
import sklearn.metrics.pairwise
import threading
import logging
import datetime
import time
import copy
import bson

# TODO: Background has been computed on normalized data and may have
# discrepancies from the actual background consumption (as measured through the
# meter)
# TODO: Breakdown into smaller modules (incl. subclassing and separation of
# "live" and retrospective analysis)
# TODO: Check for inactive appliances


class LiveHart(object):
    """
    Improved implementation of Hart's NILM algorithm
    supporting real-time feedback through a "live" monitoring mechanism.
    """
    # TODO:
    # - remove the class variables that are not needed
    # - Some of the 'constant' variables should be model parameters
    # - all "live" data should be part of a different class to simplify things a
    # little

    # ID for particular algorithm version
    VERSION = "0.1"

    # Some of the variables below could be parameters
    BUFFER_SIZE_SECONDS = 24 * 3600

    # Limiters/thresholds
    MAX_WINDOW_DAYS = 100
    MAX_NUM_STATES = 1000
    MAX_DISPLAY_SECONDS = 5 * 3600
    STEADY_THRESHOLD = 15
    SIGNIFICANT_EDGE = 50
    STEADY_SAMPLES_NUM = 5
    MATCH_THRESHOLD = 35
    MAX_MATCH_THRESHOLD_DAYS = 2
    EDGES_CLEAN_HOURS = 6
    STEADY_CLEAN_DAYS = 15
    MATCHES_CLEAN_DAYS = 365   # Unused for now
    OVERESTIMATION_SECONDS = 10

    # For clustering
    # TODO: Check the clustering values
    CLUSTERING_METHOD = "mean_shift"  # "dbscan" or "mean_shift"
    CLUSTER_STEP_HOURS = 1  # Cluster update frequency, in hours
    CLUSTER_DATA_DAYS = 30 * 3  # Use last 3 months for clustering
    MIN_EDGES_STATIC_CLUSTERING = 5  # DBSCAN parameter
    LARGE_POWER = 1e6  # Large power consumption
    BACKGROUND_UPDATE_DAYS = 15  # Use past days for background estimation
    BACKGROUND_UPDATE_PERIOD_HOURS = 1

    def __init__(self, installation_id, nominal_voltage=230.0,
                 batch_mode=False, store_live=False):
        # Almost all variables are needed as class members, to support streaming
        # support.

        # Power system parameters
        self.nominal_voltage = nominal_voltage
        # Operate in batch mode? (Does not clean up edges etc)
        self.batch_mode = batch_mode

        # Running state variables

        # Are we on transition?
        self.on_transition = False
        # Current edge estimate
        self.running_edge_estimate = np.array([0.0, 0.0])
        # How many steady samples
        self._steady_count = 0
        # How many samples in current edge
        self._edge_count = 0
        # Previous steady state
        self._previous_steady_power = np.array([0.0, 0.0])
        # Dynamic steady power estimate
        self.running_avg_power = np.array([0.0, 0.0])
        # Timestamp of last processed sample
        self.last_processed_ts = None
        # To save some computations if no edge was detected
        self._edge_detected = False

        # Data variables

        # Data passed for processing
        self._data_orig = None
        # Data after preprocessing and normalization
        self._data = None
        # Data buffer
        self._buffer = None
        # How many samples have been processed (in 1Hz rate, after resampling)
        self._samples_count = 0
        # Helper variable, index of first unprocessed sample in burffer
        self._idx = None
        # Helper variables for visualization (edges, matched devices)
        self._yest = np.array([], dtype='float64')
        self._last_visualized_ts = None
        self._online_edge_ts = None
        self._ymatch = None
        self._ymatch_live = None  # Not used (TODO?)
        # Installation id (is this necessary?)
        self.installation_id = installation_id
        # List of states and transitions detected so far.
        self._steady_states = pd.DataFrame([],
                                           columns=['start', 'end', 'active',
                                                    'reactive'])
        self._edges = pd.DataFrame([], columns=['start', 'end', 'active',
                                                'reactive', 'mark'])
        # Matched devices.
        self._matches = pd.DataFrame([], columns=['start', 'end', 'active',
                                                  'reactive'])
        self._clean_matches = False

        # Timestamps for keeping track of the data processed, for edges, steady
        # states and clusters
        self._start_ts = None
        self._last_clustering_ts = None
        self._edge_start_ts = None
        self._edge_end_ts = None
        self._steady_start_ts = None
        self._steady_end_ts = None
        # For online edge detection
        self._online_edge_detected = False
        self._online_edge = np.array([0.0, 0.0])
        # For cluster labels
        self._max_cluster_label = 0

        # List of live appliances
        self.live = []
        # Current live appliance id.
        # NOTE: bson object ids are not necessary here. They are used due to
        # integration requirements by other eeRIS modules.
        self._appliance_display_id = 0
        # Dictionaries of known appliances
        self.appliances = {}
        self.appliances_live = {}
        # Store a record of all live events for evaluation purposes
        self._store_live = store_live
        self.live_history = pd.DataFrame([], columns=['start', 'end', 'name',
                                                      'active', 'reactive'])
        # Variable to trigger potential applicance naming notifications to the
        # end-users. It stores the detected appliances that were activated for
        # naming
        self.detected_appliance = None

        # Other variables - needed for sanity checks
        self.background_active = self.LARGE_POWER
        self._background_last_update = None
        self._count_bg_overestimation = 0
        self.residual_live = np.array([0.0, 0.0])
        self._count_res_overestimation = 0

        # Variables for handling threads. For now, just a lock.
        self._clustering_thread = None
        self._lock = threading.Lock()

        # Installation statistics
        self._min_active = 0.0
        self._max_active = 0.0

        logging.debug("LiveHart object initialized.")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data variable.
        """
        if data.shape[0] == 0:
            raise ValueError('Empty dataframe')
        # Check time difference.
        duration = (data.index[-1] - data.index[0]).days
        if duration > self.MAX_WINDOW_DAYS:
            # Do not process the window, it's too long.
            raise ValueError('Data duration too long')
        self._data_orig = data.dropna()

    def _preprocess(self):
        """
        Data preprocessing steps. It also updates a sliding window of
        BUFFER_SIZE_SECONDS of data. Current version resamples to 1Hz sampling
        frequency.
        """
        if self._buffer is None:
            # Buffer initialization
            self._buffer = \
                utils.get_normalized_data(self._data_orig,
                                          nominal_voltage=self.nominal_voltage)
            assert self._start_ts is None  # Just making sure
            self._start_ts = self._data_orig.index[0]
        else:
            # Data concerning past dates update the buffer
            self._buffer = self._buffer.append(
                utils.get_normalized_data(self._data_orig,
                                          nominal_voltage=self.nominal_voltage),
                sort=True)
        # Data pre-processing (remove duplicates, resample to 1s)
        self._buffer = utils.preprocess_data(self._buffer)
        # Make sure that data is left in the buffer (can happen during startup)
        if self._buffer.empty:
            return
        # Keep only the last BUFFER_SIZE_SECONDS of the buffer, if batch mode is
        # disabled. Should this be done before preprocesssing?
        if not self.batch_mode:
            start_ts = self._buffer.index[-1] - \
                pd.offsets.Second(self.BUFFER_SIZE_SECONDS - 1)
            self._buffer = self._buffer.loc[self._buffer.index >= start_ts]
        if self.last_processed_ts is None:
            # We're just starting
            self._data = self._buffer
            self._idx = self._buffer.index[0]
            self._steady_start_ts = self._idx
        elif self.last_processed_ts < self._buffer.index[0]:
            # After a big gap, reset everything
            self.last_processed_ts = self._buffer.index[0]
            self._idx = self.last_processed_ts
            self._data = self._buffer
            self._reset()
        else:
            # Normal operation: Just get the data that has not been previously
            # been processed
            self._idx = self.last_processed_ts + 1 * self._buffer.index.freq
            self._data = self._buffer.loc[self._idx:]
        # TODO: Handle N/As and zero voltage.
        # TODO: Unit tests with all the unusual cases

        # Update ymatch (auxilliary timeseries with only matched edges)
        if self._ymatch is None:
            self._ymatch = pd.DataFrame(np.zeros([self._buffer.shape[0], 2]),
                                        index=self._buffer.index,
                                        columns=['active', 'reactive'])
        else:
            d = pd.DataFrame(np.zeros([self._data.shape[0], 2]),
                             index=self._data.index,
                             columns=['active', 'reactive'])
            self._ymatch = self._ymatch.append(d, sort=True)
            if not self.batch_mode:
                start_ts = self._ymatch.index[-1] - \
                    pd.offsets.Second(
                        self.MAX_MATCH_THRESHOLD_DAYS*3600*24 - 1)
                self._ymatch = self._ymatch.loc[self._ymatch.index >= start_ts]

    def _reset(self):
        """
        Reset model parameters. Can be useful after a big gap in the data
        collection.
        """
        self.on_transition = False
        self.running_edge_estimate = np.array([0.0, 0.0])
        self._steady_count = 0
        self._edge_count = 0
        self._previous_steady_power = np.array([0.0, 0.0])
        self.running_avg_power = np.array([0.0, 0.0])
        self._online_edge_detected = False
        self._online_edge = np.array([0.0, 0.0])
        # TODO Should we have this?
        # for a in self.live:
        #     start_ts = a.start_ts
        #     end_ts = self.last_processed_ts
        #     active = a.signature[0][0]
        #     a.append_activation(start_ts, end_ts, active)
        self.appliances_live = self.appliances.copy()
        self.live = []

    def _detect_edges_hart(self):
        """
        Edge detector, based on Hart's algorithm.
        """
        self._edge_detected = False
        self._online_edge_detected = False
        self._online_edge = np.array([0.0, 0.0])
        if self.last_processed_ts is None:
            data = self._buffer[['active', 'reactive']].values
            prev = data[0, :]
        else:
            tmp_df = self._buffer[['active', 'reactive']]
            prev = tmp_df.loc[self.last_processed_ts].values
            data = tmp_df.loc[self._idx:].values
        # These are helper variables, to have a single np.concatenate/vstack at
        # the end
        edge_list = [self._edges]
        steady_list = [self._steady_states]
        for i in range(data.shape[0]):
            current_ts = self._idx + i * self._buffer.index.freq
            diff = data[i, :] - prev
            prev = data[i, :]
            if any(np.fabs(diff) > self.STEADY_THRESHOLD):
                if not self.on_transition:
                    # Starting transition
                    # Do not register previous edge if it started from 0 (it may
                    # be due to missing data).
                    if any(self._previous_steady_power > np.finfo(float).eps):
                        previous_edge = self.running_avg_power - \
                            self._previous_steady_power
                        if any(np.fabs(previous_edge) > self.SIGNIFICANT_EDGE):
                            edge_df = \
                                pd.DataFrame(data={'start': self._edge_start_ts,
                                                   'end': self._edge_end_ts,
                                                   'active': previous_edge[0],
                                                   'reactive': previous_edge[1],
                                                   'mark': False},
                                             index=[0])
                            edge_list.append(edge_df)
                            self._edge_detected = True
                    self._steady_end_ts = current_ts
                    steady_df = \
                        pd.DataFrame(data={'start': self._steady_start_ts,
                                           'end': self._steady_end_ts,
                                           'active': self.running_avg_power[0],
                                           'reactive':
                                           self.running_avg_power[1]},
                                     index=[0])
                    steady_list.append(steady_df)
                    self._previous_steady_power = self.running_avg_power
                    self._edge_start_ts = current_ts
                    self.running_avg_power = np.array([0.0, 0.0])
                    # Do we need these two?
                    self._steady_count = 0
                    self._edge_count += 1
                    self.running_edge_estimate = diff
                    self.on_transition = True
                else:
                    # Either the transition continues, or it is the start of a
                    # steady period.
                    self._edge_count += 1
                    self.running_edge_estimate += diff
                    self.running_avg_power = data[i, :]
                    self._steady_count = 1
                    self._steady_start_ts = current_ts
                    self._edge_end_ts = current_ts
            else:
                # Update running average
                self.running_avg_power *= self._steady_count / \
                    (self._steady_count + 1.0)
                self.running_avg_power += \
                    1.0 / (self._steady_count + 1.0) * data[i, :]
                self._steady_count += 1
                if self.on_transition:
                    # We are in the process of finishing a transition
                    self.running_edge_estimate += diff
                    if self._steady_count >= self.STEADY_SAMPLES_NUM:
                        self._edge_end_ts = current_ts
                        self._steady_start_ts = current_ts
                        self.on_transition = False
                        self._online_edge_detected = True
                        # self._online_edge = self.running_edge_estimate
                        self._online_edge = self.running_avg_power - \
                            self._previous_steady_power
                        self._online_edge_ts = self._edge_start_ts
                        # self._online_edge_ts = self._edge_end_ts
                        self._edge_count = 0
            self._samples_count += 1
        # Update lists
        self._edges = pd.concat(edge_list, ignore_index=True)
        self._steady_states = pd.concat(steady_list, ignore_index=True,
                                        sort=True)
        # Update last processed
        self.last_processed_ts = self._buffer.index[-1]

    def _sync_appliances(self, a_new, mapping):
        """
        Map the newly detected appliances to the old ones.
        """
        a = dict()
        for k in a_new.keys():
            if k in mapping.keys():
                m = mapping[k]
                a[m] = self.appliances[m]
                # TODO: Should we copy the new signature (this could lead to
                # "drifting" of the signature over time)?
                a[m].signature = a_new[k].signature.copy()
                a[m].activations = a_new[k].activations.copy()
                # We want to keep the 'old' last_returned_end_ts
            else:
                # Unmapped new appliances
                a[k] = a_new[k]
                # Update names of new appliances to avoid confusion with old
                # names (e.g., both new and old appliance named 'Cluster 1')
                a[k].name = 'Cluster %d' % (self._max_cluster_label + 1)
                self._max_cluster_label += 1
        self.appliances = a

    def _sync_appliances_live(self, mapping):
        """
        Map the live appliances to the ones detected by clustering (some design
        choices made here).
        """
        a = self.appliances.copy()
        for k in self.appliances_live.keys():
            if k in mapping.keys():
                m = mapping[k]
                # TODO: .live signifies whether an appliance has been detected
                # through the clustering procedure (even if it's a copy of an
                # appliance from self.appliances). We therefore don't set it to
                # true here. TODO: Verify this.
                # a[m].live = True
                a[m].start_ts = self.appliances_live[k].start_ts
                # If live appliance is named, then copy name
                if self.appliances_live[k].verified:
                    # If both the live and the clustered appliance is verified,
                    # then do nothing
                    # TODO: Introduce a voting mechanism?
                    if not a[m].verified:
                        a[m].name = self.appliances_live[k].name
                        a[m].category = self.appliances_live[k].category
                        a[m].verified = True
                    else:
                        logging.debug('Appliance %s already has a name (%s)'
                                      ' and category (%s), not updating with'
                                      ' name (%s) and category (%s)' %
                                      (m, a[m].name, a[m].category,
                                       self.appliances_live[k].name,
                                       self.appliances_live[k].category))

                # In case the appliance is operating, replace with new live.
                try:
                    idx = self.live.index(self.appliances_live[k])
                    self.live[idx] = a[m]
                except ValueError:
                    continue
            else:
                # Non-mapped appliances remain the same, so self.live shouldn't
                # be affected.  a[k] = self.appliances_live[k]
                #
                # DEBUG: Test to see what happens if we throw away all live appliances after
                # each clustering step.
                self._appliance_display_id = 0
        self.appliances_live = a

    def _sync_appliances_live_copy(self):
        """
        The live appliances are just a copy of the clustered appliances.
        """
        a = self.appliances.copy()
        # See comment on sync_appliances live
        # for k in a.keys():
        #    a[k].live = True
        self.appliances_live = a

    def _sync_appliances_live_1_DEPRECATED(self, mapping):
        """
        Map the live appliances to the ones detected by clustering (some design
        choices made here).
        """
        # TODO: This function is more complicated than it needs to be because
        # different options need to be tested for the matching.

        # TODO: Allow many live appliances to map to the same cluster. Also,
        # consider copying active appliances to the live set after all.
        a = dict()
        mapped_keys = []
        a_new = self.appliances
        for k in self.appliances_live.keys():
            if k in mapping.keys():
                m = mapping[k]
                # Has the appliance already been mapped?
                # TODO: This is a hack. Better solutions? Is this an edge case?
                if m in mapped_keys:
                    logging.debug(("Appliance %s (%s) already mapped, ignoring"
                                   "mapping of %s") %
                                  (a_new[m].name, m,
                                   self.appliances_live[k].name),
                                  stack_info=True)
                    continue
                # TODO: Simpler solution, if we decide to just keep the live
                # appliance. Just change the name of self.appliances_live[k] and
                # do nothing else.
                # TODO: We keep the live signature. Is this correct?
                a[k] = copy.deepcopy(self.appliances_live[k])
                a[k].live = True
                a[k].name = a_new[m].name
                mapped_keys.append(m)
                # In case the appliance is operating, replace with new live.
                try:
                    idx = self.live.index(self.appliances_live[k])
                    self.live[idx] = a[k]
                except ValueError:
                    continue
            else:
                # Non-mapped appliances remain the same, so self.live shouldn't
                # be affected.
                a[k] = self.appliances_live[k]
        # TODO: Is it correct to copy non-matched, clustered appliances to live?
        # FOR NOW WE DONT DO IT.
        # Interesting side-effect. A cluster is not mapped. But then it is
        # copied. In the next iteration it is mapped (they should match) and for
        # this reason it is removed (since it is included in del_keys). This may
        # be a source of bugs, so we keep it simple for now and do not copy
        # non-matched, clustered appliances.
        #
        # for m in a_new.keys():
        #     if m not in a:
        #         a[m] = copy.deepcopy(a_new[m])
        #         a[m].live = True
        self.appliances_live = a

    def _static_cluster(self, method="dbscan"):
        """
        Clustering step of Hart's method. Here it is implemented as a static
        clustering step that runs periodically, mapping previous devices to the
        new device names. In this implementation we use "static" clustering
        (apply clustering periodically) instead of a dynamic cluster update.

        This function is designed to run as a thread.

        Parameters
        ----------

        method : string with possible values "dbscan" or "mean_shift"
        """
        # TODO: Experiment on the clustering parameters.
        # NOTE: Normalize matches in the 0-1 range, so that difference is
        # percentage! This will perhaps allow better matches.
        # NOTE: Possibly start with high detail and then merge similar
        # clusters?
        # NOTE: Explore use of reactive power for matching and clustering

        # Do not wait forever
        with self._lock:
            # Select matched edges to use for clustering
            matches = self._matches.copy()
            matches = matches[['start', 'end', 'active', 'reactive']]
            clustering_start_ts = self.last_processed_ts
            if len(matches) < self.MIN_EDGES_STATIC_CLUSTERING:
                # Set the timestamp, to avoid continuous attempts for clustering
                # Set timestamp
                self._last_clustering_ts = clustering_start_ts
                return
            if not self.batch_mode:
                start_ts = matches['start'].iloc[-1] - \
                    pd.offsets.Day(self.CLUSTER_DATA_DAYS)
                matches = matches.loc[matches['start'] > start_ts]
            matches1 = matches.copy()
            matches = matches[['active', 'reactive']].values
        debug_t_start = datetime.datetime.now()
        logging.debug('Initiating static clustering at %s' % debug_t_start)
        if method == "dbscan":
            # Apply DBSCAN.
            d = sklearn.cluster.DBSCAN(eps=self.MATCH_THRESHOLD, min_samples=3,
                                       metric='euclidean', metric_params=None,
                                       algorithm='auto', leaf_size=30)
            d.fit(matches)
            # DBSCAN only: How many clusters are there? Can we derive "centers"?
            labels = d.labels_
            u_labels = np.unique(d.labels_[d.labels_ >= 0])
            centers = np.zeros((u_labels.shape[0], matches.shape[1]))
            for l in u_labels:
                centers[l] = np.mean(matches[d.labels_ == l, :], axis=0)
        elif method == "mean_shift":
            # TODO: Normalize matches in the 0-1 range, so that difference is
            # percentage! This will perhaps allow better matching behavior.
            # Degrade the matching resolution a bit.
            bandwidth = self.MATCH_THRESHOLD
            centers, labels = sklearn.cluster.mean_shift(matches,
                                                         bandwidth=bandwidth,
                                                         cluster_all=False)
            u_labels = np.unique(labels[labels >= 0])
        else:
            raise ValueError(("Unrecognized clustering method %s. Possible "
                              "values are \"dbscan\" and \"mean-shift\"") %
                             method)
        # We need to make sure that devices that were already detected
        # previously keep the same name.
        # First build a temporary list of appliances
        # TODO: What about outliers with large consumption? For now we just
        # ignore them.
        appliances = dict()
        for l in u_labels:
            name = 'Cluster %d' % (l)
            # TODO: Heuristics for determining appliance category
            category = 'unknown'
            signature = centers[l, :].reshape(-1, 1).T
            appliance_id = str(bson.objectid.ObjectId())
            a = appliance.Appliance(appliance_id, name, category,
                                    signature=signature)
            ml = matches1.iloc[labels == l, :]
            # Remove overlapping matches. This is NOT included in Hart's
            # original algorithm, but seems to help.
            ml = utils.remove_overlapping_matches(ml)
            a.activations = a.activations.append(ml[['start', 'end', 'active']],
                                                 ignore_index=True, sort=True)
            appliances[a.appliance_id] = a
        debug_t_end = datetime.datetime.now()
        debug_t_diff = (debug_t_end - debug_t_start)
        logging.debug('Finished static clustering at %s' % (debug_t_end))
        logging.debug('Total clustering time: %s seconds' %
                      (debug_t_diff.seconds))
        logging.debug('Clustering complete. Appliances before mapping:')
        for _, a in appliances.items():
            logging.debug("------------------------\n"
                          "ID: %s\n"
                          "Signature: %s\n"
                          "Name: %s\n"
                          "Category: %s\n"
                          % (a.appliance_id, a.signature, a.name,
                             a.category))
        logging.debug('Old appliances:')
        for _, a in self.appliances.items():
            logging.debug("------------------------\n"
                          "ID: %s\n"
                          "Signature: %s\n"
                          "Name: %s\n"
                          "Category: %s\n"
                          % (a.appliance_id, a.signature, a.name,
                             a.category))
        with self._lock:
            if not self.appliances:
                # First time we detect appliances
                self.appliances = appliances
                self._max_cluster_label = np.max(u_labels)
            else:
                # Map to previous.
                mapping = appliance.appliance_mapping(appliances,
                                                      self.appliances)
                self._sync_appliances(appliances, mapping)

            # Sync live appliances.
            # Option 1: Just copy the clusters.
            # self._sync_appliances_live_copy()

            # Option 2: Map and sync. Only use power for live mapping?
            mapping = appliance.appliance_mapping(self.appliances_live,
                                                  self.appliances,
                                                  t=2*self.MATCH_THRESHOLD,
                                                  only_power=True)
            self._sync_appliances_live(mapping)

            # For debugging
            logging.debug('Clustering complete. Current list of appliances:')
            for _, a in self.appliances.items():
                logging.debug("------------------------\n"
                              "ID: %s\n"
                              "Signature: %s\n"
                              "Name: %s\n"
                              "Category: %s\n"
                              % (a.appliance_id, a.signature, a.name,
                                 a.category))

            # Set timestamp
            self._last_clustering_ts = clustering_start_ts

    def _clean_buffers(self):
        """
        Clean-up edges, steady states and match buffers. This removes matched
        edges from the buffer, but may also remove edges that have remained in
        the buffer for a very long time, perform other sanity checks etc. It
        also removes old steady states and matches.
        """
        # Clean marked edges
        self._edges.drop(self._edges.loc[self._edges['mark']].index,
                         inplace=True)
        # If batch mode is enabled, there's nothing else left to do.
        if self.batch_mode:
            return
        # Otherwise clean-up old edges and steady states.
        # Clean edges that are too far back in time
        droplist = []
        for idx, e in self._edges.iterrows():
            td = (self.last_processed_ts - e['end']).total_seconds() / 3600.0

            if td > self.EDGES_CLEAN_HOURS:
                droplist.append(idx)
            else:
                # Edge times are sorted
                break

        self._edges.drop(droplist, inplace=True)

        # Clean steady states that are too far back in time
        droplist = []
        for idx, s in self._steady_states.iterrows():
            if (self.last_processed_ts - s['end']).days > \
               self.STEADY_CLEAN_DAYS:
                droplist.append(idx)
            else:
                # Steady state times are sorted
                break
        self._steady_states.drop(droplist, inplace=True)

        # Clean match buffers (if flag _clean_matches is set to True)
        if self._clean_matches:
            droplist = []
            for idx, m in self._matches.iterrows():
                if (self._last_clustering_ts - self._matches['end']).days > \
                   self.MATCHES_CLEAN_DAYS:
                    droplist.append(idx)
                else:
                    # Match end time is "approximately" sorted (at least at the
                    # level of days)
                    break
            self._matches.drop(droplist, inplace=True)

    def _match_edges_hart(self):
        """
        On/Off matching using edges (as opposed to clusters). This is the method
        implemented by Hart for the two-state load monitor (it won't work
        directly for multi-state appliances). It is implemented as close as
        possible to Hart's original paper (1985). We search for matches at
        neighboring edges first, increasing the distance if we fail to find
        matches.
        """
        if self._edges.empty:
            return
        # This returns a copy in pandas!
        # pbuffer = self._edges.loc[~(self._edges['mark']).astype(bool)]
        # Helper, to keep code tidy
        e = self._edges
        len_e = self._edges.shape[0]
        # Multiple loops, what are the alternatives?
        # Distance of edges
        for dst in range(1, len_e):
            # Check each edge
            for i in range(len_e - 1):
                # Only check positive and unmarked
                if e.iloc[i]['active'] < 0 or e.iloc[i]['mark']:
                    continue
                j = i + dst
                # No more edges to check
                if j >= len_e:
                    break
                # Next edge has been marked before or is positive
                if e.iloc[j]['active'] >= 0 or e.iloc[j]['mark']:
                    continue
                # Do not match edges very far apart in time
                if (e.iloc[j]['start'] - e.iloc[i]['end']).days > \
                   self.MAX_MATCH_THRESHOLD_DAYS:
                    continue
                # Do they match? (use negative of edge for e2, since it is the
                # negative part)
                e1 = e.iloc[i][['active', 'reactive']
                               ].values.astype(np.float64)
                e2 = \
                    -e.iloc[j][['active', 'reactive']
                               ].values.astype(np.float64)
                try:
                    match, d = utils.match_power(e1, e2, active_only=True,
                                                 t=self.MATCH_THRESHOLD)
                except ValueError:
                    continue
                # TODO: Either here or at the sanity checks: For dst > 1 or > 2,
                # introduce a sanity check that ymatch is not higher than y
                if match:
                    # Match (e2 = -e.iloc[j], so it has the "correct" sign)
                    edge = (e1 + e2) / 2.0
                    # TODO: We keep only 'start' time for each edge. Is this OK?
                    # Should we use 'end' time of e.iloc[j]?
                    df = pd.DataFrame({'start': e.iloc[i]['start'],
                                       'end': e.iloc[j]['start'],
                                       'active': edge[0],
                                       'reactive': edge[1]}, index=[0])
                    # Sanity check
                    sanity = self._match_sanity_check(df)
                    if not sanity:
                        continue

                    self._matches = self._matches.append(df, ignore_index=True,
                                                         sort=False)
                    # Mark the edge as matched
                    c = e.columns.get_loc('mark')  # Pandas indexing...
                    e.iat[i, c] = True
                    e.iat[j, c] = True
                    continue
        # Perform sanity checks and clean buffers.
        self._clean_buffers()

    def _match_sanity_check(self, match):
        """
        Update the estimated consumption and make sure that a match does not
        lead to an impossible situation where _ymatch (the estimated
        consumption) is significantly higher than the actual consumption.
        """
        # Update ymatch.
        tds = self._ymatch.index[-1] - match.iloc[0]['start']
        start = tds.days * 3600 * 24 + tds.seconds
        tde = self._ymatch.index[-1] - match.iloc[0]['end']
        end = tde.days * 3600 * 24 + tde.seconds
        active = match.iloc[0]['active']
        # It is possible in cases where there are issues with communication with
        # NILM that start/end exceed the size of _ymatch. In that case just
        # return false.
        # Note that start > end (because these are time differences!)
        if start < end or start > self._ymatch['active'].shape[0]:
            return False
        self._ymatch['active'][-start:-end] += active
        # Hardcoded parameters. Can be more strict.
        check_threshold = max([0.2 * active, 100.0])
        if start > self._buffer.shape[0]:
            start = self._buffer.shape[0] - 1
            if end > start:
                return False
        diff = self._ymatch[-start:-end]['active'].values - \
            self._buffer[-start:-end]['active'].values
        check = np.sum(diff > check_threshold)
        if check > 30:
            self._ymatch['active'][-start:-end] -= active
            return False
        else:
            reactive = match.iloc[0]['reactive']
            self._ymatch['reactive'][-start:-end] += reactive
            return True

    def _match_edges_live(self):
        """
        Adaptation of Hart's edge matching algorithm to support the "live"
        display of eeRIS. Note that this works only in 'online' mode (small
        batches of samples at a time).
        """
        self.detected_appliance = None
        if not self._online_edge_detected:
            # No edge was detected
            if not self.live:
                return
            # If we are on a transition, make sure previous edge is finalized
            # and the device verified
            if self.on_transition:
                self.live[0].update_appliance_live()
                return
            # Update last edge.
            # Check if we need to reconsider previous matching
            if not self.live[0].final:
                p = self.running_avg_power - self._previous_steady_power
                # Update signature
                self.live[0].signature = p.reshape(-1, 1).T
            return
        e = self._online_edge
        if all(np.fabs(e) < self.SIGNIFICANT_EDGE):
            # Although no edge is added, previous should be finalised
            if self.live:
                self.live[0].update_appliance_live()
            return
        if e[0] > 0:
            appliance_id = str(bson.objectid.ObjectId())
            name = 'Live %s' % (str(self._appliance_display_id))
            # TODO: Determine appliance category.
            category = 'unknown'
            a = appliance.Appliance(appliance_id, name, category,
                                    signature=e.reshape(-1, 1).T)
            # Does this look like a known appliance that isn't already matched?
            candidates = self._match_appliances_live(a)
            if not candidates:
                # New appliance. Add to live dictionary using id as key.
                self.appliances_live[a.appliance_id] = a
                self.appliances_live[a.appliance_id].live = True
                # Display as live appliance at the top of the list
                self.live.insert(0, a)
                # Increase display id for next appliance
                self._appliance_display_id += 1
            else:
                # TODO: Trigger notification if appliance is detected (Cluster
                # X)
                # Match with previous and update signature with average
                self.live.insert(0, candidates[0][0])
                # 2x because we take both the rising and dropping edge
                n = 2 * self.live[0].activations.shape[0]
                s = self.live[0].signature[0, :]
                s_a = a.signature[0, :]
                avg_power = n / (n + 1.0) * s + 1.0 / (n + 1.0) * s_a
                self.live[0].signature[0, :] = avg_power
                # Register detection to support notifications to the users for
                # appliance naming.
                self.detected_appliance = None
                # TODO: Restrict to unknown category only? Any additional
                # constraints?
                if not candidates[0][0].live:
                    self.detected_appliance = candidates[0][0]
                    logging.debug("Detected appliance %s. Sending notification "
                                  "request" %
                                  (str(self.detected_appliance.signature)))
            # For activations
            self.live[0].start_ts = self._edge_start_ts
            # Done
            return
        # Appliance cycle stop. Does it match against previous edges (starting
        # from most recent, at 0)?
        matched = False
        for i in range(len(self.live)):
            # e0 = self.live[i].signature.reshape(-1, 1).T
            e0 = self.live[i].signature[0]
            try:
                match, d = utils.match_power(e0, -e, active_only=True,
                                             t=self.MATCH_THRESHOLD)
            except ValueError:
                match = False
                continue
            if match:
                # Store live activations as well.
                start_ts = self.live[i].start_ts
                # As in Hart's edge matching. This could be edge_end_ts
                end_ts = self._edge_start_ts
                # Update running average, take into account matching edge
                n = 2 * self.live[i].activations.shape[0] + 1
                new_e = n / (n + 1.0) * e0 + (1 / (n + 1.0)) * (-e)
                self.live[i].signature[0] = new_e
                active = self.live[i].signature[0][0]
                # Update activations
                self.live[i].append_activation(start_ts, end_ts, active)
                matched = True
                if self._store_live:
                    # If we want to store detected live events for evaluation
                    # and debugging purposes.
                    tmp_df = pd.DataFrame(data={'start': start_ts,
                                                'end': end_ts,
                                                'name': self.live[i].name,
                                                'active':
                                                self.live[i].signature[0][0],
                                                'reactive':
                                                self.live[i].signature[0][1]},
                                          index=[0])
                    self.live_history = \
                        self.live_history.append(tmp_df, ignore_index=True,
                                                 sort=True)
                # Remove appliance from live
                self.live.pop(i)
                break
        if not matched:
            # If we reached this point, then the negative edge did not match
            # anything. This may mean that an appliance will stay turned on
            # indefinitely.
            # TODO: Do we need to check this? If something's wrong, what do we
            # do? Ideas: Match pair of last two edges, keep edge open and wait
            # for next edge. Reset everything.
            pass

        # Make sure all previous live appliances are finalized
        for app in self.live:
            app.update_appliance_live()

    def _match_appliances_live(self, a, t=50.0):
        """
        Helper function to match an online detected appliance against a list of
        appliances.

        Parameters
        ----------

        a : eeris_nilm.appliance.Appliance instance. Appliance used for
        comparison

        t : float. Match threshold

        Returns
        -------

        key, value : Key and value of the matched appliance in the
        _appliances_live dictionary

        """
        # TODO: Replace the use of this function with a call to
        # appliances.match_appliances.
        candidates = []
        for k in self.appliances_live.keys():
            # If it's already in live then ignore it
            if self.appliances_live[k] in self.live:
                continue
            # Assumes two-state appliances.
            # TODO: Use Appliance.compare_power() in the future. (?)
            try:
                match, d = \
                    utils.match_power(self.appliances_live[k].signature[0],
                                      a.signature[0], active_only=False,
                                      t=self.MATCH_THRESHOLD)
            except ValueError:
                continue
            # We want to only consider appliances that are not already
            # on.
            if match and (self.appliances_live[k] not in self.live):
                candidates.append((self.appliances_live[k], d))
            # TODO: Alternative approach (evaluate):
            # d = Appliance.distance(self.appliances_live[k], a)
            # if d < t:
            #     candidates.append(self.appliances_live[k])
        if candidates:
            candidates.sort(key=lambda x: x[1])
        return candidates

    def _match_helper(self, start, end, active):
        """
        Helper function to update the "explained" power consumption _ymatch
        based on a pair of matched edges. For debugging/demonstration purposes.
        """
        # TODO: DEPRECATED. To be removed in future versions
        # TODO: Code below won't work if time difference exceeds 1 day
        # (.seconds)
        end_sec_vis = (self.last_processed_ts - end).seconds
        if end_sec_vis > self.MAX_DISPLAY_SECONDS:
            return
        start_sec_vis = (self.last_processed_ts - start).seconds
        if start_sec_vis > self.MAX_DISPLAY_SECONDS:
            start_sec_vis = self.MAX_DISPLAY_SECONDS
        self._ymatch[-start_sec_vis:-end_sec_vis] += active

    def _update_display(self):
        """
        Provide information for display at the eeRIS "live" screen.
        """
        prev = self._previous_steady_power[0]
        if self.last_processed_ts is not None:
            if self._last_visualized_ts is not None:
                td = (self.last_processed_ts - self._last_visualized_ts)
                # Days should always be zero, but just in case
                step = td.days * 3600 * 24 + td.seconds
            else:
                td = (self.last_processed_ts - self._data.index[0])
                step = td.days * 3600 * 24 + td.seconds + 1
        else:
            step = self._data.shape[0]

        # Update yest
        if self._online_edge_detected and not self.on_transition:
            if self._last_visualized_ts is not None:
                if (self._online_edge_ts > self._last_visualized_ts):
                    td = (self._online_edge_ts - self._last_visualized_ts)
                    step1 = td.days * 3600 * 24 + td.seconds
                else:
                    td = (self._last_visualized_ts - self._online_edge_ts)
                    step1 = td.days * 3600 * 24 + td.seconds + 1
            else:
                # Should not occur normally
                step1 = 1
            step2 = step - step1
            y1 = np.array([prev] * step1)
            y2 = np.array([prev + self._online_edge[0]] * step2)
            self._yest = np.concatenate([self._yest, y1, y2])
        elif self.on_transition:
            self._yest = np.concatenate(
                [self._yest, np.array([prev] * step)]
            )
        else:
            self._yest = np.concatenate(
                [self._yest, np.array([self.running_avg_power[0]] * step)]
            )
        if self._yest.shape[0] > self.MAX_DISPLAY_SECONDS:
            self._yest = self._yest[-self.MAX_DISPLAY_SECONDS:]
        self._last_visualized_ts = self.last_processed_ts

    def _sanity_checks(self):
        # TODO: Need to activate only in case of edges. Checks need to go back
        # in time sufficiently.

        # TODO:
        # Sanity check 1: Matched power should be lower than consumed power
        pass

    def _sanity_checks_live(self):
        """
        Various sanity checks to correct obvious errors of the live edge
        matching algorithm. Also, updates some variables (e.g., background,
        residual power) and statistics of the model.
        """
        self._update_background()
        self._update_residual_live()

    def _update_residual_live(self):
        """
        Function to estimate the power that is not accounted for by the NILM
        algorithms. Note that this can be negative (in case an appliance does
        not switch off!)
        """
        total_estimated = np.array([0.0, 0.0])
        for a in self.live:
            total_estimated += a.signature[0]
        total_estimated += self.background_active
        self.residual_live = self.running_avg_power - total_estimated

        # It's just the background, we don't reset
        if len(self.live) == 0:
            return

        # Allow for 20% error
        if self.running_avg_power[0] < 0.8 * total_estimated[0]:
            # We may have made a matching error, and an appliance should have
            # been switched off. Heuristic solution here.
            # self.live = []
            # self.residual_live = self.running_avg_power
            self._count_res_overestimation += 1
            if self._count_res_overestimation > self.OVERESTIMATION_SECONDS:
                logging.info(("Something's wrong with the residual estimation: "
                              "Background: %f, Residual: %f") %
                             (self.background_active, self.residual_live[0]))
                logging.info("Resetting")
                self.live = []
                total_estimated = self.background_active
                self.residual_live[0] = 0.0
        else:
            self._count_res_overestimation = 0

    def _update_background(self):
        """
        Maintain an estimate of the background power consumption
        """
        # TODO: Fix oscillation between LARGE_POWER and background that often
        # happens in the initialization phase
        if self._background_last_update is not None and \
           self.data is not None and \
           self.data.shape[0] > 0:
            td = (self.data.index[-1] - self._background_last_update)
            hours_since_update = td.total_seconds() / 3600.0
        else:
            hours_since_update = self.BACKGROUND_UPDATE_PERIOD_HOURS + 1.0
        if hours_since_update > self.BACKGROUND_UPDATE_PERIOD_HOURS and \
           not self._steady_states.empty:
            last_ts = self._steady_states.iloc[-1]['start']
            idx = self._steady_states['start'] > \
                (last_ts - pd.Timedelta(days=self.BACKGROUND_UPDATE_DAYS))
            steady_states = self._steady_states[idx]['active'].values
            # We assume that background consumption is more than 1 Watt. This
            # ignores missing values or errors in measurements that result in
            # zeros.
            v = steady_states[steady_states > 1.0]
            if v.shape[0] > 0 and \
               self.data is not None and \
               self.data.shape[0] > 0:
                self.background_active = np.min(v)
                self._background_last_update = self.data.index[-1]
            else:
                # Return if no new steady states exist (how?)
                return
        # Current background estimate seems to be inaccurate.
        if self.background_active > 1.2 * self.running_avg_power[0]:
            self._count_bg_overestimation += 1
            if self._count_bg_overestimation > self.OVERESTIMATION_SECONDS:
                logging.warning(
                    ("Something's wrong with the background estimation:"
                     "Background: %f, Residual: %f") %
                    (self.background_active, self.residual_live[0])
                )
                # self.background_active = self.LARGE_POWER
                # self._background_last_update = None
        else:
            self._count_bg_overestimation = 0

    def _guess_type(self):
        """
        Guess the appliance type using an unnamed hart model
        """
        pass

    def is_background_overestimated(self):
        """
        Returns True if current background has been overestimated.
        """
        if self._count_bg_overestimation > 10:
            return True
        else:
            return False

    def is_clustering_active(self):
        if self._clustering_thread is None:
            return False
        if self._clustering_thread.is_alive():
            return True
        else:
            # There are problems storing the thread via dill, so we set this to
            # None.
            self._clustering_thread = None
            return False

    def force_clustering(self, start_thread=False, method="dbscan"):
        """
        This function forces recomputation of appliance models using the
        detected and matched edges (after clustering) and returns the activation
        histories of each appliance. Clustering takes place in a separate
        thread.

        Parameters
        ----------

        start_thread : bool
        Threaded version, where a separate clustering thread is started
        method : string
        Method to use for clustering. Can be one of "dbscan" or "mean_shift"

        Returns
        -------

        out : bool
        True if the thead was successfully started, False if a clustering thread
        is already running for the installation
        """
        # We acquire lock here as well since this function is public and may be
        # called arbitrarily
        if not start_thread:
            self._static_cluster(method=method)
            return
        # With threads
        if (self._clustering_thread is None) or \
           (not self._clustering_thread.is_alive()):
            self._clustering_thread = \
                threading.Thread(target=self._static_cluster,
                                 name='clustering_thread',
                                 kwargs={'method': method})
            self._clustering_thread.start()
            # To ensure that the lock can be acquired by the thread
            time.sleep(0.01)
            return True
        else:
            return False

    def update(self, data=None, start_thread=True):
        """
        Wrapper to sequence of operations for model update.

        Parameters
        ----------

        data : pandas.DataFrame
        pandas dataframe with timestamp index and measurements including
        'active', 'reactive', 'voltage'

        start_thread : bool
        Threaded version, where a separate clustering thread is started
        """
        # For thread safety
        with self._lock:
            # logging.debug('Calling update()')
            if data is not None:
                self.data = data
            else:
                # If data is empty, do nothing
                return
            # Preprocessing: Resampling, normalization, missing values, etc.
            self._preprocess()

            # Make sure data still exists
            if self._buffer.empty:
                return

            # Edge detection
            self._detect_edges_hart()
            # Edge matching
            if self._edge_detected:
                self._match_edges_hart()

            # Sanity checks
            self._sanity_checks()

            # Live display update (for demo/debugging purposes)
            self._update_display()

            # TODO: Fix bugs in sanity checks
            self._match_edges_live()
            # Sanity checks - live
            self._sanity_checks_live()

            # Clustering
            #
            # Static clustering option. If needed we will add a dynamic
            # clustering option in the future. This runs as a thread in the
            # background.
            if self._last_clustering_ts is not None:
                td = self.last_processed_ts - self._last_clustering_ts
            else:
                td = self.last_processed_ts - self._start_ts + \
                    datetime.timedelta(seconds=1)

        if td.total_seconds() / 3600.0 >= self.CLUSTER_STEP_HOURS:
            self.force_clustering(method=self.CLUSTERING_METHOD,
                                  start_thread=start_thread)
        time.sleep(0.01)
