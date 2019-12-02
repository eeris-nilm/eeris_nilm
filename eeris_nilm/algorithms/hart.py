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
import scipy.signal
import pandas as pd
import eeris_nilm.appliance
import sklearn.cluster
import sklearn.metrics.pairwise
import logging
# TODO: Use object ids only on eeris branch. I think this is not a good practice in
# general.
import bson


class Hart85eeris():
    """ Modified implementation of Hart's NILM algorithm. """
    # TODO:
    # - remove the class variables that are not needed
    # - limit the number of edges and steady states that will be cached

    NOMINAL_VOLTAGE = 230.0
    BUFFER_SIZE_SECONDS = 600
    MAX_WINDOW_DAYS = 100
    MAX_NUM_STATES = 1000
    MAX_DISPLAY_SECONDS = 10 * 3600
    # These could be parameters
    STEADY_THRESHOLD = 15
    SIGNIFICANT_EDGE = 50
    STEADY_SAMPLES_NUM = 5
    MATCH_THRESHOLD = 35
    # DEBUG: Bring these values to reasonable values when finished.
    CLUSTER_STEP_DAYS = 1.0/24.0  # Update every day
    CLUSTER_FIRST_DAYS = 1.0/24.0  # Change to 10 in production
    CLUSTER_DATA_DAYS = 30 * 3  # Use last 3 months for clustering
    MIN_EDGES_STATIC_CLUSTERING = 5  # DBSCAN parameter
    LARGE_POWER = 1e6  # Large power consumption
    BACKGROUND_UPDATE_DAYS = 10  # How many days since background was updated
    # How many days since we updated the more "recent" background
    BACKGROUND_RECENT_DAYS = 5

    def __init__(self, installation_id):
        # Almost all variables are needed as class members, to support streaming
        # support.

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
        # Value of last estimate
        self._last_measurement = 0.0
        # Timestamp of last processed sample
        self._last_processed_ts = None

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
        self._ymatch = None
        self._ymatch_live = None
        # Installation id (is this necessary?)
        self.installation_id = installation_id
        # List of states and transitions detected so far.
        self._steady_states = pd.DataFrame([],
                                           columns=['start', 'end', 'active',
                                                    'reactive'])
        self._edges = pd.DataFrame([], columns=['start', 'end', 'active',
                                                'reactive', 'mark'])
        # Matched devices
        self._matches = pd.DataFrame([], columns=['start', 'end', 'active',
                                                  'reactive'])
        # Timestamps for keeping track of the data processed, for edges, steady
        # states and clusters
        self._start_ts = None
        self._last_clustering_ts = None
        self._edge_start_ts = None
        self._edge_end_ts = None
        self._steady_start_ts = None
        self._steady_end_ts = None
        # For online edge detection
        self.online_edge_detected = False
        self.online_edge = np.array([0.0, 0.0])
        # List of live appliances
        self.live = []
        # Current live appliance id.
        # TODO: Keep this only on eeris branch. Remove everywhere else.
        self._appliance_id = str(bson.objectid.ObjectId())
        # Dictionaries of known appliances
        self._appliances = {}
        self._appliances_live = {}
        # Other variables - needed for sanity checks
        self._background_active = self.LARGE_POWER
        self._background_active_recent = self.LARGE_POWER
        self._background_last_update = None
        self._background_recent_update = None
        self._residual_live = None
        logging.debug("Hart object initialized.")

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
        # Check time difference
        duration = (data.index[-1] - data.index[0]).days
        if duration > self.MAX_WINDOW_DAYS:
            # Do not process the window, it's too long.
            raise ValueError('Data duration too long')
        # Store the original data
        self._data_orig = data

    def _preprocess(self):
        """
        Data preprocessing steps. It also updates a sliding window of
        BUFFER_SIZE_SECONDS of data. Current version resamples to 1Hz sampling
        frequency.
        """
        if self._buffer is None:
            self._buffer = self._get_normalized_data()
            assert self._start_ts is None  # Just making sure
            self._start_ts = self._data_orig.index[0]
        else:
            # Data concerning past dates update the buffer
            self._buffer = self._buffer.append(self._get_normalized_data())
        # Round timestamps to 1s
        self._buffer = self._buffer.sort_index()
        self._buffer.index = self._buffer.index.round('1s')
        # Remove possible duplicate entries (keep the last entry), based on
        # timestamp
        self._buffer = self._buffer.reset_index()
        self._buffer = self._buffer.drop_duplicates(subset='index', keep='last')
        self._buffer = self._buffer.set_index('index')
        # Resample to 1s
        self._buffer = self._buffer.asfreq('1S', method='pad')
        # Keep only the last BUFFER_SIZE_SECONDS of the buffer
        start_ts = self._buffer.index[-1] - \
            pd.offsets.Second(self.BUFFER_SIZE_SECONDS - 1)
        self._buffer = self._buffer.loc[self._buffer.index >= start_ts]
        if self._last_processed_ts is None:
            self._data = self._buffer
            self._idx = self._buffer.index[0]
            self._steady_start_ts = self._idx
        else:
            self._idx = self._last_processed_ts + 1 * self._buffer.index.freq
            self._data = self._buffer.loc[self._idx:]
        # TODO: Handle N/As and zero voltage.
        # TODO: Unit tests with all the unusual cases

    def _get_normalized_data(self):
        """
        Normalize power with voltage measurements.
        """
        # Normalization. Raise active power to 1.5 and reactive power to
        # 2.5. See Hart's 1985 paper for an explanation.

        # Copy the data to be avoid in-place weirdness
        r_data = self._data_orig.copy()
        r_data.loc[:, 'active'] = self._data_orig['active'] * \
            np.power((self.NOMINAL_VOLTAGE / self._data_orig['voltage']), 1.5)
        r_data.loc[:, 'reactive'] = self._data_orig['reactive'] * \
            np.power((self.NOMINAL_VOLTAGE / self._data_orig['voltage']), 2.5)
        return r_data

    def _detect_edges(self):
        """
        TODO: Advanced identification of steady states and transitions based on
        active and reactive power.
        """
        pass

    def _detect_edges_hart(self):
        """
        Edge detector, based on Hart's algorithm.
        """
        self.online_edge_detected = False
        self.online_edge = np.array([0.0, 0.0])
        if self._last_processed_ts is None:
            data = self._buffer[['active', 'reactive']].values
            prev = data[0, :]
        else:
            tmp_df = self._buffer[['active', 'reactive']]
            prev = tmp_df.loc[self._last_processed_ts].values
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
                        self.online_edge_detected = True
                        # self.online_edge = self.running_edge_estimate
                        self.online_edge = self.running_avg_power - \
                            self._previous_steady_power
                        self._edge_count = 0
            self._samples_count += 1
        # Update lists
        self._edges = pd.concat(edge_list, ignore_index=True)
        self._steady_states = pd.concat(steady_list, ignore_index=True)
        # Update last processed
        self._last_processed_ts = self._buffer.index[-1]
        self._last_measurement = self._buffer.iloc[-1]

    def _static_cluster(self):
        """
        Clustering step of Hart's method. Here it is implemented as a static
        clustering step that runs periodically, mapping previous devices to the
        new device names.

        In this implementation we use "static" clustering (apply clustering
        periodically) instead of a dynamic cluster update. We also apply DBSCAN
        algorithm, to avoid the normality assumptions made by Hart's original
        algorithm.
        """
        # Select matched edges to use for clustering
        matches = self._matches[['start', 'active', 'reactive']]
        if len(matches) < self.MIN_EDGES_STATIC_CLUSTERING:
            return
        start_ts = matches['start'].iloc[-1] - \
            pd.offsets.Day(self.CLUSTER_DATA_DAYS)
        matches = matches.loc[matches['start'] > start_ts]
        matches = matches[['active', 'reactive']].values
        # Apply DBSCAN.
        # TODO: Experiment on the options.
        # TODO: Normalize matches in the 0-1 range, so that difference is
        # percentage! This will perhaps allow better matches.
        # TODO: Possibly start with high detail and then merge similar clusters?
        d = sklearn.cluster.DBSCAN(eps=30, min_samples=3, metric='euclidean',
                                   metric_params=None, algorithm='auto',
                                   leaf_size=30)
        d.fit(matches)
        # DBSCAN only: How many clusters are there? Can we derive "centers"?
        u_labels = np.unique(d.labels_[d.labels_ >= 0])
        centers = np.zeros((u_labels.shape[0], matches.shape[1]))
        for l in u_labels:
            centers[l] = np.mean(matches[d.labels_ == l, :], axis=0)

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
            a = eeris_nilm.appliance.Appliance(
                l, name, category, signature=centers[l, :])
            appliances[a.appliance_id] = a
        if not self._appliances:
            # First time we detect appliances
            self._appliances = appliances
        else:
            # Map to previous
            self._appliances = self._match_appliances(appliances,
                                                      self._appliances)
        # Sync live appliances
        self._appliances_live = self._match_appliances(self._appliances_live,
                                                       self._appliances)
        # Set timestamp
        self._last_clustering_ts = self._buffer.index[-1]

        logging.debug('Clustering complete. Current list of appliances:')
        logging.debug(str(self._appliances))

    def _match_appliances(self, a_from, a_to, t=20):
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
                d = eeris_nilm.appliance.Appliance.distance(a_from[k], a_to[l])
                if d < t:
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

    def _dynamic_cluster(self):
        """
        Dynamic clustering step, as proposed by Hart.

        NOT IMPLEMENTED
        """
        pass

    def _clean_edges_buffer(self):
        """
        Clean-up edges buffer. This removes matched edges from the buffer, but
        may also remove edges that have remained in the buffer for a very long
        time, perform other sanity checks etc. It's currently work in progress.
        """
        self._edges.drop(self._edges.loc[self._edges['mark']].index,
                         inplace=True)
        # TODO:
        # Sanity check 1: Matched power should be lower than consumed power

    def _match_edges_hart(self):
        """
        On/Off matching using edges (as opposed to clusters). This is the method
        implemented by Hart for the two-state load monitor (it won't work
        directly for multi-state appliances). It is implemented as close as
        possible to Hart's original paper (1985). The approach is rather
        simplistic and can lead to serious errors.
        """
        if self._edges.empty:
            return
        # This returns a copy in pandas!
        # pbuffer = self._edges.loc[~(self._edges['mark']).astype(bool)]
        # Helper, to keep code tidy
        e = self._edges
        # Double for loop, what are the alternatives?
        for i in range(len(e)):
            # Only check positive and unmarked
            if e.iloc[i]['active'] < 0 or e.iloc[i]['mark']:
                continue
            e1 = e.iloc[i][['active', 'reactive']].values.astype(np.float64)
            for j in range(i + 1, len(e)):
                # Edge has been marked before or is positive
                if e.iloc[j]['active'] >= 0 or e.iloc[j]['mark']:
                    continue
                # Do they match?
                e2 = e.iloc[j][['active', 'reactive']].values.astype(np.float64)
                if self._match_power(e1, e2):
                    # Match
                    edge = (np.fabs(e1) + np.fabs(e2)) / 2.0
                    # Ideally we should keep both start and end times for each
                    # edge
                    df = pd.DataFrame({'start': e.iloc[i]['start'],
                                       'end': e.iloc[j]['start'],
                                       'active': edge[0],
                                       'reactive': edge[1]}, index=[0])
                    self._matches = self._matches.append(df, ignore_index=True,
                                                         sort=True)
                    # Get the 'mark' column.
                    c = e.columns.get_loc('mark')
                    e.iat[i, c] = True
                    e.iat[j, c] = True
                    break
        # Perform sanity checks and clean edges.
        self._clean_edges_buffer()

    def _match_edges_hart_live(self):
        """
        Adaptation of Hart's edge matching algorithm to support the "live"
        display of eeRIS.
        """
        if not self.online_edge_detected:
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
                self.live[0].signature = p
            return
        e = self.online_edge
        if all(np.fabs(e) < self.SIGNIFICANT_EDGE):
            # Although no edge is added, previous should be finalised
            if self.live:
                self.live[0].update_appliance_live()
            return
        if e[0] > 0:
            name = 'Unknown live appliance %s' % (str(self._appliance_id))
            # TODO: Determine appliance category
            category = 'Unknown'
            a = eeris_nilm.appliance.Appliance(
                self._appliance_id, name, category, signature=e)
            # Does this look like a known appliance that isn't already matched?
            candidates = self._match_appliances_live(a)
            if not candidates:
                # New appliance. Add to live dictionary using id as key.
                self._appliances_live[a.appliance_id] = a
                self.live.insert(0, a)
                self._appliance_id = str(bson.objectid.ObjectId())
            else:
                # Match with previous
                self.live.insert(0, candidates[0])
            # Done
            return
        # Appliance cycle stop. Does it match against previous edges?
        for i in range(len(self.live)):
            e0 = self.live[i].signature
            if self._match_power(e0, e):
                self.live.pop(i)
                break
        # Make sure all previous appliances are finalized
        for app in self.live:
            app.update_appliance_live()

    def _match_power(self, p1, p2):
        """
        Match power consumption p1 against p2 according to Hart's algorithm.

        Parameters
        ----------

        p1, p2 : Numpy arrays with two elements (active and reactive power)

        Returns
        -------

        out : Boolean for match (True) or no match (False)

        """
        # Can be positive or negative edge
        if np.fabs(p2[0]) >= 1000:
            t_active = 0.05 * p2[0]
        else:
            t_active = self.MATCH_THRESHOLD
        if np.fabs(p2[1]) >= 1000:
            t_reactive = 0.05 * p2[1]
        else:
            t_reactive = self.MATCH_THRESHOLD
        T = np.fabs(np.array([t_active, t_reactive]))
        # Match only with active power for now
        if np.fabs(np.fabs(p2[0]) - np.fabs(p1[0])) < T[0]:
            # Match
            return True
        else:
            return False

    def _match_appliances_live(self, a, t=20):
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
        candidates = []
        for k in self._appliances_live.keys():
            # If it's already in live then ignore it
            if self._appliances_live[k] in self.live:
                continue
            d = eeris_nilm.appliance.Appliance.distance(
                self._appliances_live[k], a
            )
            if d < t:
                candidates.append(self._appliances_live[k])
        return candidates

    def _match_helper(self, start, end, active):
        """
        Helper function to update the "explained" power consumption _ymatch
        based on a pair of matched edges.
        """
        end_sec_inv = (self._last_processed_ts - end).seconds
        if end_sec_inv > self.MAX_DISPLAY_SECONDS:
            return
        start_sec_inv = (self._last_processed_ts - start).seconds
        if start_sec_inv > self.MAX_DISPLAY_SECONDS:
            start_sec_inv = self.MAX_DISPLAY_SECONDS
        self._ymatch[-start_sec_inv:-end_sec_inv] += active

    def _update_live(self):
        """
        Provide information for display at the eeRIS "live" screen. Preliminary
        version.
        """
        # prev = self._est_prev
        step = self._data.shape[0]
        # Update yest
        if self.online_edge_detected and not self.on_transition:
            prev = self._previous_steady_power[0]
            y1 = np.array([prev] * (step // 2))
            y2 = np.array([prev + self.online_edge[0]] * (step - step // 2))
            self._yest = np.concatenate([self._yest, y1, y2])
            # self._est_prev = self._previous_steady_power[0]
            # prev = self._est_prev
        elif self.on_transition:
            self._yest = np.concatenate(
                [self._yest, np.array([self._previous_steady_power[0]] * step)]
            )
        else:
            self._yest = np.concatenate(
                [self._yest, np.array([self.running_avg_power[0]] * step)]
            )
            # self._est_prev = self.running_avg_power[0]
            # prev = self._est_prev
        if self._yest.shape[0] > self.MAX_DISPLAY_SECONDS:
            self._yest = self._yest[-self.MAX_DISPLAY_SECONDS:]
        # Update ymatch
        self._ymatch = np.zeros_like(self._yest)
        [self._match_helper(x, y, z)
         for x, y, z in
         zip(self._matches['start'],
             self._matches['end'],
             self._matches['active'])]

    def _sanity_checks(self):
        # TODO: Need to activate only in case of edges. Checks need to go back
        # in time sufficiently.
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
        pass

    def _update_background(self):
        """
        Maintain an estimate of the background power consumption
        """
        # First, update the installation's background using a mean filter of
        # size 3
        m = scipy.signal.convolve(self._data['active'].values,
                                  np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
                                  mode='valid')
        background = np.min(m)
        if self._background_last_update is not None:
            days_since_update = (self.data.index[-1] -
                                 self._background_last_update).days
        else:
            days_since_update = 0
        if self._background_recent_update is not None:
            days_recent_update = (self.data.index[-1] -
                                  self._background_recent_update).days
        else:
            days_recent_update = 0

        if self._background_active_recent >= background or \
           days_recent_update >= self.BACKGROUND_RECENT_DAYS:
            self._background_active_recent = background
            self._background_recent_update = self.data.index[-1]
        if days_since_update >= self.BACKGROUND_UPDATE_DAYS:
            self._background_active = self._background_active_recent
            self._background_last_update = self._background_recent_update
        if self._background_active >= background:
            self._background_active = background
            self._background_last_update = self.data.index[-1]

    def _guess_type(self):
        """
        Guess the appliance type using an unnamed hart model
        """
        pass

    def update(self, data=None):
        """
        Wrapper to sequence of operations for model update
        """
        if data is not None:
            self.data = data
        # Preprocessing: Resampling, normalization, missing values, etc.
        self._preprocess()
        # Edge detection
        self._detect_edges_hart()
        # Edge matching
        self._match_edges_hart()
        # Sanity checks
        self._sanity_checks()

        # Live update
        self._update_live()

        # TODO: Fix bugs in sanity checks
        # self._match_edges_hart_live()
        # Sanity checks - live
        # self._sanity_checks_live()

        # Clustering
        #
        # 1. Static clustering option. If needed we will add a dynamic
        # clustering option in the future.
        #
        # TODO: Turn this into a thread, if we decide to keep it after
        # all. Updating will need to lock the list of appliances, while
        # clustering locks the matches, makes a copy, releases the matches and
        # then performs DBSCAN. It then locks the list of appliances, updates
        # them and releases them back.
        if self._last_clustering_ts is not None:
            td = self._last_processed_ts - self._last_clustering_ts
            if td.seconds/3600.0/24 > self.CLUSTER_STEP_DAYS:
                self._static_cluster()
        else:
            td = self._last_processed_ts - self._start_ts
            if td.seconds/3600.0/24 > self.CLUSTER_FIRST_DAYS:
                self._static_cluster()
