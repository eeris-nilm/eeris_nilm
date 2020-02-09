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
import bson

# TODO: Background has been computed on normalized data and may have
# discrepancies from the actual background consumption (as measured through the
# meter)


class Hart85eeris(object):
    """ Modified implementation of Hart's NILM algorithm. """
    # TODO:
    # - remove the class variables that are not needed
    # - all "live" data should be part of a different class to simplify things a
    # little

    # Some of the variables below could be parameters
    BUFFER_SIZE_SECONDS = 5 * 3600

    # Limiters/thresholds
    MAX_WINDOW_DAYS = 100
    MAX_NUM_STATES = 1000
    MAX_DISPLAY_SECONDS = 10 * 3600
    STEADY_THRESHOLD = 15
    SIGNIFICANT_EDGE = 50
    STEADY_SAMPLES_NUM = 7
    MATCH_THRESHOLD = 35
    MAX_MATCH_THRESHOLD_DAYS = 1
    EDGES_CLEAN_DAYS = 15
    STEADY_CLEAN_DAYS = 15
    MATCHES_CLEAN_DAYS = 6 * 30
    OVERESTIMATION_SECONDS = 10

    # For clustering
    # TODO: Check the clustering values
    CLUSTER_STEP_DAYS = 1.0  # Update every day
    CLUSTER_FIRST_DAYS = 3.0  # Period before first clustering
    CLUSTER_DATA_DAYS = 30 * 3  # Use last 3 months for clustering
    MIN_EDGES_STATIC_CLUSTERING = 5  # DBSCAN parameter
    LARGE_POWER = 1e6  # Large power consumption
    BACKGROUND_UPDATE_DAYS = 10  # How many days since background was updated

    def __init__(self, installation_id, nominal_voltage=230.0):
        # Almost all variables are needed as class members, to support streaming
        # support.

        # Power system parameters
        self.nominal_voltage = nominal_voltage

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
        self._last_processed_ts = None
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
        # TODO: Do we need these at all?
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
        self.online_edge_detected = False
        self.online_edge = np.array([0.0, 0.0])
        # List of live appliances
        self.live = []
        # Current live appliance id.
        # NOTE: I don't find using bson object ids is a good practice. Doing
        # this due to integration requirements by other eeRIS modules.
        self._appliance_id = str(bson.objectid.ObjectId())
        # This could simply be the appliance id instead of bson.objectid
        self._appliance_display_id = 0
        # Dictionaries of known appliances
        self.appliances = {}
        self.appliances_live = {}

        # Other variables - needed for sanity checks
        self.background_active = self.LARGE_POWER
        self._background_last_update = None
        self.residual_live = np.array([0.0, 0.0])
        self._count_overestimation = 0

        # Variables for handling threads. For now, just a lock.
        self._clustering_thread = None
        self._lock = threading.Lock()

        # Installation statistics
        self._min_active = 0.0
        self._max_active = 0.0

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
        # TODO: What about NAs? Apply dropna?
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
                                          nominal_voltage=self.nominal_voltage))
        # Data pre-processing (remove duplicates, resample to 1s)
        self._buffer = utils.preprocess_data(self._buffer)
        # Keep only the last BUFFER_SIZE_SECONDS of the buffer. Should this be
        # done before preprocesssing?
        start_ts = self._buffer.index[-1] - \
            pd.offsets.Second(self.BUFFER_SIZE_SECONDS - 1)
        self._buffer = self._buffer.loc[self._buffer.index >= start_ts]
        if self._last_processed_ts is None:
            # We're just starting
            self._data = self._buffer
            self._idx = self._buffer.index[0]
            self._steady_start_ts = self._idx
        elif self._last_processed_ts < self._buffer.index[0]:
            # After a big gap, reset everything
            self._last_processed_ts = self._buffer.index[0]
            self._idx = self._last_processed_ts
            self._data = self._buffer
            self._reset()
        else:
            # Normal operation: Just get the data that has not been previously
            # been processed
            self._idx = self._last_processed_ts + 1 * self._buffer.index.freq
            self._data = self._buffer.loc[self._idx:]
        # TODO: Handle N/As and zero voltage.
        # TODO: Unit tests with all the unusual cases

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
        self.online_edge_detected = False
        self.online_edge = np.array([0.0, 0.0])
        self.live = []

    def _detect_edges_hart(self):
        """
        Edge detector, based on Hart's algorithm.
        """
        self._edge_detected = False
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

    def _static_cluster(self):
        """
        Clustering step of Hart's method. Here it is implemented as a static
        clustering step that runs periodically, mapping previous devices to the
        new device names.

        In this implementation we use "static" clustering (apply clustering
        periodically) instead of a dynamic cluster update. We also apply DBSCAN
        algorithm, to avoid the normality assumptions made by Hart's original
        algorithm.

        This function is designed to run as a thread.
        """
        self._lock.acquire()
        # Select matched edges to use for clustering
        matches = self._matches.copy()
        matches = matches[['start', 'active', 'reactive']]
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
        self._lock.release()
        debug_t_start = datetime.datetime.now()
        logging.debug('Initiating static clustering at %s' % debug_t_start)
        d = sklearn.cluster.DBSCAN(eps=self.MATCH_THRESHOLD, min_samples=3,
                                   metric='euclidean', metric_params=None,
                                   algorithm='auto', leaf_size=30)
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
            a = appliance.Appliance(l, name, category, signature=centers[l, :])
            appliances[a.appliance_id] = a
        debug_t_end = datetime.datetime.now()
        debug_t_diff = (debug_t_end - debug_t_start)
        logging.debug('Finished static clustering at %s' % (debug_t_end))
        logging.debug('Total clustering time: %s seconds' %
                      (debug_t_diff.seconds))
        self._lock.acquire()
        if not self.appliances:
            # First time we detect appliances
            self.appliances = appliances
        else:
            # Map to previous
            self.appliances = \
                appliance.Appliance.match_appliances(appliances,
                                                     self.appliances)
        # Sync live appliances
        self.appliances_live = \
            appliance.Appliance.match_appliances(self.appliances_live,
                                                 self.appliances)
        # Set timestamp
        self._last_clustering_ts = self._buffer.index[-1]

        logging.debug('Clustering complete. Current list of appliances:')
        logging.debug(str(self.appliances))
        self._lock.release()

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
        # Clean edges that are too far back in time
        droplist = []
        for idx, e in self._edges.iterrows():
            if (self._last_processed_ts - e['end']).days > \
               self.EDGES_CLEAN_DAYS:
                droplist.append(idx)
            else:
                # Edge times are sorted
                break

        self._edges.drop(droplist, inplace=True)

        # Clean steady states that are too far back in time
        droplist = []
        for idx, s in self._steady_states.iterrows():
            if (self._last_processed_ts - s['end']).days > \
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
        possible to Hart's original paper (1985). The approach is rather
        simplistic and can lead to serious detection errors.
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
                # Do not match edges very far apart in time
                if (e.iloc[j]['start'] - e.iloc[i]['end']).days > \
                   self.MAX_MATCH_THRESHOLD_DAYS:
                    continue
                # Do they match? (use negative of edge, since it is the negative
                # part)
                e2 = -e.iloc[j][['active', 'reactive']].\
                    values.astype(np.float64)
                match, d = utils.match_power(e1, e2, active_only=True,
                                             t=self.MATCH_THRESHOLD)
                if match:
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
        # Perform sanity checks and clean buffers.
        self._clean_buffers()

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
            name = 'Live %s' % (str(self._appliance_display_id))
            # TODO: Determine appliance category
            category = 'unknown'
            a = appliance.Appliance(self._appliance_id, name, category,
                                    signature=e)
            # Does this look like a known appliance that isn't already matched?
            candidates = self._match_appliances_live(a)
            if not candidates:
                # New appliance. Add to live dictionary using id as key.
                self.appliances_live[a.appliance_id] = a
                self.live.insert(0, a)
                self._appliance_id = str(bson.objectid.ObjectId())
                # Increase display id for next appliance
                self._appliance_display_id += 1
            else:
                # Match with previous
                self.live.insert(0, candidates[0][0])
            # Done
            return
        # Appliance cycle stop. Does it match against previous edges?
        matched = False
        for i in range(len(self.live)):
            e0 = self.live[i].signature
            match, d = utils.match_power(e0, -e, active_only=True,
                                         t=self.MATCH_THRESHOLD)
            if match:
                self.live.pop(i)
                matched = True
                break
        if not matched:
            # If we reached this point, then the negative edge did not match
            # anything. This may mean that an appliance will stay turned on
            # indefinitely.
            # TODO: Do we need to check this? If something's wrong, what do we
            # do? Ideas: Match pair of last two edges, keep edge open and wait
            # for next edge. Reset everything.
            pass

        # Make sure all previous appliances are finalized
        for app in self.live:
            app.update_appliance_live()

    def _match_appliances_live(self, a, t=35.0):
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
            # TODO: Use Appliance.compare_power() in the future.
            match, d = utils.match_power(self.appliances_live[k].signature,
                                         a.signature, active_only=False,
                                         t=self.MATCH_THRESHOLD)
            if match:
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
            total_estimated += a.signature
        total_estimated += self.background_active
        # Allow for 10% error in edge estimation
        if self.running_avg_power[0] < 0.9*total_estimated[0]:
            # We may have made a matching error, and an appliance should have
            # been switched off. Heuristic solution here.
            # self.live = []
            # self.residual_live = self.running_avg_power
            self._count_overestimation += 1
            if self._count_overestimation > self.OVERESTIMATION_SECONDS:
                self.live = []
                total_estimated = self.background_active
        else:
            self._count_overestimation = 0
        self.residual_live = self.running_avg_power - total_estimated
        if self.residual_live[0] < 0:
            logging.debug(("Something's wrong with the residual estimation:"
                           "Background: %f, Residual: %f") %
                          (self.background_active, self.residual_live[0]))
            self.residual_live[0] = 0.0

    def _update_background(self):
        """
        Maintain an estimate of the background power consumption
        """
        steady_states = self._steady_states['active'].values
        # We assume that background consumption is more than 1 Watt. This
        # ignores missing values or errors in measurements that result in
        # zeros.
        v = steady_states[steady_states > 1.0]
        if v.shape[0] > 0:
            background = np.min(v)
        else:
            # Return if no new values have been added.
            return

        # Days since update of background
        # TODO Debug/check
        days_since_update = 0
        if self._background_last_update is not None:
            if self.data.shape[0] > 0:
                days_since_update = (self.data.index[-1] -
                                     self._background_last_update).days
        else:
            days_since_update = self.BACKGROUND_UPDATE_DAYS + 1

        # Update background estimate
        if self.background_active > background or \
           days_since_update >= self.BACKGROUND_UPDATE_DAYS:
            self.background_active = background
            if self.data.shape[0] > 0:
                self._background_last_update = self.data.index[-1]

        # Hard way of dealing with discrepancies: Reset background
        if self.background_active > self.running_avg_power[0]:
            logging.debug(("Something's wrong with the background estimation:"
                           "Background: %f, Residual: %f") %
                          (self.background_active, self.residual_live[0]))
            self.background_active = self.LARGE_POWER
            self._background_last_update = None

    def _guess_type(self):
        """
        Guess the appliance type using an unnamed hart model
        """
        pass

    def update(self, data=None):
        """
        Wrapper to sequence of operations for model update. Normally, this the
        only function anyone will need to call to use the model.
        """
        # For thread safety
        self._lock.acquire()
        if data is not None:
            self.data = data
        # Preprocessing: Resampling, normalization, missing values, etc.
        self._preprocess()
        # Edge detection
        self._detect_edges_hart()
        # Edge matching
        if self._edge_detected:
            self._match_edges_hart()

        # Sanity checks
        self._sanity_checks()

        # Live update
        self._update_live()

        # TODO: Fix bugs in sanity checks
        self._match_edges_hart_live()
        # Sanity checks - live
        self._sanity_checks_live()

        # Clustering
        #
        # Static clustering option. If needed we will add a dynamic
        # clustering option in the future. This runs as a thread in the
        # background.
        if self._last_clustering_ts is not None:
            td = self._last_processed_ts - self._last_clustering_ts
        else:
            td = self._last_processed_ts - self._start_ts
        self._lock.release()

        if td.days >= self.CLUSTER_STEP_DAYS:
            if (self._clustering_thread is None) or \
               (not self._clustering_thread.is_alive()):
                self._clustering_thread = \
                    threading.Thread(target=self._static_cluster,
                                     name='clustering_thread')
                self._clustering_thread.start()
            # To ensure that the lock can be acquired by the thread
            time.sleep(0.01)
