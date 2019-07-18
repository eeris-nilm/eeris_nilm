"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential



Modified implementation of Hart's NILM algorithm, adapted for the needs of
eeRIS. The two most important modifications are (i) support for online detection
and (ii) rules for fixing errors related with appliances that are not
two-state.
"""

import numpy as np
import pandas as pd
import eeris_nilm.appliance
import sklearn.cluster


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
    MIN_EDGES_STATIC_CLUSTERING = 5

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
        self._appliance_id = 0
        # List of live appliances
        self.live = []
        # Dictionaries of known appliances
        self._appliances = {}
        self._appliances_live = {}

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
            self._buffer = self._data_orig.copy()
            assert self._start_ts is None  # Just making sure
            self._start_ts = self._data_orig.index[0]
        else:
            # Data concerning past dates update the buffer
            self._buffer = self._buffer.append(self._data_orig)
        # Round timestamps to 1s
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

    def _normalize(self):
        """
        Resample data to 1s and normalize power with voltage measurements. Drop
        missing values.
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
        data = self._matches[['start', 'active', 'reactive']]
        if len(data) < self.MIN_EDGES_STATIC_CLUSTERING:
            return
        # ERROR here!
        start_ts = data['start'].iloc[-1] - \
            pd.offsets.Day(self.CLUSTER_DATA_DAYS)
        data = data.loc[data.index > start_ts]
        # Apply DBSCAN.
        # TODO: Experiment on the options.
        d = sklearn.cluster.DBSCAN(eps=20, min_samples=5, metric='euclidian',
                                   metric_params=None, algorithm='brute',
                                   leaf_size=30, n_jobs=None)
        d.fit(data[['active', 'reactive']].values)
        # We need to make sure that devices that were already detected
        # previously keep the same name.
        if not self._appliances:
            # First time we detect appliances
            for a in d.components_:
                name = "Appliance %d" % (self._appliance_id)
                self.appliances[self._appliance_id] = \
                    eeris_nilm.appliance.Appliance(self._appliance_id,
                                                   name, a[0], a[1])
                self._appliance_id += 1
        else:
            # First build a temporary list of appliances
            app_id = 0
            appliances = dict()
            for a in d.components_:
                name = "Appliance %d" % (app_id)
                appliances[app_id] = \
                    eeris_nilm.appliance.Appliance(app_id, name, a[0], a[1])
            # Map to previous
            self._match_appliances(appliances)
        # Sync live appliances
        self._appliances_live = self._appliances.copy()
        # Set timestamp
        self._last_clustering_ts = self._buffer.index[-1]

    def _match_appliances(self, a_from, t=20):
        """
        Helper function to match between two dictionaries of appliances.

        Parameters
        ----------
        a_from : Dictionary of eeris_nilm.appliance.Appliance objects that we
        need to map

        t : Beyond this threshold the devices are considered different

        Returns
        -------
        out : None
        The function updates the _appliances variable

        """
        # TODO: This is a greedy implementation with many to one mapping. Is
        # this correct? Should we have an globally optimal strategy instead? To
        # support this, we keep the list of all candidates in the
        # implementation.
        a = dict()
        mapping = dict()
        for k in a_from.keys():
            # Create the list of candidate matches for the k-th appliance
            candidates = []
            for l in self._appliances.keys():
                d = eeris_nilm.appliance.Appliance.distance(a_from[k],
                                                            self._appliances[l])
                if d < t:
                    candidates.append((l, d))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                # Simplest approach. Just get the minimum that is below
                # threshold t
                m = 0
                while m < len(candidates) and candidates[m][0] in mapping:
                    m += 1
                if m < len(candidates):
                    mapping[k] = candidates[m][0]
        # Finally, perform the mapping and update the _appliances class
        # variable.
        for k in a_from.keys():
            if k in mapping.keys():
                m = mapping[k]
                a[m] = self._appliances[m]
            else:
                m = "Unknown appliance %d" % (self._appliance_id)
                a[l] = a_from[k]
                self._appliance_id += 1
        self._appliances = a

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
                name = 'Unknown appliance %d' % (self._appliance_id)
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
            name = 'Unknown appliance %d' % (self._appliance_id)
            a = eeris_nilm.appliance.Appliance(self._appliance_id, name,
                                               signature=e)
            # Does this look like a known appliance that isn't already matched?
            candidates = self._match_appliances_live(a)
            if not candidates:
                # New appliance
                self._appliances_live[a.appliance_id] = a
                self.live.insert(0, a)
                self._appliance_id += 1
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
        # Normalization
        self._normalize()
        self._preprocess()
        # Edge detection
        self._detect_edges_hart()
        # Edge matching
        self._match_edges_hart()
        # Clustering
        # 1. Static clustering option
        # TODO: Turn this into a thread, if we decide to keep it after all.
        if self._last_clustering_ts is not None:
            td = self._last_processed_ts - self._last_clustering_ts
            if td.seconds/3600.0/24 > self.CLUSTER_STEP_DAYS:
                self._static_cluster()
        else:
            td = self._last_processed_ts - self._start_ts
            if td.seconds/3600.0/24 > self.CLUSTER_FIRST_DAYS:
                self._static_cluster()
        # Live update
        self._update_live()
        self._match_edges_hart_live()
