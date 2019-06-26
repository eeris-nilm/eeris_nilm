"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import numpy as np
import pandas as pd


class Hart85eeris():
    """ Modified implementation of Hart's NILM algorithm. """
    # NOMINAL_VOLTAGE = 230.0
    NOMINAL_VOLTAGE = 240.0  # Just a test
    BUFFER_SIZE_SECONDS = 600
    MAX_WINDOW_DAYS = 100
    MAX_NUM_STATES = 1000
    MAX_DISPLAY_SECONDS = 10 * 3600
    # These could be parameters
    STEADY_THRESHOLD = 15
    SIGNIFICANT_EDGE = 50
    STEADY_SAMPLES_NUM = 5
    MATCH_THRESHOLD = 35

    def __init__(self, installation_id):
        # State / helper variables
        # TODO: remove the variables that are not needed
        self.on_transition = False
        self.running_edge_estimate = np.array([0.0, 0.0])
        self._steady_count = 0
        self._edge_count = 0
        self._previous_steady_power = np.array([0.0, 0.0])
        self.running_avg_power = np.array([0.0, 0.0])
        self._last_measurement = 0.0
        self._last_processed_ts = None
        self._data = None
        self._buffer = None
        self._samples_count = 0
        self._idx = None
        self._yest = np.array([], dtype='float64')
        self._ymatch = None
        self._ymatch_live = None
        # Installation id (is this necessary?)
        self.installation_id = installation_id
        # List of states and transitions detected so far.
        # self.steady_states = np.array([], dtype=np.float64).reshape(0, 2)
        # self.edges = np.array([], dtype=np.float64).reshape(0, 2)
        self._steady_states = pd.DataFrame([],
                                           columns=['start', 'end', 'active', 'reactive'])
        self._edges = pd.DataFrame([], columns=['start', 'end', 'active',
                                                'reactive', 'mark'])
        self._matches = pd.DataFrame([], columns=['start', 'end', 'active', 'reactive'])
        self._edge_start_ts = None
        self._edge_end_ts = None
        self._steady_start_ts = None
        self._steady_end_ts = None
        # For online edge detection
        self.online_edge_detected = False
        self.online_edge = np.array([0.0, 0.0])
        self._appliance_id = 0
        self.live = pd.DataFrame(columns=['name', 'active', 'reactive'])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data variable. It also pre-processes the data and updates
        a sliding window of BUFFER_SIZE_SECONDS of data. Current version assumes 1Hz
        sampling frequency.
        """
        if data.shape[0] == 0:
            raise ValueError('Empty dataframe')
        # Check time difference
        duration = (data.index[-1] - data.index[0]).days
        if duration > self.MAX_WINDOW_DAYS:
            # Do not process the window, it's too long.
            raise ValueError('Data duration too long')
        # Normalize power measurements with voltage
        data = self._normalise(data)
        if self._buffer is None:
            self._buffer = data.copy()
        else:
            # Data concerning past dates update the buffer
            self._buffer = self._buffer.append(data)  # More effective alternatives?
        # Round timestamps
        self._buffer.index = self._buffer.index.round('1s')
        # Remove possible duplicate entries (keep the last entry), based on timestamp
        self._buffer = self._buffer.loc[~self._buffer.index.duplicated(keep='last')]
        # Resample to 1s
        self._buffer = self._buffer.asfreq('1S', method='pad')
        # Keep only the last BUFFER_SIZE_SECONDS of the buffer
        start_ts = self._buffer.index[-1] - \
            pd.offsets.Second(self.BUFFER_SIZE_SECONDS - 1)
        self._buffer = self._buffer[self._buffer.index >= start_ts]
        if self._last_processed_ts is None:
            self._data = self._buffer
            self._idx = self._buffer.index[0]
            self._steady_start_ts = self._idx
        else:
            self._idx = self._last_processed_ts + 1 * self._buffer.index.freq
            self._data = self._buffer.loc[self._idx:]
        # TODO: Handle N/As and zero voltage.
        # TODO: Unit tests with all the unusual cases

    def _normalise(self, data):
        """
        Resample data to 1s and normalise power with voltage measurements. Drop missing
        values.
        """
        # Normalisation. Raise active power to 1.5 and reactive power to 2.5. See Hart's
        # 1985 paper for an explanation.
        # Just making sure...
        r_data = data.copy()
        r_data.loc[:, 'active'] = data['active'] * \
            np.power((self.NOMINAL_VOLTAGE / data['voltage']), 1.5)
        r_data.loc[:, 'reactive'] = data['reactive'] * \
            np.power((self.NOMINAL_VOLTAGE / data['voltage']), 2.5)
        return r_data

    def _detect_edges(self):
        """
        TODO: Advanced identification of steady states and transitions based on active and
        reactive power.
        """
        pass

    def _detect_edges_hart(self):
        """
        Simplified edge detector, based on Hart's algorithm.
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
        # These are helper variables, to have a single np.concatenate/vstack at the end
        edge_list = [self._edges]
        steady_list = [self._steady_states]
        for i in range(data.shape[0]):
            current_ts = self._idx + i * self._buffer.index.freq
            diff = data[i, :] - prev
            prev = data[i, :]
            if any(np.fabs(diff) > self.STEADY_THRESHOLD):
                if not self.on_transition:
                    # Starting transition
                    # Do not register previous edge if it started from 0 (it may be due to
                    # missing data).
                    if any(self._previous_steady_power > np.finfo(float).eps):
                        previous_edge = self.running_avg_power - \
                            self._previous_steady_power
                        if any(np.fabs(previous_edge) > self.SIGNIFICANT_EDGE):
                            edge_df = pd.DataFrame(data={'start': self._edge_start_ts,
                                                         'end': self._edge_end_ts,
                                                         'active': previous_edge[0],
                                                         'reactive': previous_edge[1],
                                                         'mark': False},
                                                   index=[0])
                            edge_list.append(edge_df)
                            # self._edges = np.append(self._edges, previous_edge) Too slow
                    self._steady_end_ts = current_ts
                    steady_df = pd.DataFrame(data={'start': self._steady_start_ts,
                                                   'end': self._steady_end_ts,
                                                   'active': self.running_avg_power[0],
                                                   'reactive': self.running_avg_power[1]},
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
                    # Either the transition continues, or it is the start of a steady
                    # period.
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
                self.running_avg_power += 1.0 / (self._steady_count + 1.0) * data[i, :]
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
        Clustering step of Hart's method. Here it is implemented as a static clustering
        step that runs periodically, mapping previous devices to the new device names.

        NOT IMPLEMENTED
        """
        pass

    def _dynamic_cluster(self):
        """
        Dynamic clustering step, as proposed by Hart.

        NOT IMPLEMENTED
        """
        pass

    def _clean_edges_buffer(self):
        """
        Clean-up edges buffer. This removes matched edges from the buffer, but may also
        remove edges that have remained in the buffer for a very long time, perform other
        sanity checks etc. It's currently work in progress.
        """
        self._edges.drop(self._edges.loc[self._edges['mark']].index, inplace=True)
        # TODO:
        # Sanity check 1: Matched power should be lower than consumed power

    def _match_edges_hart(self):
        """
        On/Off matching using edges (as opposed to clusters). This is the method
        implemented by Hart for the two-state load monitor (it won't work directly for
        multi-state appliances). It is implemented as close as possible to Hart's original
        paper (1985). The approach is rather simplistic and can lead to serious errors.
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
            # Determine matching thresholds
            e1 = e.iloc[i][['active', 'reactive']].values.astype(np.float64)
            if e1[0] >= 1000:
                t_active = 0.05 * e1[0]
            else:
                t_active = self.MATCH_THRESHOLD
            if np.fabs(e1[1]) >= 1000:
                t_reactive = 0.05 * e1[1]
            else:
                t_reactive = self.MATCH_THRESHOLD
            T = np.array([t_active, t_reactive])
            for j in range(i+1, len(e)):
                # Edge has been marked before or is positive
                if e.iloc[j]['active'] >= 0 or e.iloc[j]['mark']:
                    continue
                # Do they match?
                e2 = e.iloc[j][['active', 'reactive']].values.astype(np.float64)
                # if all(np.fabs(e1 + e2) < T):
                # For now, using only active power
                # TODO: Improve using reactive with rules
                if np.fabs(e1[0] + e2[0]) < T[0]:
                    # Match
                    edge = (np.fabs(e1) + np.fabs(e2)) / 2.0
                    # Ideally we should keep both start and end times for each edge
                    df = pd.DataFrame({'start': e.iloc[i]['start'],
                                       'end': e.iloc[j]['start'],
                                       'active': edge[0],
                                       'reactive': edge[1]}, index=[0])
                    self._matches = self._matches.append(df, ignore_index=True, sort=True)
                    # Get the 'mark' column.
                    c = e.columns.get_loc('mark')
                    e.iat[i, c] = True
                    e.iat[j, c] = True
                    break
        # Perform sanity checks and clean edges.
        self._clean_edges_buffer()

    def _match_edges_hart_live(self):
        """
        Adaptation of Hart's edge matching algorithm to support the "live" display of
        eeRIS.
        """
        if not self.online_edge_detected:
            if self.live.empty:
                return
            if self.on_transition:
                last = self.live.index[-1]
                self.live.at[last, 'final'] = True
                return
            # Update last edge
            last = self.live.index[-1]
            if not self.live.loc[last, 'final']:
                self.live.at[last, 'active'] = \
                    self.running_avg_power[0] - self._previous_steady_power[0]
                self.live.at[last, 'reactive'] = \
                    self.running_avg_power[1] - self._previous_steady_power[1]
            return
        e = self.online_edge
        if all(np.fabs(e) < self.SIGNIFICANT_EDGE):
            # Although no edge is added, previous should be finalised
            if not self.live.empty:
                last = self.live.index[-1]
                self.live.at[last, 'final'] = True
            return
        if e[0] > 0:
            # Appliance cycle start
            df = pd.DataFrame({'name': 'Appliance %d' % (self._appliance_id),
                               'active': e[0],
                               'reactive': e[1],
                               'previous_active': self._previous_steady_power[0],
                               'previous_reactive': self._previous_steady_power[1],
                               'final': False},
                              index=[0])
            self.live = pd.concat([df, self.live], ignore_index=True, sort=True)
            self._appliance_id += 1
            return
        # Appliance cycle stop. Does it match against previous edges?
        for i in reversed(range(self.live.shape[0])):
            e0 = self.live.iloc[i][['active', 'reactive']].values.astype(np.float64)
            if e[0] <= -1000:
                t_active = 0.05 * e[0]
            else:
                t_active = self.MATCH_THRESHOLD
            if np.fabs(e[1]) >= 1000:
                t_reactive = 0.05 * e[1]
            else:
                t_reactive = self.MATCH_THRESHOLD
            T = np.fabs(np.array([t_active, t_reactive]))
            # Match only with active power for now
            if np.fabs(e[0] + e0[0]) < T[0]:
                # Match. Remove device from live
                self.live.drop(self.live.index[i], inplace=True)
                # Finalise all existing live edges and break
                for i in self.live.index:
                    self.live.at[i, 'final'] = True
                break

    def _match_helper(self, start, end, active):
        end_sec_inv = (self._last_processed_ts - end).seconds
        if end_sec_inv > self.MAX_DISPLAY_SECONDS:
            return
        start_sec_inv = (self._last_processed_ts - start).seconds
        if start_sec_inv > self.MAX_DISPLAY_SECONDS:
            start_sec_inv = self.MAX_DISPLAY_SECONDS
        self._ymatch[-start_sec_inv:-end_sec_inv] += active

    def _update_live(self):
        """
        Provide information for display at the eeRIS "live" screen. Preliminary version.
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

    def update(self):
        """
        Wrapper to sequence of operations for model update
        """
        self._detect_edges_hart()
        self._match_edges_hart()
        self._update_live()
        self._match_edges_hart_live()
