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
    NOMINAL_VOLTAGE = 230.0
    BUFFER_SIZE_SECONDS = 1200
    MAX_WINDOW_DAYS = 100
    MAX_NUM_STATES = 1000
    # These could be parameters
    STEADY_THRESHOLD = 15
    SIGNIFICANT_EDGE = 30
    STEADY_SAMPLES_NUM = 3

    def __init__(self, installation_id):
        # State variables
        # TODO: remove the variables that are not needed
        self._on_transition = False
        self._running_edge_estimate = np.array([0.0, 0.0])
        self._steady_count = 0
        self._edge_count = 0
        self._previous_steady_power = np.array([0.0, 0.0])
        self._running_avg_power = np.array([0.0, 0.0])
        self._last_measurement = 0.0
        self._last_processed_ts = None
        self._data = None
        self._buffer = None
        self._nbuffer = None
        self._samples_count = 0
        # Installation id (is this necessary?)
        self.installation_id = installation_id
        # List of states and transitions detected so far.
        self.steady_states = np.array([], dtype=np.float64).reshape(0, 2)
        self.edges = np.array([], dtype=np.float64).reshape(0, 2)
        # For online edge detection
        self.online_edge_detected = False
        self.online_edge = np.array([0.0, 0.0])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data variable. It also pre-processes the data and updates
        a sliding window of BUFFER_SIZE_SECONDS of data.
        """
        self._data = data
        # Check time difference
        duration = (self._data.index[-1] - self._data.index[0]).days
        if duration > self.MAX_WINDOW_DAYS:
            # Do not process the window, it's too long.
            raise ValueError('Data duration too long')
        # Set to 1s sampling rate.
        idx = pd.date_range(start=data.index[0], end=data.index[-1], freq='S')
        self._data = data.reindex(index=idx, copy=True)
        if self._buffer is None:
            self._buffer = self.data.copy()
        else:
            self._buffer = self._buffer.append(self._data)  # More effective alternatives?
            # Bug up to pandas 0.24, loses freq. Use inferred_freq instead.
            self._buffer.index.freq = self._buffer.index.inferred_freq
        # Remove possible duplicate entries (keep the last entry), based on timestamp
        self._buffer = self._buffer.loc[~self._buffer.index.duplicated(keep='last')]
        # Keep only the last BUFFER_SIZE_SECONDS of the buffer
        start_ts = self._buffer.index[-1] - pd.offsets.Second(self.BUFFER_SIZE_SECONDS-1)
        self._buffer = self._buffer[self._buffer.index >= start_ts]
        # Numpy buffer array
        self._nbuffer = self._buffer.values
        # TODO: Handle N/As and zero voltage

    def _normalisation(self):
        """
        Normalise power with voltage measurements
        """
        self._data['active'] = (self._data['active'] * (self.NOMINAL_VOLTAGE /
                                                        self._data['voltage']) ** 2)
        self._data['reactive'] = (self._data['reactive'] * (self.NOMINAL_VOLTAGE /
                                                            self._data['voltage']) ** 2)
        # TODO: Handle this at the setter.
        self._data.dropna()

    def _edge_detection(self):
        """
        Identify steady states and transitions based on active and reactive power.
        """
        self.online_edge_detected = False
        self.online_edge = np.array([0.0, 0.0])
        if self._last_processed_ts is None:
            data = self._buffer.values
            prev = data[0, :]
        else:
            idx = self._last_processed_ts + 1*self._buffer.index.freq
            prev = self._buffer.loc[self._last_processed_ts].values
            data = self._buffer.loc[idx:].values
        # d = data.diff()
        # d.drop(d.index[0])
        # These are helper variables, to have a single np.concatenate/vstack at the end
        edge_list = [self.edges]
        steady_list = [self.steady_states]
        for i in range(data.shape[0]):
            diff = data[i, :] - prev
            prev = data[i, :]
            if any(np.fabs(diff) > self.STEADY_THRESHOLD):
                if not self._on_transition:
                    # Starting transition
                    # Do not register previous edge if it started from 0 (it may be due to
                    # missing data).
                    if any(self._previous_steady_power > np.finfo(float).eps):
                        previous_edge = self._running_avg_power - \
                                        self._previous_steady_power
                        if any(np.fabs(previous_edge) > self.SIGNIFICANT_EDGE):
                            edge_list.append(previous_edge)
                            # self.edges = np.append(self.edges, previous_edge) # Too slow
                    self._previous_steady_power = self._running_avg_power
                    steady_list.append(self._running_avg_power)
                    # self.steady_states = np.append(self.steady_states,
                    # self._running_avg_power)
                    self._running_avg_power = np.array([0.0, 0.0])
                    self._steady_count = 0
                    self._edge_count += 1
                    self._running_edge_estimate = diff
                    self._on_transition = True
                else:
                    # Either the transition continues, or it is the start of a steady
                    # period.
                    self._edge_count += 1
                    self._running_edge_estimate += diff
                    self._running_avg_power = data[i, :]
                    self._steady_count = 1
            else:
                # Update running average
                self._running_avg_power *= self._steady_count / \
                                           (self._steady_count + 1.0)
                self._running_avg_power += 1.0 / (self._steady_count + 1.0) * data[i, :]
                self._steady_count += 1
                if self._on_transition:
                    # We are in the process of finishing a transition
                    self._running_edge_estimate += diff
                    if self._steady_count >= self.STEADY_SAMPLES_NUM:
                        self._on_transition = False
                        self.online_edge_detected = True
                        self.online_edge = self._running_edge_estimate
                        self._edge_count = 0
            self._samples_count += 1
        # Update lists
        self.edges = np.vstack(edge_list)
        self.steady_states = np.vstack(steady_list)
        # Update last processed
        self._last_processed_ts = self._buffer.index[-1]
        self._last_measurement = self._buffer.iloc[-1]

    def _clustering(self):
        """
        Clustering step of the hart method.
        """
        pass

    def _matching(self):
        """
        On/Off matching of the hart method
        """
        pass

    def _guess_type(self):
        """
        Guess the appliance type using an unnamed hart model
        """
        pass

    def detect_online(self, data):
        ndata = self._normalisation(data)
        self._edge_detection(ndata)

    # # Code for this is adapted from nilmtk
    # def _edge_detection(self, data):
    #     """
    #     Identify periods of steady power consumption and of changing power
    #     consumption.
    #     """
    #     data_p = data[['active', 'reactive']]
    #     x, y = find_steady_states(data_p)
    #     # Do this with queue rotation
    #     self.steady_states_list = self.steady_states_list.append(x)
    #     self.transients_list = self.transients_list.append(y)
    #     print(self.steady_states_list)
    #     print(self.transients_list)
