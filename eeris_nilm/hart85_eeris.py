import numpy as np
import pandas as pd
import datetime
# from nilmtk.feature_detectors.steady_states import find_steady_states


class Hart85eeris():
    """ to do """
    NOMINAL_VOLTAGE = 230.0
    MAX_ONLINE_HOURS = 8
    MAX_WINDOW_DAYS = 100

    def __init__(self, installation_id, steady_states_list=pd.DataFrame(),
                 transients_list=pd.DataFrame()):
        # Initialise the necessary metadata
        self.installation_id = installation_id
        self.steady_states_list = steady_states_list
        self.transients_list = transients_list
        self._data

    def preprocessing(self, data):
        """
        Data sanity checks and fixes before processing.
        """
        # Check time difference
        duration = (data.index[-1] - data.index[0]).astype('timedelta64[h]')
        if duration > self.MAX_WINDOW_DAYS:
            # Do not process the window.
            raise ValueError('Data duration too long')
        idx = pd.interval_range(start=data.index[0], end=data.index[-1],
                                freq='S')
        self._data = data.reindex(index=idx, copy=True)

    def _normalisation(self):
        """
        Normalise power with voltage measurements
        """
        self._data['active'] = (self._data['active'] *
                                (self.NOMINAL_VOLTAGE / self._data['voltage'])
                                ** 2)
        self._data['reactive'] = (self._data['reactive'] *
                                  (self.NOMINAL_VOLTAGE /
                                   self._data['voltage']) ** 2)
        self._data.dropna()

    def _edge_detection_online(self, n_seconds=0, models=None):
        """
        Identify if there are edges in the last n_seconds of data. If yes, then
        match them against a set of existing models to guess whether a device
        was activated or not.
        """
        pass

    def _edge_detection(self):
        """
        The regular edge detection step of the hart method. Identify steady
        states and transitions based on active and reactive power.
        """
        pass

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

    def detect_online(self, data):
        ndata = self._normalisation(data)
        self._edge_detection(ndata)
