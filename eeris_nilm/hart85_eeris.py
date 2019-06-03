import pandas as pd
import numpy as np
from nilmtk.feature_detectors.steady_states import find_steady_states


NOMINAL_VOLTAGE = 230.0


class Hart85():
    """ to do """

    def __init__(self, installation_id, steady_states_list=pd.DataFrame(),
                 transients_list=pd.DataFrame()):
        # Initialise the necessary metadata
        self.installation_id = installation_id
        self.steady_states_list = steady_states_list
        self.transients_list = transients_list

    # To do
    def check_data(data):
        """ Data sanity checks before processing """
        pass
        
    def _normalisation(self, data):
        """
        Perform data normalisation and return a copy of the normalised power
        columns of the original data
        """
        ndata = data.copy()
        ndata['active'] = (data['active'] *
                           (NOMINAL_VOLTAGE / data['voltage']) ** 2)
        ndata['reactive'] = data['reactive'] * (NOMINAL_VOLTAGE /
                                                data['voltage']) ** 2
        ndata.dropna()
        return ndata

    # Code for this is adapted from nilmtk
    def _edge_detection(self, data):
        """
        Identify periods of steady power consumption and of changing power
        consumption.
        """
        data_p = data[['active', 'reactive']]
        
        
        x, y = find_steady_states(data_p)
        # Do this with queue rotation
        self.steady_states_list = self.steady_states_list.append(x)
        self.transients_list = self.transients_list.append(y)
        print(self.steady_states_list)
        print(self.transients_list)

    def test_hart(self, data):
        ndata = self._normalisation(data)
        self._edge_detection(ndata)
