"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import numpy as np

class Appliance():
    """
    Unsupervised appliance model. Includes signatures, usage statistics and
    other data useful for identification through NILM. Work in progress.
    """

    def __init__(self, appliance_id, name, active, reactive=None):
        self._id = appliance_id
        self._name = name
        self.num_states = 2  # Default is two-state appliance
        self._active = active  # Active power consumption when on
        self._reactive = reactive  # Reactive power consumption when on

    @property
    def appliance_id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    @property
    def reactive(self):
        return self._reactive

    @reactive.setter
    def reactive(self, value):
        self._reactive = value

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
        return np.sqrt((app1.active - app2.active) ** 2 +
                       (app1.reactive - app2.reactive) ** 2)
