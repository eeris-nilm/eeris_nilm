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

import pandas as pd
from numpy import linalg as LA


class Appliance():
    """
    Unsupervised appliance model. Includes signatures, usage data and statistics
    as well as other data useful for identification through NILM.
    """

    def __init__(self, appliance_id, name, category, signature=None):
        self.appliance_id = appliance_id
        self.name = name
        self.category = category
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
