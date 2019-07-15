"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

from numpy import linalg as LA


class Appliance():
    """
    Unsupervised appliance model. Includes signatures, usage statistics and
    other data useful for identification through NILM. Work in progress.
    """

    def __init__(self, appliance_id, name, signature=None):
        self.appliance_id = appliance_id
        self.name = name
        self.num_states = 2  # Default is two-state appliance
        self.signature = signature
        self.final = False  # We are allowed to modify signature
        self.verified = False  # Not sure it is actually a new appliance
        self.p_signature = None  # Previous signature (for running average)

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
        # Running average
        self.signature = 0.9 * self.p_signature + 0.1 * self.signature
        self.p_signature = self.signature
