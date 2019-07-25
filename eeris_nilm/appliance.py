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
    _appliance_id = 0

    def __init__(self, name=None, appliance_id=None, signature=None):
        self.num_states = 2  # Default is two-state appliance
        self.signature = signature
        self.final = False  # We are allowed to modify signature
        self.verified = False  # Not sure it is actually a new appliance
        self.p_signature = signature  # Previous signature (for running average)
        if appliance_id is not None:
            type(self)._appliance_id = appliance_id
        self.appliance_id = type(self)._appliance_id
        type(self)._appliance_id += 1
        if name is None:
            name = "Unknown appliance %d" % (self.appliance_id)
        else:
            self.name = name

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
