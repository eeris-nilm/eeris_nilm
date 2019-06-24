"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import numpy as np
import pandas as pd


class Appliance():
    """ Unsupervised appliance model. Includes signatures, usage statistics and other data
    useful for identification through NILM. Work in progress.
    """

    def __init__(self, appliance_id):
        self._id = appliance_id
        self.num_states = 2  # Default is two-state appliance
