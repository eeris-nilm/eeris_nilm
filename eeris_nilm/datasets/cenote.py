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

# Under development, do not use

import pandas as pd
import numpy as np


def read_cenote(path, installation_id, date_start=None, date_end=None):
    """
    Parse CSV file exported from cenote by eeRIS.

    Parameters
    ----------
    path : String. Path to the dataset file (single CSV)

    installation_id: String. The object id of the installation

    date_start : String. Starting date of data to load, in format
    (%Y-%m-%dT%H:%M). For example '2011-04-18T17:00'.

    date_end : String. Same as above, for the end date.

    Returns
    -------
    data: A pandas dataframe  with timestamp index and measurements
    including 'active', 'reactive', 'current', 'voltage'.
    """
    dtype = {"installationid": str,
             "active": np.float64,
             "reactive": np.float64,
             "current": np.float64,
             "voltage": np.float64,
             "frq": np.float64,
             "uuid": str,
             "cenote$created_at": pd.Timestamp,
             "cenote$timestamp": pd.Timestamp,
             "cenote$id": pd.Timestamp,
             "powerl1": np.float64,
             "phaseanglecurrentvoltagel1": np.float64}
    cenote = pd.read_csv(path, index_col='cenote$timestamp',
                         sep=',', dtype=dtype)
    data = cenote[['active', 'reactive', 'current', 'voltage']]
    return data
