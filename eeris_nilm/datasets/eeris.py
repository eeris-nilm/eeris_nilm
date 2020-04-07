"""
Copyright 2020 Christos Diou

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

import os
import pandas as pd
import numpy as np
import datetime


# To be tested. Do not use.
def read_eeris(path_prefix, date_start=None, date_end=None):
    """
    Parse CSV files exported by eeRIS.

    Parameters
    ----------
    path_prefix : String. Prefix to the data files (including path)

    date_start : String. Starting date of data to load, in format
    (%Y-%m-%dT%H:%M). For example '2011-04-18T17:00'.

    date_end : String. Same as above, for the end date.

    Returns
    -------
    data: A pandas dataframe  with timestamp index and measurements
    including 'active', 'reactive', 'current', 'voltage'.
    """
    dtype = {"GW_MAC": str,
             "Device_MAC": str,
             "timestamp": pd.Timestamp,
             "date": pd.Timestamp,
             "time": pd.Timestamp,
             "vltA": np.float64,
             "curA": np.float64,
             "pwrA": np.float64,
             "rpwrA": np.float64,
             "frq": np.float64}
    d_start = datetime.datetime.strptime(date_start, '%Y-%m-%dT%H:%M')
    d = d_start
    start_sec = d.hour * 3600 + d.minute * 60 + d.second
    end_sec = 24 * 3600
    d_end = datetime.datetime.strptime(date_end, '%Y-%m-%dT%H:%M')
    df_list = []
    while d <= d_end:
        print('Loading building ' + os.path.basename(path_prefix) + ', time ' +
              d.strftime('%Y-%m-%dT%H:%M'))
        f = os.path.join(path_prefix, d.strftime('%Y-%m-%d') + '.csv')
        if not os.path.exists(f):
            d += datetime.timedelta(days=1)
            d = d.replace(hour=0, minute=0)
            continue
        if d.date() == d_end.date():
            # Just making sure, this is redundant
            d = d.replace(hour=0, minute=0)
            end_sec = d_end.hour * 3600 + d_end.minute * 60 + d_end.second
        df = pd.read_csv(f, index_col=False, dtype=dtype)
        df = df.iloc[start_sec:end_sec]
        df_list.append(df)
        d += datetime.timedelta(days=1)
        d = d.replace(hour=0, minute=0)
        start_sec = 0
    data = pd.concat(df_list)
    mapper = {"pwrA": "active",
              "rpwrA": "reactive",
              "curA": "current",
              "vltA": "voltage"}
    data.rename(mapper=mapper, axis='columns', inplace=True)
    data.set_index("timestamp", drop=True, inplace=True)
    return data
