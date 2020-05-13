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
import datetime


def read_eeris(path_prefix, date_start=None, date_end=None):
    """
    Parse CSV files exported by eeRIS.

    Parameters
    ----------
    path_prefix : String. Prefix to the data files (including path)

    date_start : String. Starting date of data to load, in format
    (%Y-%m-%dT%H:%M). For example '2020-01-01T17:00'.

    date_end : String. Same as above, for the end date.

    Returns
    -------
    data: A pandas dataframe  with timestamp index and measurements
    including 'active', 'reactive', 'current', 'voltage'.
    """
    parse_dates = ["date", "time"]
    d_start = datetime.datetime.strptime(date_start, '%Y-%m-%dT%H:%M')
    d = d_start
    start_sec = d.hour * 3600 + d.minute * 60 + d.second
    end_sec = 24 * 3600
    d_end = datetime.datetime.strptime(date_end, '%Y-%m-%dT%H:%M')
    df_list = []
    while d <= d_end:
        bldng = os.path.basename(os.path.dirname(path_prefix))
        print('Loading building ' + bldng + ', time ' +
              d.strftime('%Y-%m-%dT%H:%M'))
        f = path_prefix + d.strftime('%Y-%m-%d') + '.csv'
        if not os.path.exists(f):
            d += datetime.timedelta(days=1)
            d = d.replace(hour=0, minute=0)
            continue
        if d.date() == d_end.date():
            # Just making sure, this is redundant
            d = d.replace(hour=0, minute=0)
            end_sec = d_end.hour * 3600 + d_end.minute * 60 + d_end.second
        df = pd.read_csv(f, index_col='timestamp', parse_dates=parse_dates)
        df.index = pd.to_datetime(df.index, unit='s', origin='unix')
        df = df.iloc[start_sec:end_sec]
        df_list.append(df)
        d += datetime.timedelta(days=1)
        d = d.replace(hour=0, minute=0)
        start_sec = 0
    data = pd.concat(df_list)
    # Set the appropriate column names and remove index name
    mapper = {"pwrA": "active",
              "rpwrA": "reactive",
              "curA": "current",
              "vltA": "voltage"}
    data.rename(mapper=mapper, axis='columns', inplace=True)
    data.index = data.index.rename(None)
    return data
