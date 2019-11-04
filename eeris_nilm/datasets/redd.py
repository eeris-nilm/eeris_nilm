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
import os.path
import datetime


def date_parser(x):
    return datetime.datetime.fromtimestamp(float(x))


def read_redd(path, date_start, date_end, get_channels=True):
    """Parse REDD data files.

    Parameters
    ----------
    path : String. Path to the dataset files of a specific house (e.g.,
    [dataset_root]/house_1)

    date_start : String. Starting date of data to load, in format
    (%Y-%m-%dT%H:%M). For example '2011-04-18T17:00'.

    date_end : String. Same as above, for the end date.

    Returns
    -------
    (data, labels): A pair (data, labels) where
    data: Dictionary of the form {channel_id: df} of pandas dataframes with
    timestamp index and measurements including 'active', 'reactive',
    'voltage'. The 'reactive' is a dummy column (all zeros) and 'voltage' is a
    constant value (230V).
    labels: Pandas dataframe with index corresponding to the channel_id and the
    column 'label' indicating the label of the  channels in the
    dataset. An additional label 'mains' is also added combining the two 'mains'
    labels present in the houses (to simulate a single 'mains').

    """
    # Load labels
    labels_path = os.path.join(path, 'labels.dat')
    print('Loading data from %s' % (labels_path))
    # DEBUG: Dates are not being read correctly here.
    labels = pd.read_csv(labels_path, header=None, index_col=0, sep=' ',
                         names=['label'])
    data = {}
    for ch, la in labels.iterrows():
        if get_channels is False and la != 'mains':
            continue
        print('Loading channel %d (%s)' % (ch, la['label']))
        filepath = os.path.join(path, ('channel_%d.dat') % ch)
        data[ch] = pd.read_csv(filepath, header=None, index_col=0, sep=' ',
                               names=['active'], parse_dates=True,
                               date_parser=date_parser)
    # Create an extra dataframe with the sum of the two mains measurements in
    # the house. Channels 1 and 2 are always the 'mains' channels.
    data['mains'] = data[1]
    data['mains']['active'] = data[1]['active'] + data[2]['active']
    # Insert fake reactive power and voltage columns.
    data['mains'].insert(1, 'reactive', 0.0)
    data['mains'].insert(1, 'voltage', 230.0)

    # Filter by date
    d_start = datetime.datetime.strptime(date_start, '%Y-%m-%dT%H:%M')
    d_end = datetime.datetime.strptime(date_end, '%Y-%m-%dT%H:%M')
    for k in data.keys():
        data[k] = data[k].loc[(data[k].index >= d_start) &
                              (data[k].index <= d_end)]
    if get_channels:
        return (data, labels)
    else:
        return (data, None)
