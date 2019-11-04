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

# TODO: Load ground truth as well (?)
import numpy as np
import pandas as pd
import os.path
import datetime


def read_eco(path, date_start, date_end):
    """
    Parse ECO csv files.

    Parameters
    ----------
    path : Path to the directory of ECO csv files

    date_start : Same as file name (e.g., '2012-06-01T09:00')

    date_end : As above


    Returns
    -------
    data : Pandas dataframe with measurements including 'active', 'reactive',
    'voltage, 'phase_angle', 'current'
    """

    # d = datetime.date.fromisoformat(date_start)  # Only valid in python 3.7,
    # dropped for now.
    d_start = datetime.datetime.strptime(date_start, '%Y-%m-%dT%H:%M')
    d = d_start
    d_end = datetime.datetime.strptime(date_end, '%Y-%m-%dT%H:%M')
    phase_df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    while d <= d_end:
        print('ECO: Loading building ' + os.path.basename(path) + ', time ' +
              d.strftime('%Y-%m-%dT%H:%M'))
        f = os.path.join(path, d.strftime('%Y-%m-%d') + '.csv')
        df = pd.read_csv(f, header=None, index_col=False,
                         names=[i for i in range(1, 17)], dtype=np.float32)
        # From nilmtk ECO dataset converter
        phases = []
        for phase in range(1, 4):
            df_phase = df.loc[:, [1 + phase, 5 + phase, 8 + phase, 13 + phase]]
            power = df_phase.loc[:, (1 + phase, 13 + phase)].values
            reactive = power[:, 0] * np.tan(power[:, 1] * np.pi / 180)
            df_phase['Q'] = reactive
            # No timezone
            df_phase.index = pd.date_range(
                start=d, periods=3600 * 24, freq='S')
            column_names = {
                1 + phase: 'active',
                5 + phase: 'current',
                8 + phase: 'voltage',
                13 + phase: 'phase_angle',
                'Q': 'reactive',
            }
            df_phase.columns = [column_names[col] for col in df_phase.columns]
            power_active = df_phase['active']
            tmp_before = np.size(power_active)
            df_phase = df_phase[power_active != -1]
            power_active = df_phase['active']
            tmp_after = np.size(power_active)
            if tmp_before != tmp_after:
                print('Removed missing measurements - Size before: ' +
                      str(tmp_before) + ', size after:' + str(tmp_after))
            phases.append(df_phase)
            phase_df_list[phase - 1] = \
                pd.concat([phase_df_list[phase - 1], df_phase])
        d += datetime.timedelta(days=1)
        if d > d_end:
            d = d_end
    agg_df = pd.DataFrame([], columns=['active', 'reactive', 'voltage'])
    agg_df['active'] = phase_df_list[0]['active'] + \
        phase_df_list[1]['active'] + \
        phase_df_list[2]['active']
    agg_df['reactive'] = phase_df_list[0]['reactive'] + \
        phase_df_list[1]['reactive'] + phase_df_list[2]['reactive']
    agg_df['voltage'] = (phase_df_list[0]['voltage'] +
                         phase_df_list[1]['voltage'] +
                         phase_df_list[2]['voltage']) / 3.0
    for i in range(len(phase_df_list)):
        phase_df_list[i] = \
            phase_df_list[i].loc[(phase_df_list[i].index >= d_start) &
                                 (phase_df_list[i].index <= d_end)]
    agg_df = agg_df.loc[(agg_df.index >= d_start) & (agg_df.index <= d_end)]
    return (phase_df_list, agg_df)
