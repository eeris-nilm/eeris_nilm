"""
A simple demo of the edge detection functrionality of Hart85eeris.


Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

# Demo of edge detection without REST service implementation
import numpy as np
import pandas as pd
import os.path
import datetime
import matplotlib.pyplot as plt
from eeris_nilm.hart85_eeris import Hart85eeris


def read_eco(path, date_start, date_end):
    """
    Parse ECO csv files.

    Parameters
    ----------
    path : Path to the directory of ECO csv files

    date_start : Same as file name (e.g., '2012-06-01')

    date_end : As above


    Returns
    -------
    data : Pandas dataframe with measurements including 'active', 'reactive', 'voltage,
    'phase_angle', 'current'

    """

    # d = datetime.date.fromisoformat(date_start)  # Only valid in python 3.7, dropped for
    # now.
    d = datetime.datetime.strptime(date_start, '%Y-%m-%d')
    d_end = datetime.datetime.strptime(date_end, '%Y-%m-%d')
    phase_df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    while d <= d_end:
        print('Processing ' + d.strftime('%Y-%m-%d'))
        f = os.path.join(path, d.strftime('%Y-%m-%d') + '.csv')
        df = pd.read_csv(f, header=None, index_col=False, names=[i for i in range(1, 17)],
                         dtype=np.float32)
        # From nilmtk ECO dataset converter
        phases = []
        for phase in range(1, 4):
            df_phase = df.loc[:, [1 + phase, 5 + phase, 8 + phase, 13 + phase]]
            power = df_phase.loc[:, (1 + phase, 13 + phase)].values
            reactive = power[:, 0] * np.tan(power[:, 1] * np.pi / 180)
            df_phase['Q'] = reactive
            # No timezone
            df_phase.index = pd.date_range(
                start=d.isoformat(), periods=3600 * 24, freq='S')
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
            phase_df_list[phase - 1] = pd.concat([phase_df_list[phase - 1], df_phase])
        d += datetime.timedelta(days=1)
    agg_power = pd.DataFrame([], columns=['active', 'reactive'])
    agg_power['active'] = phase_df_list[0]['active'] + phase_df_list[1]['active'] + \
        phase_df_list[2]['active']
    agg_power['reactive'] = phase_df_list[0]['reactive'] + \
        phase_df_list[1]['reactive'] + phase_df_list[2]['reactive']
    return (phase_df_list, agg_power)


p = 'tests/data/01_sm_csv/01'
date_start = '2012-06-19'
date_end = '2012-06-19'
step = 5
plot_step = 3600
model = Hart85eeris(installation_id=1)
current_sec = 0

phase_list, power = read_eco(p, date_start, date_end)
prev = power['active'].iloc[0]
for i in range(0, power.shape[0], plot_step):
    est_y = list()
    for j in range(i, min(i + plot_step, power.shape[0]), step):
        print("Seconds %d to %d\n" % (current_sec, current_sec + step - 1))
        data = power.iloc[j:j + step][['active', 'reactive']]
        model.data = data
        model.edge_detection()
        if model.online_edge_detected and not model.on_transition:
            est_y.append(np.array([prev] * (step // 2)))
            est_y.append(np.array([prev + model.online_edge[0]] * (step - step // 2)))
        elif model.on_transition:
            est_y.append(np.array([prev] * step))
        else:
            est_y.append(np.array([model.running_avg_power[0]] * step))
            prev = model.running_avg_power[0]
        current_sec += step
    print(model.edges)
    print(model.steady_states)
    y = np.concatenate(est_y)
    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(power.iloc[i:i + plot_step].index,
             power.iloc[i:i + plot_step]['active'].values)
    plt.plot(power.iloc[i:i + plot_step].index, y, 'r')
    fig.autofmt_xdate()
    plt.pause(0.05)
