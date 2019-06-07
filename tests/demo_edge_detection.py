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
import matplotlib.dates as mdates
from eeris_nilm.hart85_eeris import Hart85eeris


def read_eco(path, inst_id, date_start, date_end):
    """
    Parse ECO csv files.

    Parameters
    ----------
    path : Path the the directory of the ECO csv files

    inst_id : Installation id (in the form 01, 02, ...)

    date_start : Same as file name (e.g., '2012-06-01')

    date_end : As above


    Returns
    -------
    data : Pandas dataframe with measurements including 'active', 'reactive', 'voltage,
    'phase_angle', 'current'

    """
    date_list = [date_start]
    d = datetime.date.fromisoformat(date_start)
    d_end = datetime.date.fromisoformat(date_end)
    while d <= d_end:
        f = os.path.join(path, "%02d" % (inst_id), d.isoformat(), '.csv')
        df = pd.read_csv(f, header=None, index_col=False,
                         names=['powerallphases',
                                'powerl1',
                                'powerl2',
                                'powerl3',
                                'currentneutral',
                                'currentl1',
                                'currentl2',
                                'currentl3',
                                'voltagel1',
                                'voltagel2',
                                'voltagel3',
                                'phaseanglevoltagel2l1',
                                'phaseanglevoltagel3l1',
                                'phaseanglecurrentvoltagel1',
                                'phaseanglecurrentvoltagel2',
                                'phaseanglecurrentvoltagel3'],
                         dtype=np.float32)
        # From nilmtk dataset converter
        phases = []
        for phase in range(1, 4):
            df_phase = df.loc[:, [1 + phase, 5 + phase, 8 + phase, 13 + phase]]
            power = df_phase.loc[:, (1 + phase, 13 + phase)].values
            reactive = power[:, 0] * np.tan(power[:, 1] * np.pi / 180)
            df_phase['Q'] = reactive
            # No timezone
            df_phase.index = pd.DatetimeIndex(start=d.isoformat(), freq='S', periods=3600 * 24)
            column_names = {
                1 + phase: 'active',
                5 + phase: 'current',
                8 + phase: 'voltage',
                13 + phase: 'phase_angle',
                'Q': 'reactive',
            }
            df_phase.columns = [column_names[col] for col in df_phase.columns]


eco = DataSet('tests/ECO_1.h5')
eco.set_window(start='2012-09-01 07:00', end='2012-09-01 10:59')
chunksize = 3600
step = 5
plot_length = 3000
elec = eco.buildings[1].elec
mains = elec.mains()

model = Hart85eeris(installation_id=1)
current_sec = 0
for chunk in mains.load(chunksize=chunksize):
    chunk = chunk[[('power', 'active'), ('power', 'reactive'), ('voltage', ''),
                   ('phase_angle', ''), ('current', '')]]
    chunk.set_axis(['active', 'reactive', 'voltage', 'phase_angle', 'current'],
                   axis='columns', inplace=True)
    est_y = list()
    prev = chunk['active'].iloc[0]
    for i in range(0, chunk.shape[0], step):
        print("Seconds %d to %d\n" % (current_sec, current_sec + step - 1))
        data = chunk.iloc[i:i + step][['active', 'reactive']]
        model.data = data
        model._edge_detection()
        if model.online_edge_detected or model._on_transition:
            est_y.append(np.array([prev] * (step // 2)))
            est_y.append(np.array([prev + model.online_edge[0]] * (step - step // 2)))
        else:
            est_y.append(np.array([model._running_avg_power[0]] * step))
            prev = model._running_avg_power[0]
        current_sec += step
    print(model.edges)
    print(model.steady_states)
    y = np.concatenate(est_y)
    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(chunk.index, chunk['active'].values)
    plt.plot(chunk.index, y, 'r')
    fig.autofmt_xdate()
    # ax.fmt_xdata = mdates.DateFormatter('%H:%M:%S')
    # ax.fmt_xdata = mdates.AutoDateFormatter()
    plt.pause(0.05)
