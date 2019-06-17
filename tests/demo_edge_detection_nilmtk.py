"""
A simple demo of the edge detection functrionality of Hart85eeris. 


Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

# Demo of edge detection without REST service implementation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nilmtk import DataSet
from eeris_nilm.hart85_eeris import Hart85eeris

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
