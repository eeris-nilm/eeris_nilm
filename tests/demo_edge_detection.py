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
import eco
from eeris_nilm.hart85_eeris import Hart85eeris


p = 'tests/data/01_sm_csv/01'
date_start = '2012-06-19T00:00'
date_end = '2012-06-19T23:59'
step = 5
plot_step = 3600
model = Hart85eeris(installation_id=1)
current_sec = 0

phase_list, power = eco.read_eco(p, date_start, date_end)
prev = power['active'].iloc[0]
for i in range(0, power.shape[0] // 2, plot_step):
    print("Seconds %d to %d\n" % (current_sec, current_sec + plot_step - 1))
    est_y = list()
    for j in range(i, min(i + plot_step, power.shape[0]), step):
        # print("Seconds %d to %d\n" % (current_sec, current_sec + step - 1))
        data = power.iloc[j:j + step]
        model.data = data
        model._detect_edges_hart()
        model._match_edges_hart()
        if model.online_edge_detected and not model.on_transition:
            est_y.append(np.array([prev] * (step // 2)))
            est_y.append(np.array([prev + model.online_edge[0]] * (step - step // 2)))
        elif model.on_transition:
            # est_y.append(np.array([prev + model.running_edge_estimate[0]] * step))
            est_y.append(np.array([prev] * step))
        else:
            est_y.append(np.array([model.running_avg_power[0]] * step))
            prev = model.running_avg_power[0]
        current_sec += step
    print(model._edges)
    print(model._steady_states)
    y = np.concatenate(est_y)
    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(power.iloc[i:i + plot_step].index,
             power.iloc[i:i + plot_step]['active'].values)
    plt.plot(power.iloc[i:i + plot_step].index, y, 'r')
    fig.autofmt_xdate()
    plt.pause(0.05)
