"""
A simple demo of the edge detection functrionality of Hart85eeris.


Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

# Demo of edge detection without REST service implementation
import numpy as np
import requests
import timeit
import json
import matplotlib.pyplot as plt
#from tests import eco
import eco

p = 'tests/data/01_sm_csv/01'
date_start = '2012-06-19'
date_end = '2012-06-19'
step = 5
plot_step = 3600
base_url = 'http://localhost:8000/nilm/1'
current_sec = 0

phase_list, power = eco.read_eco(p, date_start, date_end)
prev = power['active'].iloc[0]
for i in range(0, power.shape[0], plot_step):
    print("Seconds %d to %d\n" % (current_sec, current_sec + plot_step - 1))
    est_y = list()
    n_requests = 0
    start = timeit.timeit()
    for j in range(i, min(i + plot_step, power.shape[0]), step):
        # print("Seconds %d to %d\n" % (current_sec, current_sec + step - 1))
        data = power.iloc[j:j + step]
        r = requests.put(base_url, data=data.to_json())
        if r.status_code != 200:
            print("Something went wrong, received HTTP %d" % (r.status_code))
        resp = json.loads(r.text)
        est_y.append(np.array(resp['est_y']))
        n_requests += 1
        current_sec += step
    end = timeit.timeit()
    print("Performed %d put requests in %f seconds" % (n_requests, start - end))
    y = np.concatenate(est_y)
    print("concatenation: %f seconds" % (start - end))
    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(power.iloc[i:i + plot_step].index,
             power.iloc[i:i + plot_step]['active'].values)
    plt.plot(power.iloc[i:i + plot_step].index, y, 'r')
    fig.autofmt_xdate()
    plt.pause(0.05)
