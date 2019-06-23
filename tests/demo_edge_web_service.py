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
import json
import matplotlib.pyplot as plt
# from tests import eco
import eco

p = 'tests/data/01_sm_csv/01'
date_start = '2012-06-10T19:00'
date_end = '2012-06-10T23:59'
step = 5
plot_step = 600
base_url = 'http://localhost:8000/nilm/1'
current_sec = 0

phase_list, power = eco.read_eco(p, date_start, date_end)
fig, ax = plt.subplots()
plt.grid()
y_est = np.array([], dtype='float64')
y_match = np.array([], dtype='float64')
for i in range(0, power.shape[0], step):
    n_requests = 0
    data = power.iloc[i:i + step]
    r = requests.put(base_url, data=data.to_json())
    if r.status_code != 200:
        print("Something went wrong, received HTTP %d" % (r.status_code))
    resp = json.loads(r.text)
    y_est = np.concatenate([y_est, np.array(resp['y_est'])])
    y_match = np.concatenate([y_match, np.array(np.array(resp['y_match']))])
    r = requests.get(base_url)
    if r.status_code != 200:
        print("Something went wrong, received HTTP %d" % (r.status_code))
    live = r.text
    print(live)
    n_requests += 1
    current_sec += step
    if i > plot_step:
        y_est = y_est[-plot_step:]
        y_match = y_match[-plot_step:]
        ax.clear()
        plt.grid()
        d = power.iloc[i-plot_step+step:i+step]
        plt.plot(d.index, d['active'].values, 'b')
        plt.plot(d.index, y_est, 'r')
        plt.plot(d.index, y_match, 'm')
        fig.autofmt_xdate()
        plt.pause(0.05)
