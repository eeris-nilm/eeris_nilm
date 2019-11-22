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

# Demo of edge detection without REST service implementation
import sys
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from eeris_nilm.datasets import eco

p = 'tests/data/01_sm_csv/01'
date_start = '2012-06-10T19:00'
date_end = '2012-06-10T23:59'
step = 5
plot_step = 600
nilm_url = 'http://localhost:8000/nilm/1'
nnst_url = 'http://localhost:8000/installation/1/model'
# nilm_url = 'http://localhost:9999/nilm/1'
# nnst_url = 'http://localhost:9999/installation/1/model'
current_sec = 0

# Prepare, by deleting possible existing models
# r = requests.delete(inst_url)
# print('Result of model delete: %s' % (r.text))

# Prepare data and plots
phase_list, power = eco.read_eco(p, date_start, date_end)
fig, ax = plt.subplots()
plt.grid()
yest = np.array([], dtype='float64')

# Main loop
for i in range(0, power.shape[0], step):
    n_requests = 0
    data = power.iloc[i:i + step]
    r = requests.put(nilm_url, data=data.to_json())
    if r.status_code != 200:
        print("Something went wrong, received HTTP %d" % (r.status_code))
        sys.exit(1)
    resp = json.loads(r.text)
    yest = np.concatenate([yest, np.array(resp['_yest'])])
    r = requests.get(nilm_url)
    if r.status_code != 200:
        print("Something went wrong, received HTTP %d" % (r.status_code))
    live = r.text
    print(live)
    n_requests += 1
    current_sec += step
    if i > plot_step:
        yest = yest[-plot_step:]
        ax.clear()
        plt.grid()
        d = power.iloc[i-plot_step+step:i+step]
        plt.plot(d.index, d['active'].values, 'b')
        plt.plot(d.index, yest, 'r')
        fig.autofmt_xdate()
        plt.pause(0.05)
