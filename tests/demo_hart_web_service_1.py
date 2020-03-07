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

# Demo of real-time "live" detection, as well as clustering and activation
# history. This version tests the following endpoints at the server:
#
# POST /nilm/{installation_id}/clustering
# GET /nilm/{installation_id}/activations
# {PUT,GET, DELETE} /nilm/{installation_id}/
# DELETE /installation/{installation_id}/model
#
# Before running this script, start MongoDB and run
# start_app_wsgi.sh to start the services at localhost, after making sure that
# 'threads=False' in wsgi.py.

import sys
import time
import requests
import threading
import json
import atexit
from eeris_nilm.datasets import eco

# Uncomment these two lines if you want to see detailed debug logs
# import logging
# logging.basicConfig(level=logging.DEBUG)


# Function to initiate clustering at regular intervals
def request_clustering(nilm_url, stop_event, interval=60):
    clustering_url = nilm_url + '/clustering'
    while not stop_event.is_set():
        time.sleep(interval)
        r = requests.post(clustering_url)
        if r.status_code != 200:
            print("Clustering: Received HTTP %d" % (r.status_code))
    print('Clustering requests thread stopping.')


# Function to request appliance activations at regular intervals
def request_activations(nilm_url, stop_event, interval=90):
    activations_url = nilm_url + '/activations'
    while not stop_event.is_set():
        time.sleep(interval)
        r = requests.get(activations_url)
        if r.status_code != 200:
            print("Activations: Received HTTP %d" % (r.status_code))
        resp = json.loads(r.text)
        print("Activations response: %s" % (resp))
    print('Activations requests thread stopping.')


p = 'tests/data/01_sm_csv/01'
date_start = '2012-06-15T00:00'
date_end = '2012-06-15T23:59'
step = 5
plot_step = 600
nilm_url = 'http://localhost:8000/nilm/1'
inst_url = 'http://localhost:8000/installation/1/model'

# Prepare data
phase_list, power = eco.read_eco(p, date_start, date_end)

# Delete model at database and the server memory (optional)
delete = True
if delete:
    r = requests.delete(nilm_url)
    if r.status_code != 200:
        print("Could not delete model from server memory, exiting")
        sys.exit(1)
    r = requests.delete(inst_url)
    if r.status_code != 200:
        print("Could not delete model at database, exiting")
        sys.exit(1)

# Initiate clustering and activation threads
stop_event = threading.Event()
atexit.register(stop_event.set)
cluster_requests_thread = threading.Thread(target=request_clustering,
                                           name='clustering',
                                           args=(nilm_url, stop_event))
cluster_requests_thread.start()
activation_requests_thread = threading.Thread(target=request_activations,
                                              name='activations',
                                              args=(nilm_url, stop_event))
activation_requests_thread.start()


# Main loop
for i in range(0, power.shape[0], step):
    n_requests = 0
    data = power.iloc[i:i + step]
    r = requests.put(nilm_url, data=data.to_json())
    if r.status_code != 200:
        print("Something went wrong, received HTTP %d" % (r.status_code))
        sys.exit(1)
    resp = json.loads(r.text)
    print("PUT response: %s" % (resp))
    r = requests.get(nilm_url)
    if r.status_code != 200:
        print("Something went wrong, received HTTP %d" % (r.status_code))
    live = r.text
    print("GET response (live): %s" % (live))
    time.sleep(0.01)
stop_event.set()
cluster_requests_thread.join()
activation_requests_thread.join()
