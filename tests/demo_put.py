"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import numpy as np
import pandas as pd
import datetime
import pickle
import pymongo
import sys
import timeit
import json
import matplotlib.pyplot as plt

from tests import eco
from eeris_nilm.hart85_eeris import Hart85eeris


class NILMtest(object):
    """
    Class to handle streamed data processing for NILM in eeRIS. It also maintains a
    document of the state of appliances in an installation.
    """

    # How often (every n PUT requests) should we store the document persistently?
    STORE_PERIOD = 10

    def __init__(self, mdb):
        # Add state variables as needed
        self._mdb = mdb
        self._models = dict()
        self._put_count = dict()
        self._prev = 0.0

    def on_put(self, stream, inst_id):
        """
        This method receives new measurements and processes them to update the state of
        the installation. This is where most of the work is being done.

        req.stream must contain a json serialized Pandas dataframe (with at least
        timestamp as index, active, reactive power and voltage as columns).
        """
        if len(stream) > 0:
            data = pd.read_json(stream)
        inst_iid = int(inst_id)

        # Update the models
        if (inst_iid not in self._models.keys()):
            inst_doc = self._mdb.models.find_one({"meterId": inst_iid})
            if inst_doc is None:
                modelstr = pickle.dumps(Hart85eeris(installation_id=inst_iid))
                inst_doc = {'meterId': inst_iid,
                            'lastUpdate': str(datetime.datetime.now()),
                            'model_hart': modelstr}
                self._mdb.models.insert_one(inst_doc)
            self._models[inst_iid] = pickle.loads(inst_doc['model_hart'])
            self._put_count[inst_iid] = 0
        model = self._models[inst_iid]
        # TODO update this dummy data processing steps to detect edges, devices and
        # update the "Live" collection.
        model.data = data
        model.detect_edges()
        # Add NILM steps
        step = data.shape[0]
        # This whole conditional block is not needed. It should be replaced with an update
        # of the "Live" collection at MongoDB.
        if model.online_edge_detected and not model.on_transition:
            before = np.array([self._prev] * (step // 2))
            after = np.array([self._prev + model.online_edge[0]] * (step - step // 2))
            est_y = np.concatenate((before, after))
        elif model.on_transition:
            est_y = np.array([self._prev] * step)
        else:
            est_y = np.array([model.running_avg_power[0]] * step)
            self._prev = model.running_avg_power[0]
        # Store data if needed, and prepare response.
        self._put_count[inst_iid] += 1
        if (self._put_count[inst_iid] % self.STORE_PERIOD == 0):
            # Persistent storage
            modelstr = pickle.dumps(model)
            self._mdb.models.update_one({'meterId': inst_iid},
                                        {'$set':
                                         {'meterId': inst_iid,
                                          'lastUpdate': str(datetime.datetime.now()),
                                          'model_hart': modelstr
                                          }
                                         })
        resp = '{ "edge_detected": %s, "edge_size": [%f, %f], "est_y": %s }' % \
               (str(model.online_edge_detected).lower(),
                model.online_edge[0],
                model.online_edge[1],
                np.array2string(est_y, separator=', '))
        return resp


dburl = "mongodb://localhost:27017/"
dbname = "eeris"
mclient = pymongo.MongoClient(dburl)
dblist = mclient.list_database_names()
if dbname in dblist:
    mdb = mclient[dbname]
else:
    sys.stderr.write('ERROR: Database ' + dbname + ' not found. Exiting.')

n = NILMtest(mdb)
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
        r_str = n.on_put(data.to_json(), 1)
        r = json.loads(r_str)
        est_y.append(np.array(r['est_y']))
        n_requests += 1
        current_sec += step
    end = timeit.timeit()
    print("Performed %d put requests in %f seconds" % (n_requests, start - end))
    y = np.concatenate(est_y)
    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(power.iloc[i:i + plot_step].index,
             power.iloc[i:i + plot_step]['active'].values)
    plt.plot(power.iloc[i:i + plot_step].index, y, 'r')
    fig.autofmt_xdate()
    plt.pause(0.05)
