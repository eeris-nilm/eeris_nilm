"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import falcon
import numpy as np
import pandas as pd
import datetime
import pickle

from .hart85_eeris import Hart85eeris


class NILM(object):
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

    def on_get(self, req, resp, inst_id):
        """
        On get, the service returns a document describing the status of a specific
        installation.
        """
        # TODO
        pass

    def on_put(self, req, resp, inst_id):
        """
        This method receives new measurements and processes them to update the state of
        the installation. This is where most of the work is being done.

        req.stream must contain a json serialized Pandas dataframe (with at least
        timestamp as index, active, reactive power and voltage as columns).
        """
        if req.content_length:
            data = pd.read_json(req.stream)
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
        resp.body = '{ "edge_detected": %s, "edge_size": [%f, %f], "est_y": %s }' % \
                    (str(model.online_edge_detected).lower(),
                     model.online_edge[0],
                     model.online_edge[1],
                     np.array2string(est_y, separator=', '))
        resp.status = falcon.HTTP_200  # Default status
