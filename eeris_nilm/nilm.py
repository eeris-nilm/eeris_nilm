import falcon
import json
import pandas as pd
import datetime
from .hart85_eeris import Hart85

# Note: This code needs modifications for parallel operation
# Note: Add try/catch, error checks etc


class NILM(object):
    COUNT_THRESHOLD = 1

    def __init__(self, mdb):
        self._mdb = mdb
        self._models = dict()
        self._data_count = dict()
        # Load variables

    def on_get(self, req, resp, inst_id):
        """ Handles GET requests """
        resp.body = self._mdb.find_one({"inst_id": inst_id})
        resp.status = falcon.HTTP_200

    def on_put(self, req, resp, inst_id):
        """ Handles PUT requests """
        if req.content_length:
            data = pd.read_json(req.stream)
        inst_iid = int(inst_id)
        if (inst_iid not in self._models.keys()):
            inst_doc = self._mdb.installations.find_one({"meterId": inst_iid})
            if inst_doc == None:
                inst_doc = {"meterId": inst_iid, "insertDate": str(datetime.date.today()),
                            "steadyStates": pd.DataFrame().to_json(), "transients": pd.DataFrame().to_json()}
                self._mdb.installations.insert_one(inst_doc)
            ss_list = pd.read_json(inst_doc['steadyStates'])
            tr_list = pd.read_json(inst_doc['transients'])
            self._models[inst_iid] = Hart85(inst_iid, steady_states_list=ss_list,
                                            transients_list=tr_list)
            self._data_count[inst_iid] = 0
        model = self._models[inst_iid]
        print(data)
        print(type(data))
        model.test_hart(data)
        self._data_count[inst_iid] += 1
        if (self._data_count[inst_iid] > self.COUNT_THRESHOLD):
            # Persistent storage
            self._mdb.installations.update_one({"meterId": inst_iid},
                                               {'$set': {"steadyStates": model.steady_states_list.to_json(),
                                                         "transients": model.transients_list.to_json()}
                                                })
            self._data_count[inst_iid] = 0
        resp.status = falcon.HTTP_200  # Default status
