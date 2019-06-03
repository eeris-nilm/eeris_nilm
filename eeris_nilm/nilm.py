import falcon
import json
import pandas as pd
import datetime
from .hart85_eeris import Hart85

# Note: This code needs modifications for parallel operation
# Note: Add try/catch, error checks etc


class NILM(object):
    # After how many updates should we store persistently?
    # To do: Perhaps do this based on timestamp instead of counts?
    COUNT_THRESHOLD = 1
    # Data window (in seconds) to store for each installation as a buffer
    BUFFER_WINDOW = 300

    def __init__(self, mdb):
        self._mdb = mdb
        self._models = dict()
        self._buffer = dict()
        self._data_count = dict()
        # Load variables

    def on_get(self, req, resp, inst_id):
        """ Handles GET requests """
        resp.body = self._mdb.find_one({"inst_id": inst_id})
        resp.status = falcon.HTTP_200

    def on_put(self, req, resp, inst_id):
        """ Handles PUT requests """
        # Read the data
        if req.content_length:
            data = pd.read_json(req.stream)
        inst_iid = int(inst_id)

        # Update data buffer
        if (inst_iid not in self._buffer.keys()):
            self._buffer[inst_iid] = data
        else:
            self._buffer[inst_iid].append(data)
        start_ts = self._buffer[inst_iid].index[0] - \
            pd.offsets.Second(self.BUFFER_WINDOW)
        self._buffer[inst_iid] = self._buffer[inst_iid][self._buffer.index >= start_ts]

        # Update steady states and transients
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
