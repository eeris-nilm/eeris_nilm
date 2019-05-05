import falcon
import json
import pandas as pd


# Note: This code needs modifications for parallel operation
class Installation(object):
    def __init__(self, mdb):
        self._mdb = mdb
        self._buffer = {}
        # Load variables

    def on_put(self, req, resp):
        """ Handles PUT requests """
        if req.content_length:
            doc = json.load(req.stream)
        inst = doc['inst_id']
        data = pd.read_json(doc['data'])
        self._hart85(inst, data)
        resp.status = falcon.HTTP_200  # Default status

    # Interfaces to the disaggregation algorithms
    def _hart85(self, inst, data):
        self._update_clustering(inst, data)
        self._disaggregate(inst, data)
        return

    def _update_clustering(self, inst, data):
        return

    def _disaggregate(self, inst, data):
        return
