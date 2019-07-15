"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import falcon
import pandas as pd
import datetime as dt
import pickle
import json

from eeris_nilm.algorithms import hart


class NILM(object):
    """
    Class to handle streamed data processing for NILM in eeRIS. It also
    maintains a document of the state of appliances in an installation.
    """

    # How often (every n PUT requests) should we store the document
    # persistently?
    STORE_PERIOD = 10

    def __init__(self, mdb, response='cenote'):
        # Add state variables as needed
        self._mdb = mdb
        self._models = dict()
        self._put_count = dict()
        self._prev = 0.0
        self._response = response

    def _prepare_response_body(self, model):
        """ Wrapper function """
        body = None
        if self._response == 'cenote':
            body = self._prepare_response_body_cenote(model)
        elif self._response == 'debug':
            body = self._prepare_response_body_debug(model)
        return body

    def _prepare_response_body_cenote(self, model):
        """
        Prepare a response according to the specifications of Cenote.
        Check https://authecesofteng.github.io/cenote/ for more information.
        """
        ts = dt.datetime.now().timestamp() * 1000
        payload = []
        for i in range(len(model.live)):
            app = model.live[i]
            app_d = {"name": app.name,
                     "active": ("%.2f") % (app.signature[0]),
                     "reactive": ("%.2f") % (app.signature[1])}
            d = {"data": app_d, "timestamp": ts}
            payload.append(d)
        body_d = {"installation_id": str(model.installation_id),
                  "payload": payload}
        body = json.dumps(body_d)
        return body

    def _prepare_response_body_debug(self, model, lret=5):
        """
        DO NOT USE. NEEDS REFACTORING


        Helper function to prepare response body. lret is the length of the
        returned _yest array (used for development/debugging, ignore it in
        production).
        """
        return None  # TODO Refactor according to new "live".
        live = model.live[['name', 'active', 'reactive']].to_json()
        ts = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')
        body = '''{
        "timestamp": "%s",
        "appliances": %s,
        "edge_detected": %s,
        "edge_size": [%.2f, %.2f],
        "_yest": %s }''' % (ts,
                            live,
                            str(model.online_edge_detected).lower(),
                            model.online_edge[0],
                            model.online_edge[1],
                            model._yest[-lret:].tolist())
        return body

    def on_get(self, req, resp, inst_id):
        """
        On get, the service returns a document describing the status of a
        specific installation.
        """
        inst_iid = int(inst_id)
        # Load the model, if not loaded already
        if (inst_iid not in self._models.keys()):
            inst_doc = self._mdb.models.find_one({"meterId": inst_iid})
            if inst_doc is None:
                raise falcon.HTTPBadRequest("Installation does not exist",
                                            "You have requested data from " +
                                            "an installation that does not" +
                                            "exist")
            else:
                self._models[inst_iid] = pickle.loads(inst_doc['modelHart'])
        model = self._models[inst_iid]
        resp.body = self._prepare_response_body(model)
        resp.status = falcon.HTTP_200

    def on_put(self, req, resp, inst_id):
        """
        This method receives new measurements and processes them to update the
        state of the installation. This is where most of the work is being
        done.
        req.stream must contain a json serialized Pandas dataframe (with at
        least timestamp as index, active, reactive power and voltage as
        columns).
        """
        if req.content_length:
            data = pd.read_json(req.stream)
        else:
            raise falcon.HTTPBadRequest("No data provided", "No data provided")
        inst_iid = int(inst_id)

        # Load the model, if not available
        if (inst_iid not in self._models.keys()):
            inst_doc = self._mdb.models.find_one({"meterId": inst_iid})
            if inst_doc is None:
                modelstr = pickle.dumps(
                    hart.Hart85eeris(installation_id=inst_iid))
                inst_doc = {'meterId': inst_iid,
                            'lastUpdate':
                            dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'debugInstallation': True,
                            'modelHart': modelstr}
                self._mdb.models.insert_one(inst_doc)
            self._models[inst_iid] = pickle.loads(inst_doc['modelHart'])
            self._put_count[inst_iid] = 0
        model = self._models[inst_iid]
        # Process the data
        model.update(data)
        # Store data if needed, and prepare response.
        self._put_count[inst_iid] += 1
        if (self._put_count[inst_iid] % self.STORE_PERIOD == 0):
            # Persistent storage
            modelstr = pickle.dumps(model)
            self._mdb.models.update_one({'meterId': inst_iid},
                                        {'$set':
                                         {'meterId': inst_iid,
                                          'lastUpdate': str(dt.datetime.now()),
                                          'debugInstallation': True,
                                          'modelHart': modelstr
                                          }
                                         })
        # resp.body = 'OK'
        # lret = data.shape[0]
        resp.body = self._prepare_response_body(model)
        resp.status = falcon.HTTP_200  # Default status
