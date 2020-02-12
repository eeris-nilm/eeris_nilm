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

import falcon
import pandas as pd
import datetime as dt
import dill
import json
import threading
import logging
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
        self._model_lock = dict()

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
        # ts = dt.datetime.now().timestamp() * 1000
        if model.last_processed_ts is not None:
            ts = model.last_processed_ts.timestamp() * 1000
        else:
            payload = []
            body_d = {"installation_id": str(model.installation_id),
                      "payload": payload}
            return json.dumps(body_d)
        payload = []
        # Insert background
        if model.background_active < 10000.0:
            app_d = {"_id": '000000000000000000000001',
                     "name": "Background",
                     "type": "background",
                     "status": True,
                     "active": model.background_active,
                     "reactive": 0.0}
            d = {"data": app_d, "timestamp": ts}
            payload.append(d)
        for i in range(len(model.live)):
            app = model.live[i]
            app_d = {"_id": app.appliance_id,
                     "name": app.name,
                     "type": app.category,
                     "status": True,
                     "active": app.signature[0, 0],
                     "reactive": app.signature[0, 1]}
            d = {"data": app_d, "timestamp": ts}
            payload.append(d)
        # We ignore residuals under 5 Watts.
        if model.residual_live[0] > 5.0 and model.background_active < 10000:
            app_d = {"_id": '000000000000000000000002',
                     "name": "Other",
                     "type": "residual",
                     "status": True,
                     "active": model.residual_live[0],
                     "reactive": model.residual_live[1]}
            d = {"data": app_d, "timestamp": ts}
            payload.append(d)
        body_d = {"installation_id": str(model.installation_id),
                  "payload": payload}
        try:
            body = json.dumps(body_d)
        except (ValueError, TypeError):
            logging.debug(body_d['installation_id'])
            logging.debug(body_d['payload'])
            for k in body_d['payload']:
                logging.debug(body_d[k])
            raise

        # For debugging only (use of logging package not necessary)
        print(body)
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

    def _load_model(self, inst_id):
        # Load the model, if not loaded already
        if (inst_id not in self._models.keys()):
            inst_doc = self._mdb.models.find_one({"meterId": inst_id})
            if inst_doc is None:
                raise falcon.HTTPBadRequest("Installation does not exist",
                                            "You have requested data from " +
                                            "an installation that does not" +
                                            "exist")
            else:
                self._models[inst_id] = dill.loads(inst_doc['modelHart'])
                self._model_lock[inst_id] = threading.Lock()
        return self._models[inst_id]

    def on_get(self, req, resp, inst_id):
        """
        On getn, the service returns a document describing the status of a
        specific installation.
        """
        model = self._load_model(inst_id)
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
        # Load or create the model, if not available
        if (inst_id not in self._models.keys()):
            inst_doc = self._mdb.models.find_one({"meterId": inst_id})
            if inst_doc is None:
                modelstr = dill.dumps(hart.Hart85eeris(installation_id=inst_id))
                inst_doc = {'meterId': inst_id,
                            'lastUpdate':
                            dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'debugInstallation': True,
                            'modelHart': modelstr}
                self._mdb.models.insert_one(inst_doc)
            self._models[inst_id] = dill.loads(inst_doc['modelHart'])
            self._model_lock[inst_id] = threading.Lock()
            self._put_count[inst_id] = 0
        model = self._models[inst_id]
        # Process the data
        self._model_lock[inst_id].acquire()
        model.update(data)
        self._model_lock[inst_id].release()
        # Store data if needed, and prepare response.
        self._put_count[inst_id] += 1
        if (self._put_count[inst_id] % self.STORE_PERIOD == 0):
            # Persistent storage
            modelstr = dill.dumps(model)
            self._mdb.models.update_one({'meterId': inst_id},
                                        {'$set':
                                         {'meterId': inst_id,
                                          'lastUpdate': str(dt.datetime.now()),
                                          'debugInstallation': True,
                                          'modelHart': modelstr
                                          }
                                         })
        # resp.body = 'OK'
        # lret = data.shape[0]
        resp.body = self._prepare_response_body(model)
        # resp.status = falcon.HTTP_200  # Default status
        resp.status = '200'

    def on_delete(self, req, resp, inst_id):
        """
        On delete, the service flushes the requested model from memory
        """
        # Remove the model, if it is loaded
        if (inst_id in self._models.keys()):
            del self._models[inst_id]
            del self._model_lock[inst_id]
        resp.status = falcon.HTTP_200

    def on_post_clustering(self, req, resp, inst_id):
        """
        Starts a clustering thread on the target model
        """
        # Load the model, if not loaded already
        model = self._load_model(inst_id)
        self._model_lock[inst_id].acquire()
        if model.force_clustering():
            resp.status = falcon.HTTP_200
        else:
            # Conflict
            resp.status = falcon.HTTP_409
        self._model_lock[inst_id].release()

    def on_get_activations(self, req, resp, inst_id):
        """
        Requests the list of activations for the appliances of an installation.
        """
        # Load the model, if not loaded already
        model = self._load_model(inst_id)
        payload = []
        self._model_lock[inst_id].acquire()
        for a_k, a in model.appliances:
            for row in a.activations.itertuples():
                # Energy consumption in kWh
                consumption = (row.end - row.start).seconds / 3600.0 * \
                    row.active / 1000.0
                b = {"installationid": inst_id,
                     "deviceid": a.appliance_id,
                     "start": row.start.timestamp() * 1000,
                     "end": row.end.timestamp() * 1000,
                     "consumption": consumption}
                payload.append(b)
        self._model_lock[inst_id].release()
        body = {"payload": payload}
        resp.body = json.dumps(body)
        resp.status = falcon.HTTP_200
