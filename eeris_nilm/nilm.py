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
import logging
import time
import datetime
import uwsgi  # Experiment with uwsgi locks. Can we replace with threading?
import threading
import atexit
import requests
# import multiprocessing

from eeris_nilm import utils
from eeris_nilm.algorithms import livehart

# TODO: Refactoring to break into eeris-specific and general-purpose components
# TODO: Support for multiple processes for specified list of installation ids


class NILM(object):
    """
    Class to handle streamed data processing for NILM in eeRIS. It also
    maintains a document of the state of appliances in an installation.
    """

    # How often (every n PUT requests) should we store the model
    # persistently?
    STORE_PERIOD = 10

    def __init__(self, mdb, act_url=None, comp_url=None,
                 response='cenote', thread=False):
        """
        Parameters:
        ----------

        mdb: pymongo.database.Database
        PyMongo database instance for persistent storage and loading

        act_url: URL where detected activations will be submitted. If None, they
        are simply printed on the debug output (through logging.debug)

        comp_url: URL where the data are requested from for model recomputation
        purposes. If None, then no recomputation is possible.

        response: Selected response format. Default is 'cenote' for integration
        with the eeRIS system

        thread: bool
        Whether to start a thread for clustering and activation detection at
        startup
        """
        # Add state variables as needed
        self._mdb = mdb
        self._models = dict()
        self._put_count = dict()
        self._prev = 0.0
        self._response = response
        self._model_lock_id = dict()
        self._model_lock_num = 1
        self._recomputation_active = dict()
        self._recomputation_thread = None
        self._thread = None
        self._activations_url = act_url  # Where to send device activations
        self._computations_url = comp_url  # Recomputation data URL
        if thread:
            self._periodic_thread(period=3600)
            atexit.register(self._cancel_thread)

    def _send_activations(self):
        """
        Send activations to service responsible for storing appliance
        activation events

        Returns:
        -------

        out : dictionary of strings
        The responses of _activations_url, if it has been provided, otherwise a
        list of json objects with the appliance activations for each
        installation. Keys of the dictionary are the installation ids.
        """
        ret = {}
        for inst_id, model in self._models.items():
            if inst_id not in self._model_lock_id.keys():
                self._model_lock_id[inst_id] = self._model_lock_num
                self._model_lock_num += 1
            uwsgi.lock(self._model_lock_id[inst_id])
            payload = []
            for a_k, a in model.appliances:
                # For now, do not mark activations as sent (we'll do that only
                # if data have been successfully sent)
                activations = a.return_new_activations(update_ts=False)
                for row in activations.itertuples():
                    # Energy consumption in kWh
                    # TODO: Correct consumption by reversing power normalization
                    consumption = (row.end - row.start).seconds / 3600.0 * \
                        row.active / 1000.0
                    b = {
                        "data":
                        {
                            "installationid": inst_id,
                            "deviceid": a.appliance_id,
                            "start": row.start.timestamp() * 1000,
                            "end": row.end.timestamp() * 1000,
                            "consumption": consumption,
                            "algorithm_id": model.VERSION
                        }
                    }
                    payload.append(b)
            body = {
                "payload": payload
            }
            # Send stuff
            if self._activations_url is not None:
                resp = requests.post(self._activations_url,
                                     json=json.dumps(body))
                if resp.status_code != falcon.HTTP_200:
                    logging.debug(
                        "Sending of activation data for %s failed: (%d, %s)" %
                        (inst_id, resp.status_code, resp.text)
                    )
                    ret[inst_id] = resp
                else:
                    # Everything went well, mark the activations as sent
                    ret[inst_id] = \
                        json.dumps(a.return_new_activations(update_ts=True))
            else:
                ret[inst_id] = \
                    json.dumps(a.return_new_activations(update_ts=True))
            # Move on to the next installation
            uwsgi.unlock(self._model_lock_id[inst_id])
        return ret

    def _cancel_thread(self):
        """Stop clustering thread when the app terminates."""
        if self._thread is not None:
            self._thread.cancel()

    def _periodic_thread(self, period=3600, clustering=False):
        """
        Start a background clustering thread which will perform clustering in
        all loaded models at regular intervals.

        Parameters
        ----------
        period : int
        Call the clustering thread every 'period' seconds.

        clustering : bool
        Perform clustering or not.
        """
        logging.debug("Starting periodic thread.")
        if clustering:
            self.perform_clustering()
            # Wait 5 minutes, in case clustering is completed within this time.
            time.sleep(300)
        # Send activations
        act_result = self._send_activations()
        logging.debug("Activations report:")
        logging.debug(act_result)
        # Submit new thread
        self._thread = threading.Timer(period, target=self._periodic_thread)

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
                self._model_lock_id[inst_id] = self._model_lock_num
                self._model_lock_num += 1
                self._recomputation_active[inst_id] = False
        return self._models[inst_id]

    def _store_model(self, inst_id):
        """
        Helper function to store a model in the database.  WARNING: This function
        assumes that the model is already loaded. Also, is NOT thread safe,
        never call it directly unless you know what you're doing.
        """
        model = self._models[inst_id]
        modelstr = dill.dumps(model)
        upd = {'$set': {
            'meterId': inst_id,
            'lastUpdate': str(dt.datetime.now()),
            'debugInstallation': True,
            'modelHart': modelstr}
        }
        self._mdb.models.update_one({'meterId': inst_id}, upd)

    def _recompute_model(self, inst_id, start_ts, end_ts, step=6 * 3600):
        """
        Recompute a model from data provided by a service. Variations of this
        routine can be created for different data sources.

        Parameters
        ----------
        inst_id: str
        Installation id whose model we wan to recompute

        start_ts : int
        Start timestamp in seconds since UNIX epoch

        end_ts : int
        End timestamp in seconds since UNIX epoch

        step : int
        Step, in seconds to use for calculations
        """
        # TODO: Take into account naming events in the model
        if self._computations_url is None:
            logging.debug(("No URL has been provided for past data.",
                           "Model re-computation is not supported."))
            return
        if inst_id not in self._model_lock_id.keys():
            self._model_lock_id[inst_id] = self._model_lock_num
            self._model_lock_num += 1
        uwsgi.lock(self._model_lock_id[inst_id])
        self._recomputation_active[inst_id] = True
        # Delete model from memory
        self._models.pop(inst_id, None)
        # Delete model from database
        result = self._mdb.models.delete_many({'meterId': inst_id})
        dcount = int(result.deleted_count)
        if dcount == 0:
            logging.debug("Installation did not exist in the database")
        elif dcount > 1:
            logging.debug("More than one documents deleted in database")
        # Prepare new model
        modelstr = dill.dumps(livehart.LiveHart(installation_id=inst_id))
        inst_doc = {'meterId': inst_id,
                    'lastUpdate':
                    dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'),
                    'debugInstallation': True,
                    'modelHart': modelstr}
        self._mdb.models.insert_one(inst_doc)
        self._models[inst_id] = dill.loads(inst_doc['modelHart'])
        self._put_count[inst_id] = 0
        model = self._models[inst_id]
        # Recomputation loop
        for ts in range(start_ts, end_ts, step):
            url = self._computations_url + inst_id
            # Endpoint expects timestamp in milliseconds since unix epoch
            st = ts * 1000
            if ts + step < end_ts:
                et = (ts + step) * 1000
            else:
                et = end_ts * 1000

            params = {
                "start": st,
                "end": et
            }
            r = requests.get(url, params)
            data = utils.get_data_from_cenote_response(r)
            if data is None:
                continue
            model.update(data)
            self._put_count[inst_id] += 1
            if (self._put_count[inst_id] % self.STORE_PERIOD == 0):
                # Persistent storage
                self._store_model(inst_id)
        uwsgi.unlock(self._model_lock_id[inst_id])
        self._recomputation_active[inst_id] = False

    def on_get(self, req, resp, inst_id):
        """
        On get, the service returns a document describing the status of a
        specific installation.
        """
        if inst_id not in self._model_lock_id.keys():
            self._model_lock_id[inst_id] = self._model_lock_num
            self._model_lock_num += 1
        uwsgi.lock(self._model_lock_id[inst_id])
        model = self._load_model(inst_id)
        logging.debug('WSGI lock (GET)')
        resp.body = self._prepare_response_body(model)
        uwsgi.unlock(self._model_lock_id[inst_id])
        logging.debug('WSGI unlock (GET)')
        time.sleep(0.01)
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
                modelstr = dill.dumps(
                    livehart.LiveHart(installation_id=inst_id)
                )
                inst_doc = {'meterId': inst_id,
                            'lastUpdate':
                            dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'debugInstallation': True,
                            'modelHart': modelstr}
                self._mdb.models.insert_one(inst_doc)
            self._models[inst_id] = dill.loads(inst_doc['modelHart'])
            self._model_lock_id[inst_id] = self._model_lock_num
            self._model_lock_num += 1
            self._recomputation_active[inst_id] = False
            self._put_count[inst_id] = 0
        # Inform caller if recomputation is active
        if self._recomputation_active[inst_id]:
            resp.status = falcon.HTTP_204
            resp.body = "Model recomputation in progress, send data again later"
            return
        if inst_id not in self._model_lock_id.keys():
            self._model_lock_id[inst_id] = self._model_lock_num
            self._model_lock_num += 1
        uwsgi.lock(self._model_lock_id[inst_id])
        model = self._models[inst_id]
        logging.debug('WSGI lock (PUT)')
        # Process the data
        model.update(data)
        uwsgi.unlock(self._model_lock_id[inst_id])
        logging.debug('WSGI unlock (PUT)')
        time.sleep(0.01)
        # Store data if needed, and prepare response.
        self._put_count[inst_id] += 1
        if (self._put_count[inst_id] % self.STORE_PERIOD == 0):
            # Persistent storage
            self._store_model(inst_id)
        # resp.body = 'OK'
        # lret = data.shape[0]
        resp.body = self._prepare_response_body(model)
        resp.status = falcon.HTTP_200  # Default status

    def on_delete(self, req, resp, inst_id):
        """
        On delete, the service flushes the requested model from memory
        """
        # Remove the model, if it is loaded
        if (inst_id in self._models.keys()):
            del self._models[inst_id]
            del self._model_lock_id[inst_id]
        resp.status = falcon.HTTP_200

    def perform_clustering(self):
        """
        Starts clustering thread for all loaded installations
        """
        for inst_id, model in self._models.items():
            # This should never happen
            if inst_id not in self._model_lock_id.keys():
                self._model_lock_id[inst_id] = self._model_lock_num
                self._model_lock_num += 1
            uwsgi.lock(self._model_lock_id[inst_id])
            model.force_clustering(start_thread=True)
            uwsgi.unlock(self._model_lock_id[inst_id])
            time.sleep(5)

    def on_post_clustering(self, req, resp, inst_id):
        """
        Starts a clustering thread on the target model
        """
        # Load the model, if not loaded already
        if inst_id not in self._model_lock_id.keys():
            self._model_lock_id[inst_id] = self._model_lock_num
            self._model_lock_num += 1
        uwsgi.lock(self._model_lock_id[inst_id])
        model = self._load_model(inst_id)
        logging.debug('WSGI lock (clustering)')
        if model.force_clustering(start_thread=True):
            resp.status = falcon.HTTP_200
        else:
            # Conflict
            resp.status = falcon.HTTP_409
        uwsgi.unlock(self._model_lock_id[inst_id])
        logging.debug('WSGI unlock (clustering)')
        time.sleep(0.01)

    def on_get_activations(self, req, resp, inst_id):
        """
        Requests the list of activations for the appliances of an installation.
        """
        # Load the model, if not loaded already
        if inst_id not in self._model_lock_id.keys():
            self._model_lock_id[inst_id] = self._model_lock_num
            self._model_lock_num += 1
        uwsgi.lock(self._model_lock_id[inst_id])
        model = self._load_model(inst_id)
        payload = []
        logging.debug('WSGI lock (activations)')
        for a_k, a in model.appliances.items():
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
        uwsgi.unlock(self._model_lock_id[inst_id])
        logging.debug('WSGI unlock (activations)')
        time.sleep(0.01)
        body = {"payload": payload}
        resp.body = json.dumps(body)
        resp.status = falcon.HTTP_200

    def on_post_recomputation(self, req, resp, inst_id):
        """
        Recompute an installation model based on all available data.

        Request has the following format:
        URL?start=$start_timestamp&end=$end_timestamp$&step=$step_timestamp
        where timestamps are in seconds since unix epoch.

        The old model is discarded and a new model is recomputed based on data
        available between start and end timestamps.
        """
        # Start recomputation thread
        if 'start' not in req.params or \
           'end' not in req.params or \
           'step' not in req.params:
            resp.status = falcon.HTTP_400
            resp.body = "Incorrect query string in request"
            return
        start_ts = int(req.params['start'])
        end_ts = int(req.params['end'])
        step = int(req.params['step'])
        name = "recomputation_%s" % (inst_id)
        self._recomputation_thread = threading.Thread(
            target=self._recompute_model, name=name,
            args=(inst_id, start_ts, end_ts, step)
        )
        self._recomputation_thread.start()
        now = datetime.datetime.now()
        resp.body = "Recomputation thread submitted on %s " % (now)
        resp.status = falcon.HTTP_200  # Default status

    def on_post_start_thread(self, req, resp, inst_id):
        """
        Start periodic computation thread.
        """
        # TODO: Fixed period for now. Move this to the request, if needed.
        self._periodic_thread(period=3600)
        atexit.register(self._cancel_thread)
        resp.status = falcon.HTTP_200

    def on_post_stop_thread(self, req, resp, inst_id):
        """
        Cancels the periodic computation thread.
        """
        self._cancel_thread()
        resp.status = falcon.HTTP_200

    def on_post_appliance_name(self, req, resp, inst_id):
        """
        Sets a new name for an appliance and store the model that has been
        created.
        """
        if 'appliance_id' not in req.params or 'name' not in req.params:
            resp.status = falcon.HTTP_400
            resp.body = "Incorrect query string in request"
            return
        appliance_id = req.params['appliance_id']
        name = req.params['name']
        category = req.params['category']
        if inst_id not in self._model_lock_id.keys():
            self._model_lock_id[inst_id] = self._model_lock_num
            self._model_lock_num += 1
        uwsgi.lock(self._model_lock_id[inst_id])
        model = self._load_model(inst_id)
        prev_name = model.appliances[appliance_id].name
        prev_category = model.appliances[appliance_id].category
        model.appliances[appliance_id].name = name
        model.appliances[appliance_id].category = category
        logging.debug(("Installation: %s. Renamed appliance %s from %s with"
                       "category %s to %s with category %s") %
                      (inst_id, appliance_id, prev_name, prev_category, name, category))
        # Make sure to store the model
        self._store_model(inst_id)
        uwsgi.unlock(self._model_lock_id[inst_id])
        time.sleep(0.01)
        resp.status = falcon.HTTP_200
