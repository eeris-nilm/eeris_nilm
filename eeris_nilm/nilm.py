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
import numpy as np
import datetime as dt
import dill
import json
import paho.mqtt.client as mqtt
import logging
import time
import datetime
import threading
import atexit
import requests
# import multiprocessing

from eeris_nilm import utils
from eeris_nilm.algorithms import livehart

# TODO: Refactoring to break into eeris-specific and general-purpose components
# TODO: Support for multiple processes for specified list of installation ids
# TODO: Edit functionality for appliances


class NILM(object):
    """
    Class to handle streamed data processing for NILM in eeRIS. It also
    maintains a document of the state of appliances in an installation.
    """

    # How often (after how many updates) should we store the model
    # persistently?
    STORE_PERIOD = 10

    def __init__(self, mdb, config, response='cenote'):
        """
        Parameters:
        ----------

        mdb: pymongo.database.Database
        PyMongo database instance for persistent storage and loading

        config: configparser.ConfigParser
        ConfigParser object with sections described in app.create_app()

        """
        # Add state and configuration variables as needed
        self._mdb = mdb
        self._config = config
        self._models = dict()
        self._put_count = dict()
        self._prev = 0.0
        self._response = response
        self._model_lock = dict()
        self._recomputation_active = dict()
        self._recomputation_thread = None
        self._mqtt_thread = None
        self._p_thread = None

        # Configuration variables
        # How are we receiving data
        self._input_method = config['eeRIS']['input_method']
        # Installation ids that we accept for processing
        self._inst_list = \
            [x.strip() for x in config['eeRIS']['inst_ids'].split(",")]
        # Orchestrator JWT pre-shared key
        self._orch_jwt_psk = config['orchestrator']['jwt_psk']
        # Orchestrator URL
        orchestrator_url = config['orchestrator']['url']
        # Endpoint to send device activations
        self._activations_url = orchestrator_url + \
            config['orchestrator']['act_endpoint']
        # Recomputation data URL
        self._computations_url = orchestrator_url + \
            config['orchestrator']['comp_endpoint']
        self._notifications_url = orchestrator_url + \
            config['orchestrator']['notif_endpoint']
        # Initialize thread for sending activation data periodically (if thread
        # = True in the eeRIS configuration section)
        thread = config['eeRIS'].getboolean('thread')
        if thread:
            self._periodic_thread(period=3600)
            # We want to be able to cancel this. If we don't, remove this and
            # just make it a daemon thread.
            atexit.register(self._cancel_periodic_thread)
            logging.debug("Registered periodic thread")
        if config['eeRIS']['input_method'] == 'mqtt':
            self._mqtt_thread = threading.Thread(target=self._mqtt, name='mqtt',
                                                 daemon=True)
            self._mqtt_thread.start()
            logging.debug("Registered mqtt thread")

    def _accept_inst(self, inst_id):
        if self._inst_list is None or inst_id in self._inst_list:
            return True
        else:
            logging.debug(("Received installation id %s which is not in list."
                           "Ignoring.") % (inst_id))
            return False

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
        logging.debug("Sending activations")
        ret = {}
        for inst_id, model in self._models.items():
            logging.debug("Activations for installation %s" % (inst_id))
            if inst_id not in self._model_lock.keys():
                self._model_lock[inst_id] = threading.Lock()
            with self._model_lock[inst_id]:
                payload = []
                for a_k, a in model.appliances:
                    # For now, do not mark activations as sent (we'll do that
                    # only if data have been successfully sent)
                    activations = a.return_new_activations(update_ts=False)
                    for row in activations.itertuples():
                        # Energy consumption in kWh
                        # TODO: Correct consumption by reversing power
                        # normalization
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
                    # Create a JWT for the orchestrator (alternatively, we can
                    # do this only when token has expired)
                    self._orch_token = utils.get_jwt('nilm', self._orch_jwt_psk)
                    # Change data= to json= depending on the orchestrator setup
                    resp = requests.post(self._activations_url,
                                         data=json.dumps(body),
                                         headers={'Authorization': 'jwt %s' % (self._orch_token)})
                    if resp.status_code != falcon.HTTP_200:
                        logging.debug(
                            "Sending of activation data for %s failed: (%d, %s)"
                            % (inst_id, resp.status_code, resp.text)
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
                logging.debug(
                    "Done sending activations of installation %s" % (inst_id))
        return ret

    def _cancel_periodic_thread(self):
        """Stop clustering thread when the app terminates."""
        if self._p_thread is not None:
            self._p_thread.cancel()

    def _periodic_thread(self, period=3600, clustering=False):
        """
        Start a background thread which will send activations for storage and
        optionally perform clustering in all loaded models at regular intervals.

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
        try:
            act_result = self._send_activations()
            logging.debug("Activations report:")
            logging.debug(act_result)
        except:
            logging.debug("Sending of activations failed")
            # Continue
        # Submit new thread
        self._p_thread = threading.Timer(period, self._periodic_thread)
        self._p_thread.daemon = True
        self._p_thread.start()

    def _mqtt(self):
        """
        Thread for data acquisition from the mqtt broker.
        """
        def on_connect(client, userdata, flags, rc):
            print("MQTT connection returned " + mqtt.connack_string(rc))

        def on_log(client, userdata, level, buf):
            print(buf)

        def on_disconnect(client, userdata, rc):
            print("MQTT client disconnected" + mqtt.connack_string(rc))

        def on_message(client, userdata, message):
            x = message.topic.split('/')
            inst_id = x[1]
            if not self._accept_inst(inst_id):
                return

            msg = message.payload.decode('utf-8')
            # Convert message payload to pandas dataframe
            try:
                msg_d = json.loads(msg.replace('\'', '\"'))
            except:
                logging.debug("Exception occurred while decoding message:")
                logging.debug(msg)
                logging.debug("Ignoring data")
                return

            data = pd.DataFrame(msg_d, index=[0])
            data.rename(columns={"ts": "timestamp", "p": "active",
                                 "q": "reactive", "i": "current",
                                 "v": "voltage", "f": "frequency"},
                        inplace=True)
            data.set_index("timestamp", inplace=True, drop=True)
            data.index = pd.to_datetime(data.index, unit='s')
            data.index.name = None
            # Submit for processing by the model
            with self._model_lock[inst_id]:
                model = self._models[inst_id]
                logging.debug('NILM lock (MQTT message)')
                # Process the data
                model.update(data)
            logging.debug('NILM unlock (MQTT message)')
            time.sleep(0.01)
            # Notify orchestrator for appliance detection
            if model.detected_appliance is not None:
                body = {
                    "_id": model.detected_appliance.appliance_id,
                    "name": model.detected_appliance.name,
                    "type": model.detected_appliance.category,
                    "status": "true"
                }
                # TODO: Only when expired?
                self._orch_token = utils.get_jwt('nilm', self._orch_jwt_psk)
                resp = requests.post(self._notifications_url + 'newdevice',
                                     data=json.dumps(body),
                                     headers={'Authorization': 'jwt %s' % (self._orch_token)})
                if resp.status_code != falcon.HTTP_200:
                    logging.debug(
                        "Sending of notification data for %s failed: (%d, %s)"
                        % (inst_id, resp.status_code, resp.text)
                    )
                    logging.debug("Request body:")
                    logging.debug("%s" % (json.dumps(body)))
            self._put_count[inst_id] += 1
            if (self._put_count[inst_id] % self.STORE_PERIOD == 0):
                # Persistent storage
                self._store_model(inst_id)
        ca = self._config['MQTT']['ca']
        key = self._config['MQTT']['key']
        crt = self._config['MQTT']['crt']
        broker = self._config['MQTT']['broker']
        port = int(self._config['MQTT']['port'])
        topic_prefix = self._config['MQTT']['topic_prefix']
        if self._config['MQTT']['identity'] == "random":
            identity = "nilm" + str(int(np.random.rand() * 1000000))
        else:
            identity = self._config['MQTT']['identity']
        client = mqtt.Client(identity, clean_session=False)
        client.tls_set(ca_certs=ca, keyfile=key, certfile=crt)
        client.tls_insecure_set(True)
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_log = on_log
        client.on_message = on_message
        client.connect(broker, port=port)
        # Prepare the models
        for inst_id in self._inst_list:
            # Load or create the model, if not available
            if (inst_id not in self._models.keys()):
                inst_doc = self._mdb.models.find_one({"meterId": inst_id})
                if inst_doc is None:
                    modelstr = dill.dumps(
                        livehart.LiveHart(installation_id=inst_id)
                    )
                    dtime = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')
                    inst_doc = {'meterId': inst_id,
                                'lastUpdate': dtime,
                                'debugInstallation': True,
                                'modelHart': modelstr}
                    self._mdb.models.insert_one(inst_doc)
                self._models[inst_id] = dill.loads(inst_doc['modelHart'])
                if self._models[inst_id]._lock.locked():
                    self._models[inst_id]._lock.release()
                self._model_lock[inst_id] = threading.Lock()
                self._recomputation_active[inst_id] = False
                self._put_count[inst_id] = 0
            if inst_id not in self._model_lock.keys():
                self._model_lock[inst_id] = threading.Lock()
        # Subscribe
        sub_list = [(topic_prefix + "/" + x, 2) for x in self._inst_list]
        client.subscribe(sub_list)
        client.loop_forever()

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
                # Make sure threading lock is released
                if self._models[inst_id]._lock.locked():
                    self._models[inst_id]._lock.release()
                self._model_lock[inst_id] = threading.Lock()
                self._recomputation_active[inst_id] = False
        return self._models[inst_id]

    def _store_model(self, inst_id):
        """
        Helper function to store a model in the database.
        WARNING: This function assumes that the model is already loaded. Also,
        is NOT thread safe, never call it directly unless you know what you're
        doing.
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

    def _recompute_model(self, inst_id, start_ts, end_ts, step=6 * 3600,
                         warmup_period=12*3600):
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

        warmup_period : int
        How many seconds to operate in non-batch mode with 3-second data frames,
        in order to prepare for the appropriate live operation.
        """
        # TODO: Take into account naming events in the model
        if self._computations_url is None:
            logging.debug(("No URL has been provided for past data.",
                           "Model re-computation is not supported."))
            return
        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()
        with self._model_lock[inst_id]:
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
            if self._models[inst_id]._lock.locked():
                self._models[inst_doc]._lock.release()
            self._put_count[inst_id] = 0
            model = self._models[inst_id]
            url = self._computations_url + inst_id
            # Main recomputation loop.
            rstep = step
            for ts in range(start_ts, end_ts-warmup_period, rstep):
                # Endpoint expects timestamp in milliseconds since unix epoch
                st = ts * 1000
                if ts + rstep < end_ts:
                    et = (ts + rstep) * 1000
                else:
                    et = end_ts * 1000

                params = {
                    "start": st,
                    "end": et
                }
                self._orch_token = utils.get_jwt('nilm', self._orch_jwt_psk)
                r = utils.request_with_retry(url, data=json.dumps(params),
                                             request='get',
                                             token=self._orch_token)
                data = utils.get_data_from_cenote_response(r)
                if data is None:
                    continue
                model.update(data, start_thread=False)
                self._put_count[inst_id] += 1
                if (self._put_count[inst_id] % self.STORE_PERIOD == 0):
                    # Persistent storage
                    self._store_model(inst_id)
            # Warmup loop (3-seconds step)
            st = (end_ts - warmup_period + 1) * 1000
            et = end_ts * 1000
            params = {
                "start": st,
                "end": et
            }
            self._orch_token = utils.get_jwt('nilm', self._orch_jwt_psk)
            r = utils.request_with_retry(url, params, request='get',
                                         token=self._orch_token)
            data = utils.get_data_from_cenote_response(r)
            rstep = 3
            for i in range(0, data.shape[0], rstep):
                d = data.iloc[i:i+rstep, :]
                model.update(d, start_thread=False)
                self._put_count[inst_id] += 1
                if (self._put_count[inst_id] % self.STORE_PERIOD == 0):
                    # Persistent storage
                    self._store_model(inst_id)
        self._recomputation_active[inst_id] = False

    def on_get(self, req, resp, inst_id):
        """
        On get, the service returns a document describing the status of a
        specific installation.
        """
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return

        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()
        with self._model_lock[inst_id]:
            model = self._load_model(inst_id)
            logging.debug('NILM lock (GET)')
            resp.body = self._prepare_response_body(model)
        logging.debug('NILM unlock (GET)')
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
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return
        if self._input_method != 'rest':
            resp.status = falcon.HTTP_405
            resp.body = "PUT method not allows when operating" + \
                "outside 'REST' mode"
            return

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
            if self._models[inst_id]._lock.locked():
                self._models[inst_id]._lock.release()
            self._model_lock[inst_id] = threading.Lock()
            self._recomputation_active[inst_id] = False
            self._put_count[inst_id] = 0
        # Inform caller if recomputation is active
        if self._recomputation_active[inst_id]:
            resp.status = falcon.HTTP_204
            resp.body = "Model recomputation in progress, send data again later"
            return
        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()
        with self._model_lock[inst_id]:
            model = self._models[inst_id]
            logging.debug('NILM lock (PUT)')
            # Process the data
            model.update(data)
        logging.debug('NILM unlock (PUT)')
        time.sleep(0.01)
        if model.detected_appliance is not None:
            # TODO: Send notification to orchestrator
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
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return

        # Remove the model, if it is loaded
        if (inst_id in self._models.keys()):
            del self._models[inst_id]
            del self._model_lock[inst_id]
        resp.status = falcon.HTTP_200

    def perform_clustering(self):
        """
        Starts clustering thread for all loaded installations
        """
        for inst_id, model in self._models.items():
            # This should never happen
            if inst_id not in self._model_lock.keys():
                self._model_lock[inst_id] = threading.Lock()
            with self._model_lock[inst_id]:
                model.force_clustering(start_thread=True)
            time.sleep(5)

    def on_post_clustering(self, req, resp, inst_id):
        """
        Starts a clustering thread on the target model
        """
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return

        # Load the model, if not loaded already
        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()
        with self._model_lock[inst_id]:
            model = self._load_model(inst_id)
            logging.debug('NILM lock (clustering)')
            if model.force_clustering(start_thread=True):
                resp.status = falcon.HTTP_200
            else:
                # Conflict
                resp.status = falcon.HTTP_409
        logging.debug('NILM unlock (clustering)')
        time.sleep(0.01)

    def on_get_activations(self, req, resp, inst_id):
        """
        Requests the list of activations for the appliances of an installation.
        """
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return

        # Load the model, if not loaded already
        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()
        with self._model_lock[inst_id]:
            model = self._load_model(inst_id)
            payload = []
            logging.debug('NILM lock (activations)')
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
        logging.debug('NILM unlock (activations)')
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
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return

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
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return

        # TODO: Fixed period for now. Move this to the request, if needed.
        self._periodic_thread(period=3600)
        atexit.register(self._cancel_periodic_thread)
        resp.status = falcon.HTTP_200

    def on_post_stop_thread(self, req, resp, inst_id):
        """
        Cancels the periodic computation thread.
        """
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return

        self._cancel_periodic_thread()
        resp.status = falcon.HTTP_200

    def on_post_appliance_name(self, req, resp, inst_id):
        """
        Sets a new name for an appliance and store the model that has been
        created. Expects parameters appliance_id, name and category, all
        strings.
        """
        # TODO: Also update live appliance
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            return

        if 'appliance_id' not in req.params or \
           'name' not in req.params or \
           'type' not in req.params:
            resp.status = falcon.HTTP_400
            resp.body = "Incorrect query string in request"
            return
        appliance_id = req.params['appliance_id']
        name = req.params['name']
        category = req.params['type']
        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()
        with self._model_lock[inst_id]:
            model = self._load_model(inst_id)
            if appliance_id not in model.appliances:
                logging.debug("Appliance id %s not found in model")
                self._model_lock[inst_id].release()
                time.sleep(0.01)
                resp.status = falcon.HTTP_400
                resp.body = ("Appliance id %s not found" % (appliance_id))
                return
            prev_name = model.appliances[appliance_id].name
            prev_category = model.appliances[appliance_id].category
            model.appliances[appliance_id].name = name
            model.appliances[appliance_id].category = category
            model.appliances[appliance_id].verifyied = True
            logging.debug(("Installation: %s. Renamed appliance "
                           "%s from %s with "
                           "category %s to %s with category %s") %
                          (inst_id, appliance_id, prev_name,
                           prev_category, name, category))
            # Make sure to store the model
            self._store_model(inst_id)
        time.sleep(0.01)
        resp.status = falcon.HTTP_200
