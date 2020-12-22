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
from eeris_nilm.datasets import eeris

# TODO: Refactoring to break into eeris-specific and general-purpose components
# TODO: Support for multiple processes for specified list of installation ids
# TODO: Edit functionality for appliances
# TODO: Include checks for config keys
# TODO: Introduce URL checks and remove double slashes '//' that may result from
#       the configuration strings.


class NILM(object):
    """
    Class to handle streamed data processing for NILM in eeRIS. It also
    maintains a document of the state of appliances in an installation.
    """

    # How often (after how many updates) should we store the model
    # persistently?
    STORE_PERIOD = 10
    # THREAD_PERIOD = 3600
    THREAD_PERIOD = 300

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
        self._input_file_prefix = None
        self._file_date_start = None
        self._file_date_end = None
        self._store_flag = False

        # Configuration variables
        # How are we receiving data
        self._input_method = config['eeRIS']['input_method']
        # Installation ids that we accept for processing
        self._inst_list = \
            [x.strip() for x in config['eeRIS']['inst_ids'].split(",")]
        # Whether to send requests to orchestrator (True) or just print the
        # requests (False)
        self._orch_debug_mode = False
        if 'debug_mode' in config['orchestrator'].keys():
            self._orch_debug_mode = \
                config['orchestrator'].getboolean('debug_mode')
        # Orchestrator JWT pre-shared key
        self._orch_jwt_psk = config['orchestrator']['jwt_psk']
        # Orchestrator URL
        orchestrator_url = config['orchestrator']['url']
        # Endpoint to send device activations
        url = orchestrator_url + '/' + config['orchestrator']['act_endpoint']
        self._activations_url = url
        logging.debug('Activations URL: %s' % (self._activations_url))
        # Recomputation data URL
        url = orchestrator_url + '/' +\
            config['orchestrator']['recomputation_endpoint']
        self._computations_url = url
        logging.debug(self._computations_url)
        url = orchestrator_url + '/' + \
            config['orchestrator']['notif_endpoint_prefix'] + '/'
        self._notifications_url = url
        logging.debug(self._notifications_url)
        self._notifications_suffix = \
            config['orchestrator']['notif_endpoint_suffix']
        self._notifications_batch_suffix = \
            config['orchestrator']['notif_batch_suffix']
        # Prepare the models (redundant?)
        logging.debug('Loading models from database')
        for inst_id in self._inst_list:
            logging.debug('Loading %s' % (inst_id))
            self._load_or_create_model(inst_id)

        if config['eeRIS']['input_method'] == 'file':
            self._input_file_prefix = config['FILE']['prefix']
            self._file_date_start = config['FILE']['date_start']
            self._file_date_end = config['FILE']['date_end']
            self._file_thread = threading.Thread(target=self._process_file,
                                                 name='file', daemon=True)
            self._file_thread.start()
        if config['eeRIS']['input_method'] == 'mqtt':
            self._mqtt_thread = threading.Thread(target=self._mqtt, name='mqtt',
                                                 daemon=True)
            self._mqtt_thread.start()
            logging.info("Registered mqtt thread")

        # Initialize thread for sending activation data periodically (if thread
        # = True in the eeRIS configuration section)
        time.sleep(self.THREAD_PERIOD)
        thread = config['eeRIS'].getboolean('thread')
        if thread:
            self._periodic_thread(period=self.THREAD_PERIOD)
            # We want to be able to cancel this. If we don't, remove this and
            # just make it a daemon thread.
            atexit.register(self._cancel_periodic_thread)
            logging.info("Registered periodic thread")

    def _accept_inst(self, inst_id):
        if self._inst_list is None or inst_id in self._inst_list:
            return True
        else:
            logging.warning(("Received installation id %s which is not in list."
                             "Ignoring.") % (inst_id))
            return False

    # TODO: Part of these operations should be in livehart (?)
    def _load_or_create_model(self, inst_id):
        # Load or create the model, if not available
        if (inst_id not in self._models.keys()):
            inst_doc = self._mdb.models.find_one({"meterId": inst_id})
            if inst_doc is None:
                logging.debug('Installation %s does not exist in database,'
                              'creating' % (inst_id))
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
            self._models[inst_id]._lock = threading.Lock()
            self._models[inst_id]._clustering_thread = None
            self._models[inst_id].detected_appliance = None
            self._model_lock[inst_id] = threading.Lock()
            self._recomputation_active[inst_id] = False
            self._put_count[inst_id] = 0
        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()

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
        logging.info("Sending activations")
        ret = {}
        for inst_id, model in self._models.items():
            logging.debug("Activations for installation %s" % (inst_id))
            if inst_id not in self._model_lock.keys():
                self._model_lock[inst_id] = threading.Lock()
            with self._model_lock[inst_id]:
                payload = []
                activations = {}
                for a_k, a in model.appliances.items():
                    # For now, do not mark activations as sent (we'll do that
                    # only if data have been successfully sent)
                    activations[a_k] = a.return_new_activations(update_ts=False)
                    for row in activations[a_k].itertuples():
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
                logging.debug('Activations body: %s' % (json.dumps(body)))
            # Send stuff, outside lock (to avoid unnecessary delays)
            if self._activations_url is not None and \
               not self._orch_debug_mode:
                # Create a JWT for the orchestrator (alternatively, we can
                # do this only when token has expired)
                self._orch_token = utils.get_jwt('nilm', self._orch_jwt_psk)
                # Change data= to json= depending on the orchestrator setup
                try:
                    resp = requests.post(self._activations_url,
                                         data=json.dumps(body),
                                         headers={'Authorization': 'jwt %s' %
                                                  (self._orch_token)})
                except Exception as e:
                    logging.warning("Sending of activations failed!")
                    logging.warning("Exception type: %s" % (str(type(e))))
                    logging.warning(e)
                if not resp.ok:
                    logging.error(
                        "Sending of activation data for %s failed: (%d, %s)"
                        % (inst_id, resp.status_code, resp.text)
                    )
                    ret[inst_id] = resp
                else:
                    # Everything went well, mark the activations as sent
                    with self._model_lock[inst_id]:
                        for a_k, a in model.appliances.items():
                            if activations[a_k].empty:
                                continue
                            # Should never happen
                            if a_k not in activations:
                                logging.warning('Appliance key %s not'
                                                'found in model' % (a_k))
                                continue
                            else:
                                # Just making sure
                                activations[a_k].sort_values('end',
                                                             ascending=True,
                                                             ignore_index=True)
                                a.last_returned_end_ts = \
                                    (activations[a_k])['end'].iloc[-1]
                    logging.debug(
                        "Activations for %s sent successfully", inst_id)
                    ret[inst_id] = \
                        json.dumps(body)
            else:
                # Assume everything went well (simulate), and mark activations
                # as sent
                with self._model_lock[inst_id]:
                    for a_k, a in model.appliances.items():
                        if activations[a_k].empty:
                            continue
                        # Should never happen
                        if a_k not in activations:
                            logging.warning('Appliance key %s not'
                                            'found in model' % (a_k))
                            continue
                        else:
                            # Just making sure
                            activations[a_k].sort_values('end',
                                                         ascending=True,
                                                         ignore_index=True)
                            a.last_returned_end_ts = \
                                (activations[a_k])['end'].iloc[-1]
                            logging.debug('Appliance %s: Last return ts: %s'
                                          % (a.appliance_id,
                                             str(a.last_returned_end_ts)))

                logging.debug(
                    "Activations for %s marked (debug)", inst_id)
                ret[inst_id] = json.dumps(body)
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
        logging.info("Starting periodic thread.")
        if clustering:
            self.perform_clustering()
            # Wait 5 minutes, in case clustering is completed within this time.
            time.sleep(300)
        # Send activations
        try:
            act_result = self._send_activations()
            logging.debug("Activations report:")
            logging.debug(act_result)
        except Exception as e:
            logging.warning("Sending of activations failed!")
            logging.warning("Exception type: %s" % (str(type(e))))
            logging.warning(e)
            # Continue
        # Submit new thread
        self._p_thread = threading.Timer(period, self._periodic_thread,
                                         kwargs={'period': period})
        self._p_thread.daemon = True
        self._p_thread.start()

    def _handle_notifications(self, model):
        """
        Helper function to assist in sending of notifications to the
        orchestrator, when unnamed appliances are detected.
        """
        if model.detected_appliance is None:
            # No appliance was detected, do nothing
            return

        body = {
            "_id": model.detected_appliance.appliance_id,
            "name": model.detected_appliance.name,
            "type": model.detected_appliance.category,
            "status": "true"
        }
        inst_id = model.installation_id
        logging.debug("Sending notification data:")
        # TODO: Only when expired?
        self._orch_token = utils.get_jwt('nilm', self._orch_jwt_psk)
        if not self._orch_debug_mode:
            url = self._notifications_url + inst_id + '/' + \
                self._notifications_suffix
            try:
                resp = requests.post(url, data=json.dumps(body),
                                     headers={'Authorization': 'jwt %s' %
                                              (self._orch_token)})
            except Exception as e:
                logging.warning("Sending of notification failed!")
                logging.warningn("Exception type: %s" % (str(type(e))))
                logging.warning(e)
            if not resp.ok:
                logging.error(
                    "Sending of notification data for %s failed: (%d, %s)"
                    % (model.installation_id, resp.status_code, resp.text)
                )
                logging.error("Request body:")
                logging.error("%s" % (json.dumps(body)))
            else:
                logging.debug("Appliance detection notification sent: %s" %
                              (json.dumps(body)))
        else:
            logging.debug("Notification: %s", json.dumps(body))

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

            if rc != 0:
                logging.error(
                    'Unexpected MQTT disconnect. Attempting to reconnect')
            else:
                logging.error('MQTT disconnected, rc value:' + str(rc))

            counter = 0
            while counter < 10:
                try:
                    broker = self._config['MQTT']['broker']
                    port = int(self._config['MQTT']['port'])
                    logging.info("Waiting 10 seconds...")
                    time.sleep(10)
                    logging.info("Trying to Reconnect...")
                    client.connect(broker, port=port, keepalive=30)
                    break
                except Exception as e:
                    logging.warning("Error in broker connection"
                                    "attempt. Retrying.")
                    logging.warningn("Exception type: %s" % (str(type(e))))
                    logging.warning(e)
                    counter += 1

        def on_message(client, userdata, message):
            x = message.topic.split('/')
            inst_id = x[1]
            if not self._accept_inst(inst_id):
                return

            msg = message.payload.decode('utf-8')
            # Convert message payload to pandas dataframe
            try:
                msg_d = json.loads(msg.replace('\'', '\"'))
            except Exception as e:
                logging.error("Exception occurred while decoding message:")
                logging.error(msg)
                logging.error(e)
                logging.error("Ignoring data")
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
            time.sleep(0.05)
            # Notify orchestrator for appliance detection
            self._handle_notifications(model)
            self._put_count[inst_id] += 1
            # It is possible that _store_model cannot store, and keeps this to
            # True afterwards
            if self._put_count[inst_id] % self.STORE_PERIOD == 0:
                self._store_flag = True
            if self._store_flag:
                # Persistent storage
                self._store_model(inst_id)
        # Prepare the models (As it stands, it shouldn't do anything)
        logging.debug('Loading models from database (MQTT) thread')
        for inst_id in self._inst_list:
            logging.debug('Loading %s' % (inst_id))
            self._load_or_create_model(inst_id)
        # Connect to MQTT
        ca = self._config['MQTT']['ca']
        key = self._config['MQTT']['key']
        crt = self._config['MQTT']['crt']
        broker = self._config['MQTT']['broker']
        port = int(self._config['MQTT']['port'])
        topic_prefix = self._config['MQTT']['topic_prefix']
        if self._config['MQTT']['identity'] == "random":
            # Option for random identity:
            identity = "nilm" + str(int(np.random.rand() * 1000000))
        else:
            identity = self._config['MQTT']['identity']
        clean_session = self._config['MQTT'].getboolean('clean_session')
        client = mqtt.Client(identity, clean_session=clean_session)
        client.tls_set(ca_certs=ca, keyfile=key, certfile=crt)
        client.tls_insecure_set(True)
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_log = on_log
        client.on_message = on_message
        client.connect(broker, port=port, keepalive=30)
        # Subscribe
        sub_list = [(topic_prefix + "/" + x, 2) for x in self._inst_list]
        client.subscribe(sub_list)
        # Sleep for a while, while the system initializes
        time.sleep(10)
        client.loop_forever()

    def _process_file(self):
        if len(self._inst_list) > 1:
            raise ValueError('Only one installation is supported in file mode')
        inst_id = self._inst_list[0]
        self._load_or_create_model(inst_id)
        model = self._models[inst_id]
        power = eeris.read_eeris(self._input_file_prefix, self._file_date_start,
                                 self._file_date_end)
        step = 3  # Hardcoded
        logging_step = 7200
        start_ts = pd.Timestamp(self._file_date_start)
        power = power.loc[power.index > start_ts].dropna()
        end = power.shape[0] - power.shape[0] % step
        for i in range(0, end, step):
            data = power.iloc[i:i+step]
            model.update(data)
            if (i + step) % logging_step == 0:
                logging.debug('Processed %d seconds' % (i + step))
            # Model storage
            if ((self._put_count[inst_id] // step) % self.STORE_PERIOD == 0):
                self._store_flag = True
            if self._store_flag:
                # Persistent storage
                self._store_model(inst_id)
            time.sleep(0.01)
        self._handle_notifications(model)
        self._put_count[inst_id] += step

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
        if not model.is_background_overestimated():
            app_d = {"_id": '000000000000000000000001',
                     "name": "Background",
                     "type": "background",
                     "active": model.background_active,
                     "reactive": 0.0}
            d = {"data": app_d, "timestamp": ts}
            payload.append(d)
        for i in range(len(model.live)):
            app = model.live[i]
            app_d = {"_id": app.appliance_id,
                     "name": app.name,
                     "type": app.category,
                     "active": app.signature[0, 0],
                     "reactive": app.signature[0, 1]}
            d = {"data": app_d, "timestamp": ts}
            payload.append(d)
        # We ignore residuals under 5 Watts.
        if model.residual_live[0] > 5.0 and model.background_active < 10000:
            app_d = {"_id": '000000000000000000000002',
                     "name": "Other",
                     "type": "residual",
                     "active": model.residual_live[0],
                     "reactive": model.residual_live[1]}
            d = {"data": app_d, "timestamp": ts}
            payload.append(d)
        body_d = {"installation_id": str(model.installation_id),
                  "payload": payload}
        try:
            body = json.dumps(body_d)
        except (ValueError, TypeError):
            logging.error("Error while preparing response body:")
            logging.error(body_d['installation_id'])
            logging.error(body_d['payload'])
            for k in body_d['payload']:
                logging.error(body_d[k])
            raise
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

    # TODO: This is probably a better implementation, should replace current
    # after testing.
    def _store_model_NEXT_VERSION(self, inst_id):
        """
        Helper function to store a model in the database.
        WARNING: This function assumes that the model is already loaded. Also,
        is NOT thread safe, never call it directly unless you know what you're
        doing.
        """
        model = self._models[inst_id].deepcopy()
        model.clustering_thread = None
        model._lock = None
        modelstr = dill.dumps(model)
        upd = {'$set': {
            'meterId': inst_id,
            'lastUpdate': str(dt.datetime.now()),
            'debugInstallation': True,
            'modelHart': modelstr}
        }
        self._mdb.models.update_one({'meterId': inst_id}, upd)
        self._store_flag = False

    # TODO: This will probably be deprecated
    def _store_model(self, inst_id):
        """
        Helper function to store a model in the database.
        WARNING: This function assumes that the model is already loaded. Also,
        is NOT thread safe, never call it directly unless you know what you're
        doing.
        """
        model = self._models[inst_id]
        if model.is_clustering_active():
            # Cannot store at this point
            logging.debug('Clustering thread active for %s, do not store' %
                          (inst_id))
            return
        modelstr = dill.dumps(model)
        upd = {'$set': {
            'meterId': inst_id,
            'lastUpdate': str(dt.datetime.now()),
            'debugInstallation': True,
            'modelHart': modelstr}
        }
        self._mdb.models.update_one({'meterId': inst_id}, upd)
        self._store_flag = False
        # logging.debug('Stored model for %s' % (inst_id))

    def _recompute_model(self, inst_id, start_ts, end_ts, step=6 * 3600,
                         warmup_period=2*3600):
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
            logging.warning(("No URL has been provided for past data.",
                             "Model re-computation is not supported."))
            return
        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()
        with self._model_lock[inst_id]:
            logging.debug('Recomputing model for %s' % (inst_id))
            self._recomputation_active[inst_id] = True
            # Delete model from memory
            self._models.pop(inst_id, None)
            # Delete model from database
            result = self._mdb.models.delete_many({'meterId': inst_id})
            dcount = int(result.deleted_count)
            if dcount == 0:
                logging.warning("Installation does not exist in the database")
            elif dcount > 1:
                logging.warning("More than one documents deleted in database")
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
            logging.debug('Starting recomputation loop')
            rstep = step
            for ts in range(start_ts, end_ts - warmup_period, rstep):
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
                if (self._put_count[inst_id] // step) % \
                   self.STORE_PERIOD == 0:
                    self._store_flag = True
                if self._store_flag:
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
            rstep = 3  # Hardcoded
            for i in range(0, data.shape[0], rstep):
                d = data.iloc[i:i+rstep, :]
                model.update(d, start_thread=False)
                self._put_count[inst_id] += 1
                if (self._put_count[inst_id] // rstep) % \
                   self.STORE_PERIOD == 0:
                    self._store_flag = True
                if self._store_flag:
                    # Persistent storage
                    self._store_model(inst_id)

            # Name the appliances based on past user resposes
            url = self._notifications_url + '/' + inst_id + '/' + \
                self._notifications_past_suffix
            self._orch_token = utils.get_jwt('nilm', self._orch_jwt_psk)
            r = utils.request_with_retry(url, token=self._orch_token)
            if not r.ok:
                logging.error(
                    "Error in receiving data for %s failed: (%d, %s)"
                    % (inst_id, r.status_code, r.text)
                )
            else:
                # Everything went well, process the past notification responses
                logging.debug("Notifications for %s received successfully,"
                              "processing.", inst_id)
                self._recomputation_appliance_naming(inst_id, r.text)
        self._recomputation_active[inst_id] = False

    def _recomputation_appliance_naming(self, inst_id, naming):
        """
        Helper function to name detected appliances based on their activation
        time and past user responses.

        The 'naming' argument is a json string that has the following format:
        {
        'ok': True, 'results': [
            {'profileid': '5e4e576484ad10180c404ecb',
             'notificationid': '5f9d7187bdb8411380b8ad00',
             'deviceid': '5d80e4c5e209594310f3dd07',
             'selecteddevice': 'oven',
             'timestamp': '1606129690518', // when the user responded
             'uuid': 'ff77433f-e72e-4988-baf2-a48dd7c3afdd',
             'cenote$created_at': 1606130024761,
             'cenote$timestamp': 1606130024761,
             'cenote$id': '54b67e7d-2c15-4ec6-82e7-5b80c585ad95',
             'createdat': '1604153735987' // when the notification was sent  },
        {
           ...
        },
        ...
        ]
        }

        In practice, we are interested for the fields 'selecteddevice' and
        'createdat'.
        """
        model = self._models[inst_id]
        notif = json.loads(naming)['results']
        # Beyond 5 seconds we think there is no match (it's already a very large
        # difference)
        min_diff = pd.Timedelta(value=5, units='seconds')
        appliance = None
        for n in notif:
            ts = pd.to_datetime(n['created_at'], unit='ms')
            for a in model.appliances:
                idx = a.activations['start'].sub(ts).abs().idxmin()
                nearest = a.at[idx, 'start']
                if ts - nearest < min_diff:
                    appliance = a
        if appliance is not None:
            appliance.category = notif['selecteddevice']
            # Handle multiple appliances of same type
            count = 0
            for a in model.appliances:
                if a.category == appliance.category:
                    count += 1
            appliance.name = '%s %d' % (notif['selecteddevice'], count)

    def on_get(self, req, resp, inst_id):
        """
        On get, the service returns a document describing the status of a
        specific installation.
        """
        if not self._accept_inst(inst_id):
            resp.status = falcon.HTTP_400
            resp.body = "Installation not in list for this NILM instance."
            logging.info(("Rejected request for installation %s") % (inst_id))
            return

        if inst_id not in self._model_lock.keys():
            self._model_lock[inst_id] = threading.Lock()
        with self._model_lock[inst_id]:
            logging.debug('NILM lock (GET)')
            resp.body = self._prepare_response_body(self._models[inst_id])
            logging.debug('Response body: %s' % (resp.body))
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
            logging.info(("Rejected request for installation %s") % (inst_id))
            return
        if self._input_method != 'rest':
            resp.status = falcon.HTTP_405
            resp.body = "PUT method not allowed when operating" + \
                "outside 'REST' mode"
            logging.info("Attempted to call PUT method outside \'REST\' mode")
            return

        if req.content_length:
            data = pd.read_json(req.stream)
        else:
            logging.info("No data provided")
            raise falcon.HTTPBadRequest("No data provided", "No data provided")
        # Load or create the model, if not available
        self._load_or_create_model(inst_id)
        # Inform caller if recomputation is active
        if self._recomputation_active[inst_id]:
            resp.status = falcon.HTTP_204
            resp.body = "Model recomputation in progress, send data again later"
            logging.info(
                "Received request while model recomputation is in progress")
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
        self._handle_notifications(model)
        self._put_count[inst_id] += 1
        if (self._put_count[inst_id] % self.STORE_PERIOD == 0):
            self._store_flag = True
        if self._store_flag:
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
            logging.debug('NILM lock (clustering)')
            if self._models[inst_id].force_clustering(start_thread=True):
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
            payload = []
            logging.debug('NILM lock (activations)')
            for a_k, a in self._models[inst_id].appliances.items():
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
        self._periodic_thread(period=self.THREAD_PERIOD)
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
        logging.debug('Received appliance naming event for installation %s' %
                      (inst_id))
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
            model = self._models[inst_id]
            if appliance_id not in model.appliances:
                logging.warning("Appliance id %s not found in model")
                self._model_lock[inst_id].release()
                time.sleep(0.01)
                resp.status = falcon.HTTP_400
                resp.body = ("Appliance id %s not found" % (appliance_id))
                return
            prev_name = model.appliances[appliance_id].name
            prev_category = model.appliances[appliance_id].category
            model.appliances[appliance_id].name = name
            model.appliances[appliance_id].category = category
            model.appliances[appliance_id].verified = True
            # Also update live appliance
            live_app = next((item for item in model.live if
                             item.appliance_id == appliance_id), None)
            if live_app is None:
                logging.warning(
                    "Notifications: Appliance id %s not in the"
                    "list of live appliances" % (live_app.appliance_id))
            else:
                live_app.name = name
                live_app.category = category
                live_app.verified = True
            logging.info(("Installation: %s. Renamed appliance "
                          "%s from %s with "
                          "category %s to %s with category %s") %
                         (inst_id, appliance_id, prev_name,
                          prev_category, name, category))
            # Make sure to store the model
            self._store_model(inst_id)
        time.sleep(0.01)
        resp.status = falcon.HTTP_200
