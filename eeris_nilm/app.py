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

import sys
import falcon
import logging
# from falcon_auth import FalconAuthMiddleware, JWTAuthBackend
import configparser
import pymongo
import eeris_nilm.nilm
import eeris_nilm.installation


# TODO: Authentication
def create_app(conf_file):
    """
    Main web application.

    IMPORTANT NOTICE: This application is designed to run under a single process
    and single thread in WSGI. It will not work properly if multiple processes
    operate using the same data at once.

    Parameters
    ----------

    conf_file: String
    Configuration file in ini format, with the following structure:

    [eeRIS]
    dburl = mongodb://localhost:27017/ # URL for the local mongodb
    dbname = eeris  # Database name
    #input_method = rest  # Input method can be "rest" or "mqtt"
    input_method = mqtt
    inst_ids = [id1], [id2], ... # Which installations should we monitor?
    response = cenote  # Response format. Possible values are cenote and debug
    thread = False  # Initiate a periodic thread to send activations. If false,
                    # then these should be sent manually.

    [REST]
    jwt_psk = [secret]  # jwt pre-shared key

    [MQTT]
    mqtt_broker = [url] # URL to the mqtt broker
    mqtt_crt = /path/to/client.crt # Client certificate path
    mqtt_key = /path/to/client.key # Client certificate key path
    mqtt_ca_key = /path/to/CA.crt # Certificate Authority path
    mqtt_client_pass = [secret] # Client key passphrase
    mqtt_topic_prefix = eeris # mqtt topic prefix to subscribe

    [orchestrator]
    url = [url] # eeRIS orchestrator URL
    act_endpoint = historical/events   # Activations service URL (for
                                       # submitting detected device activations
                                       # for storage). If none, then a JSON with
                                       # the activations is printed in the
                                       # stdout, for debugging purposes.
    comp_endpoint = historical/   # Endpoint for requesting batch historical
                                  # data for recomputation purposes
    """
    # # Authentication
    # def user_loader(username, password):
    #     return {'username': username}
    # auth_backend = JWTAuthBackend()
    # auth_middleware = FalconAuthMiddleware(auth_backend)

    # Config file parsing
    config = configparser.ConfigParser()
    config.read(conf_file)

    # DB connection
    logging.debug("Connecting to database")
    mclient = pymongo.MongoClient(config['eeRIS']['dburl'])
    dbname = config['eeRIS']['dbname']
    dblist = mclient.list_database_names()
    if dbname in dblist:
        mdb = mclient[dbname]
    else:
        sys.stderr.write('ERROR: Database ' + dbname + ' not found. Exiting.')
        return

    # Gunicorn expects the 'application' name
    # api = falcon.API(middleware=[auth_middleware])
    api = falcon.API()

    # NILM
    logging.debug("Setting up connections")
    nilm = eeris_nilm.nilm.NILM(mdb, config)
    input_method = config['eeRIS']['input_method']
    if input_method == "rest":
        api.add_route('/nilm/{inst_id}', nilm)
    elif input_method == "mqtt":
        pass
    else:
        raise ValueError(("Invalid input method %s") % (input_method))

    api.add_route('/nilm/{inst_id}/clustering', nilm, suffix='clustering')
    api.add_route('/nilm/{inst_id}/activations', nilm, suffix='activations')
    api.add_route('/nilm/{inst_id}/recomputation', nilm, suffix='recomputation')
    api.add_route('/nilm/{inst_id}/start_thread', nilm, suffix='start_thread')
    api.add_route('/nilm/{inst_id}/stop_thread', nilm, suffix='stop_thread')
    api.add_route('/nilm/{inst_id}/appliance_name',
                  nilm, suffix='appliance_name')
    # Installation manager (for database management - limited functionality)
    inst_ids = [x.strip() for x in config['eeRIS']['inst_ids'].split(",")]
    inst_manager = eeris_nilm.installation.\
        InstallationManager(mdb, inst_list=inst_ids)
    api.add_route('/installation/{inst_id}/model', inst_manager, suffix='model')
    logging.debug("Ready")
    return api


def get_app(conf_file):
    return create_app(conf_file)
