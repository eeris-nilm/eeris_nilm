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
from falcon_auth import FalconAuthMiddleware, JWTAuthBackend
import configparser
import pymongo
import eeris_nilm.nilm
import eeris_nilm.installation


def create_app(conf_file):
    """
    Main web application.

    IMPORTANT NOTICE: This application is designed to run under a single process
    and single thread in WSGI. It will not work properly if multiple processes
    operate using the same data at once.

    Parameters
    ----------

    conf_file: String
    Configuration file in ini format. See file ini/example_eeris.ini for an
    example.

    """
    # Config file parsing
    config = configparser.ConfigParser()
    config.read(conf_file)

    # Set logging level
    if 'loglevel' not in config['eeRIS'].keys():
        loglevel = logging.DEBUG
    else:
        loglevel = eval('logging.' + config['eeRIS']['loglevel'])

    logging.basicConfig(level=loglevel)

    # DB connection
    logging.info("Connecting to database")
    mclient = pymongo.MongoClient(config['eeRIS']['dburl'])
    dbname = config['eeRIS']['dbname']
    dblist = mclient.list_database_names()
    if dbname in dblist:
        mdb = mclient[dbname]
    else:
        sys.stderr.write('ERROR: Database ' + dbname + ' not found. Exiting.')
        return

    # Authentication
    auth_backend = JWTAuthBackend(lambda user: user,
                                  config['REST']['jwt_psk'],
                                  algorithm='HS256',
                                  expiration_delta=24*60*60)
    auth_middleware = FalconAuthMiddleware(auth_backend,
                                           exempt_methods=['HEAD'])
    # auth_middleware = FalconAuthMiddleware(auth_backend)

    api = falcon.API(middleware=[auth_middleware])
    # api = falcon.API()

    # NILM
    logging.info("Setting up connections")
    nilm = eeris_nilm.nilm.NILM(mdb, config)
    api.add_route('/nilm/{inst_id}', nilm)
    api.add_route('/nilm/{inst_id}/clustering', nilm, suffix='clustering')
    api.add_route('/nilm/{inst_id}/activations', nilm, suffix='activations')
    api.add_route('/nilm/{inst_id}/recomputation', nilm, suffix='recomputation')
    api.add_route('/nilm/{inst_id}/start_thread', nilm, suffix='start_thread')
    api.add_route('/nilm/{inst_id}/stop_thread', nilm, suffix='stop_thread')
    api.add_route('/nilm/{inst_id}/appliance_name',
                  nilm, suffix='appliance_name')
    # Installation manager (for database management - limited functionality)
    # inst_ids = [x.strip() for x in config['eeRIS']['inst_ids'].split(",")]
    # inst_manager = eeris_nilm.installation.\
    #     InstallationManager(mdb, inst_list=inst_ids)
    # api.add_route('/installation/{inst_id}/model', inst_manager,
    # suffix='model')
    logging.info("Ready")
    return api


def get_app(conf_file):
    return create_app(conf_file)
