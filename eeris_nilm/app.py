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
import pymongo
import eeris_nilm.nilm
import eeris_nilm.installation

# TODO: Authentication


def create_app(dburl, dbname, act_url=None, recomp_url=None,
               secret_key=None, inst_list=None, thread=False):
    """
    Main web application.

    IMPORTANT NOTICE: This application is designed to run under a single process
    and single thread in WSGI. It will not work properly if multiple processes
    operate using the same data at once.

    Parameters
    ----------

    dburl: string
    MongoDB url (used for model persistent storage)

    dbname: string
    MongoDB database name

    act_url : string
    Activations service URL (for submitting detected device activations for
    storage). If none, then a JSON with the activations is printed in the
    stdout, for debugging purposes.

    recomp_url: string
    URL of service that provides retrospective appliance data

    secret_key: string
    Key used for JWT authentication. NOT IMPLEMENTED

    inst_list: list of strings
    List of installation ids to be handled by this application instance. This
    parameter is directly passed to the NILM object instance.

    thread: bool
    Initiate a periodic thread to send activations.
    """
    # # Authentication
    # def user_loader(username, password):
    #     return {'username': username}
    # auth_backend = JWTAuthBackend()
    # auth_middleware = FalconAuthMiddleware(auth_backend)

    # DB connection
    logging.debug("Connecting to database")
    mclient = pymongo.MongoClient(dburl)
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
    # orchestrator_url = 'http://localhost:8001/'
    orchestrator_url = 'http://83.212.104.172:8000/'
    act_url = orchestrator_url + 'historical/events/'
    comp_url = orchestrator_url + 'historical/'

    logging.debug("Setting up connections")
    nilm = eeris_nilm.nilm.NILM(mdb, thread=thread, act_url=act_url,
                                comp_url=comp_url)
    api.add_route('/nilm/{inst_id}', nilm)
    api.add_route('/nilm/{inst_id}/clustering', nilm, suffix='clustering')
    api.add_route('/nilm/{inst_id}/activations', nilm, suffix='activations')
    api.add_route('/nilm/{inst_id}/recomputation', nilm, suffix='recomputation')
    api.add_route('/nilm/{inst_id}/start_thread', nilm, suffix='start_thread')
    api.add_route('/nilm/{inst_id}/stop_thread', nilm, suffix='stop_thread')
    api.add_route('/nilm/{inst_id}/appliance_name',
                  nilm, suffix='appliance_name')
    # Installation
    api.add_route('/installation/{inst_id}/model',
                  eeris_nilm.installation.InstallationManager(mdb),
                  suffix='model')
    logging.debug("Ready")
    return api


def get_app(inst_list=None, thread=False):
    dburl = "mongodb://localhost:27017/"
    dbname = "eeris"
    return create_app(dburl, dbname, inst_list, thread)
