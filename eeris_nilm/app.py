"""
Copyright 2019 Christos Diou

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
import pymongo
import eeris_nilm.nilm
import eeris_nilm.installation


def create_app(dburl, dbname):
    # DB connection
    mclient = pymongo.MongoClient(dburl)
    dblist = mclient.list_database_names()
    if dbname in dblist:
        mdb = mclient[dbname]
    else:
        sys.stderr.write('ERROR: Database ' + dbname + ' not found. Exiting.')
        return

    # Gunicorn expects the 'application' name
    api = falcon.API()
    api.add_route('/nilm/{inst_id}', eeris_nilm.nilm.NILM(mdb))
    api.add_route('/installation/{inst_id}/model',
                  eeris_nilm.installation.InstallationManager(mdb),
                  suffix='model')
    return api


def get_app():
    dburl = "mongodb://localhost:27017/"
    dbname = "eeris"
    return create_app(dburl, dbname)
