"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import sys
import falcon
import pymongo

import nilm
import installation


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
    api.add_route('/nilm/{inst_id}', nilm.NILM(mdb))
    api.add_route('/installation/{inst_id}/model',
                  installation.InstallationManager(mdb),
                  suffix='model')
    return api


def get_app():
    dburl = "mongodb://localhost:27017/"
    dbname = "eeris"
    return create_app(dburl, dbname)
