import sys
import falcon
import pymongo

from .nilm import Installation


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
    api.add_route('/nilm/installation/', Installation(mdb))
    return api


def get_app():
    dburl = "mongodb://localhost:27017/"
    dbname = "eeris"
    return create_app(dburl, dbname)
