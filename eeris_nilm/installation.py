"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import falcon


class InstallationManager(object):
    """
    Class to handle database management for installations/models.
    """

    def __init__(self, mdb):
        # Database
        self._mdb = mdb

    def on_get(self, req, resp, inst_id):
        """
        Handling data retrieval, besides "live". Not implemented.
        """
        raise falcon.HTTPNotImplemented("Data retrieval not implemented",
                                        "Data retrieval not implemented")

    def on_delete_model(self, req, resp, inst_id):
        """
        Delete the stored model and start again. Useful for testing, or when the
        model needs to be rebuilt from scratch from the data.  CAUTION: This can
        cause problems if this route is called when the service is in
        production. It will not delete model unless the 'debugInstallation'
        field is set to True.
        """
        inst_iid = int(inst_id)
        model = self._mdb.models.find_one({"meterId": inst_iid})
        if model is None:
            debug = False
            d_count = 0
        else:
            debug = model['debugInstallation']
            if debug:
                # There should be only one document. In any case, delete only
                # one.
                result = self._mdb.models.delete_one({'meterId': inst_iid})
                d_count = int(result.deleted_count)
            else:
                d_count = 0
            resp.body = '''{
            "acknowledged": true,
            "debugInstallation": %s,
            "deletedCount": %d }''' % (debug, d_count)
        resp.status = falcon.HTTP_200
