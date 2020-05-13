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
        model = self._mdb.models.find_one({"meterId": inst_id})
        if model is None:
            debug = False
            d_count = 0
        else:
            debug = model['debugInstallation']
            if debug:
                # There should be only one document. In any case, delete only
                # one.
                result = self._mdb.models.delete_one({'meterId': inst_id})
                d_count = int(result.deleted_count)
            else:
                d_count = 0
            resp.body = '''{
            "acknowledged": true,
            "debugInstallation": %s,
            "deletedCount": %d }''' % (debug, d_count)
        resp.status = falcon.HTTP_200

    # TODO: Appliance naming/renaming
