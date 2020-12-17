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

# Test requests to the NILM service
import time
import requests
import configparser
from eeris_nilm import utils

import logging
logging.basicConfig(level=logging.DEBUG)

# Create jwt
config = configparser.ConfigParser()
config.read('ini/eeris.ini')
secret = config['REST']['jwt_psk']
token = utils.get_jwt('orchestrator', secret)
prefix = 'jwt'

# Just select the first installation for the demo.
inst = config['eeRIS']['inst_ids'].split(',')[0].strip()
nilm_url = 'http://localhost:9991/nilm/' + inst
inst_url = 'http://localhost:9991/installation/' + inst + '/model'

# Main loop
for i in range(0, 1000):
    r = requests.get(nilm_url,
                     headers={'Authorization': '%s %s' % (prefix, token)})
    if r.status_code != 200:
        print("Something went wrong, received HTTP %d" % (r.status_code))
    live = r.text
    print("GET response (live): %s" % (live))
    time.sleep(1)
