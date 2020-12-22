import sys
import configparser
import requests
import json
from datetime import datetime
from eeris_nilm import utils

try:
    inst = sys.argv[1]
except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} installation_id")

conf_file = '/home/diou/git/projects/eeris_nilm/ini/eeris.ini'
config = configparser.ConfigParser()
config.read(conf_file)
psk = config['REST']['jwt_psk']
url = 'http://83.212.104.172:9991/recomputation/' + inst
# One month back from current time, in seconds since unix epoch.
start = datetime.now().timestamp() - 3600 * 24 * 2
end = datetime.now().timestamp()
params = {
    'start': start,
    'end': end,
    'step': 2*3600
}
token = utils.get_jwt('nilm', psk)
resp = requests.post(url, data=json.dumps(params), headers={'Authorization':
                                                            'jwt %s' % (token)})
print('Response: %d, %s', (resp.status_code, resp.body))
