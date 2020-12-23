import sys
import configparser
import requests
import time
import json
from eeris_nilm import utils

try:
    inst = sys.argv[1]
except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} installation_id")

conf_file = 'ini/eeris.ini'
config = configparser.ConfigParser()
config.read(conf_file)
psk = config['REST']['jwt_psk']
url = config['REST']['url'] + '/' + inst
# Set this to the time you want to go back (in seconds)
token = utils.get_jwt('nilm', psk)

for i in range(1000):
    resp = requests.get(url, headers={'Authorization':
                                      'jwt %s' % (token)})
    txt = json.loads(resp.text)
    print('Response: %d, %s' % (resp.status_code, json.dumps(txt, indent=4,
                                                             sort_keys=True)))
    time.sleep(1)
