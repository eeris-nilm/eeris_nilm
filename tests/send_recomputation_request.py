import sys
import configparser
import requests
from datetime import datetime
from eeris_nilm import utils

try:
    inst = sys.argv[1]
except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} installation_id")

conf_file = 'ini/eeris.ini'
config = configparser.ConfigParser()
config.read(conf_file)
psk = config['REST']['jwt_psk']
url = config['REST']['url'] + '/' + inst + 'recomputation'
# Set this to the time you want to go back (in seconds)
start = datetime.now().timestamp() - 3600 * 24 * 2
end = datetime.now().timestamp()
params = {
    'start': int(start),
    'end': int(end),
    'step': 2*3600
}
token = utils.get_jwt('nilm', psk)
resp = requests.post(url, params=params, headers={'Authorization':
                                                  'jwt %s' % (token)})
print('Response: %d, %s' % (resp.status_code, resp.text))
