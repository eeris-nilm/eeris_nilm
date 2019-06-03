import time
import warnings
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
# from nilmtk import utils
from nilmtk import DataSet

import requests
import falcon
# import eeris_nilm.app

# ECO parameters
# data = DataSet('/home/diou/datasets/NILM/ECO.h5')
data = DataSet('/home/diou/research/nilm/eeris_nilm/tests/ECO_1.h5')
data.set_window(start='2012-09-01', end='2012-09-02')
chunksize = 60  # One-minute chunksize

elec = data.buildings[1].elec
mains = elec.mains()
base_url = 'http://localhost:8000/nilm/2'
for chunk in mains.load(chunksize=chunksize):
    chunk.set_axis(['active', 'voltage', 'phase_angle', 'current',
                    'reactive'], axis='columns', inplace=True)
    r = requests.put(base_url, data=chunk.to_json(orient='table'))
    print(r.status_code)
