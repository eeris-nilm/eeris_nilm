import dill
import os
import sys
from eeris_nilm import evaluation
from eeris_nilm.datasets import redd

if len(sys.argv) < 2:
    print("Please provide full path to a REDD dataset house directory")
    sys.exit(1)
path = sys.argv[1]
house = os.path.basename(path)

# redd_path = 'tests/data/redd/'
# house = 'house_2'
# path = os.path.join(redd_path, house)
# # date_start = '2011-04-18T00:00'
# # date_end = '2011-04-30T23:59'
date_start = None
date_end = None
step = 7
data, labels = redd.read_redd(path, date_start=date_start,
                              date_end=date_end)
model = evaluation.live_run(data['mains'], step=step)
outfile = os.path.join('tests/data/redd_live/', house + '_live_hist.dill')
with open(outfile, 'wb') as f:
    dill.dump(model.live_history, f)
