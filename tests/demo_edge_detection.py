"""
Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

# Demo of edge detection without REST service implementation
import matplotlib.pyplot as plt
from nilmtk import DataSet
from eeris_nilm.hart85_eeris import Hart85eeris

eco = DataSet('tests/ECO_1.h5')
eco.set_window(start='2012-09-01 07:00', end='2012-09-01 23:59')
chunksize =  * 24 * 3600
step = 5
plot_length = 3000
elec = eco.buildings[1].elec
mains = elec.mains()

# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111)

model = Hart85eeris(installation_id=1)
current_sec = 0
for chunk in mains.load(chunksize=chunksize):
    chunk = chunk[[('power', 'active'), ('power', 'reactive'), ('voltage', ''),
                   ('phase_angle', ''), ('current', '')]]
    chunk.set_axis(['active', 'reactive', 'voltage', 'phase_angle', 'current'],
                   axis='columns', inplace=True)
    for i in range(0, chunk.shape[0], step):
        print("Seconds %d to %d\n" % (i, i+step-1))
        data = chunk.iloc[i:i+step][['active', 'reactive']]
        model.data = data
        model._edge_detection()
        # xdata = [x.isoformat(timespec='seconds') for x in data.index.time]
        # ax.plot(xdata, data['active'].values, 'b')
        # ax.tick_params(axis='x', bottom=False, labelbottom=False, labelrotation=45)
        # plt.pause(0.05)
        if model.online_edge_detected:
            pass
        current_sec += step
