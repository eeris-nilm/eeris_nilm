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

# Demo of edge detection without REST service implementation
import sys
import os
import dill
import datetime
import logging
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.table import table
from matplotlib.font_manager import FontProperties

from eeris_nilm.datasets import redd
from eeris_nilm.datasets import eco
from eeris_nilm.datasets import cenote
from eeris_nilm.datasets import eeris
from eeris_nilm.algorithms import livehart

logging.basicConfig(level=logging.DEBUG)


class Demo(object):
    # Change this to "zoom-in/zoom-out" in the graph
    TIME_WINDOW = 2*3600
    # How often should the model be saved?
    MODEL_SAVE_STEP = 100

    def __init__(self, path, date_start, date_end, ax, axt,
                 dataset='redd', step=3, inst_id=None, model_path_r=None,
                 model_path_w=None):
        self.step = step

        # Load data
        if dataset == 'redd':
            self.data, self.labels = redd.read_redd(path, date_start=date_start,
                                                    date_end=date_end,
                                                    get_channels=False)
            self.power = self.data['mains']
        elif dataset == 'eco':
            _, self.power = eco.read_eco(path, date_start, date_end)
        elif dataset == 'cenote':
            self.power = cenote.read_cenote(path, inst_id, date_start, date_end)
        elif dataset == 'eeris':
            self.power = eeris.read_eeris(path, date_start, date_end)
        else:
            raise ValueError("Unknown dataset %s" % (dataset))

        self.xdata, self.ydata = [], []
        self.ymatch = None
        self.start_day = None
        self.vis = None

        # Prepare model.
        self.model_path_r = model_path_r
        new_model = False
        if model_path_r is None:
            new_model = True
        else:
            try:
                with open(self.model_path_r, "rb") as fp_r:
                    self.model = dill.load(fp_r)
                self.start_ts = self.model.last_processed_ts + \
                    datetime.timedelta(seconds=1)
                date_start_ts = pd.Timestamp(date_start)
                self.start_sec = (self.start_ts - date_start_ts).seconds
            except IOError:
                print("Warning: Cannot read model file." +
                      "Creating model from scratch.")
                new_model = True
            else:
                new_model = False  # Not needed, for emphasis/readability
        if new_model:
            self.model = livehart.LiveHart(installation_id=1, store_live=True)
            self.start_ts = pd.Timestamp(date_start)
            self.start_sec = 0
        self.model_path_w = model_path_w
        self.prev = self.power['active'].iloc[0]

        # Plot parameters
        self.pause = False
        self.ax = ax
        self.line_active, = ax.plot([], [], 'b')
        self.line_est, = ax.plot([], [], 'r')
        self.line_match, = ax.plot([], [], 'm')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-100, 1000)
        self.ax.grid(True)
        self.time_window = min(self.TIME_WINDOW, self.model.MAX_DISPLAY_SECONDS)

        # Table plot parameters
        self.axt = axt
        self.axt.set_axis_off()

        # Model saving
        self.save_counter = 0

    # def on_click(self, event):
    #     self.pause ^= True

    def init(self):
        self.line_active.set_data([], [])
        self.line_est.set_data([], [])
        return (self.line_active, self.line_est)

    # def data_gen(self):
    #     self.power = self.power.loc[self.power.index > self.start_ts].dropna()
    #     end = self.power.shape[0] - self.power.shape[0] % self.step
    #     for i in range(0, end, self.step):
    #         data = self.power.iloc[i:i+self.step]
    #         yield self.start_sec + i, data
    def data_gen(self):
        self.power = self.power.loc[self.power.index > self.start_ts].dropna()
        end = self.power.index[-1]
        t = self.start_ts - datetime.timedelta(seconds=self.step)
        while t < end:
            t += datetime.timedelta(seconds=self.step)
            tnext = t+datetime.timedelta(seconds=self.step)
            data = self.power.loc[t:tnext]
            if data.shape[0] == 0:
                continue
            yield t, data

    def __call__(self, data):
        t, y = data
        if self.model._last_visualized_ts is None or self.vis is None:
            self.vis = t
        elif self.vis < self.model._last_visualized_ts:
            # This is to avoid cases with repeated past measurements. Should
            # perhaps be handled in livehart.py
            self.vis = self.model._last_visualized_ts
        if self.start_day is None:
            self.start_day = t.day
        try:
            self.model.update(y)
        except ValueError as err:
            logging.error('ERROR: %s' % (err))
            return None
        # Update lines
        bufidx = self.model._buffer.index
        tmpdata = self.model._buffer.loc[bufidx > self.vis]
        day = tmpdata.index.day - self.start_day
        self.xdata.extend((day * 3200 * 24 + tmpdata.index.hour * 3200 + tmpdata.index.minute * 60
                           + tmpdata.index.second).values.tolist())
        self.ydata.extend(tmpdata['active'].values.tolist())
        lim = min(len(self.xdata), self.time_window)
        self.line_active.set_data(self.xdata[-lim:], self.ydata[-lim:])
        lim1 = (bufidx[-1] - self.model._last_visualized_ts).seconds
        ydisp = self.model._yest[-(lim-lim1):].tolist()
        if lim1 > 0:
            self.line_est.set_data(self.xdata[-lim:-lim1], ydisp)
            ymatchdisp = self.model._ymatch['active'].values[-lim:-lim1].tolist()
            self.line_match.set_data(self.xdata[-lim:-lim1], ymatchdisp)
        else:
            self.line_est.set_data(self.xdata[-lim:], ydisp)
            ymatchdisp = self.model._ymatch['active'].values[-lim:].tolist()
            self.line_match.set_data(self.xdata[-lim:], ymatchdisp)
        # Update axis limits
        tidx = t.hour * 3200 + t.minute * 60 + t.second
        xmin, xmax = self.ax.get_xlim()
        xmin = max(0, tidx + self.step - self.time_window + 1)
        xmax = max(self.time_window, tidx + self.step)
        ymin = min(self.ydata[-self.time_window:] + [0])  # List concatenation
        ymax = max(self.ydata[-self.time_window:] + [0])  # List concatenation
        self.ax.set_xlim(xmin - 100, xmax + 100)
        self.ax.set_ylim(ymin - 50, ymax + 100)
        self.ax.figure.canvas.draw()
        # Add table
        if not self.model.live:
            cell_text = [['None', '-', '-']]
        else:
            cell_text = [[m.name, m.signature[0, 0], m.signature[0, 1]]
                         for m in self.model.live]
        cell_text.append(['Other', self.model.residual_live[0], '-'])
        cell_text.append(['Background', self.model.background_active, '-'])
        tab = table(self.axt, cell_text,
                    colLabels=['Appliance', 'Active', 'Reactive'],
                    cellLoc='left', colLoc='left', edges='horizontal')
        for (row, col), cell in tab.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(
                    fontproperties=FontProperties(weight='bold')
                )
        self.axt.clear()
        self.axt.add_table(tab)
        self.axt.set_axis_off()
        self.axt.figure.canvas.draw()
        # TODO (for dates)
        # self.xdata.extend(y.index.strftime('%Y-%m-%d %H:%M:%S').tolist())
        # Save model
        if self.model_path_w and \
           (self.save_counter % self.MODEL_SAVE_STEP == 0):
            with open(self.model_path_w, "wb") as fp:
                dill.dump(self.model, fp)
        self.save_counter += 1
        return self.line_active, \
            self.line_est, \
            self.line_match


# If enabled the saving hangs
# logging.basicConfig(level=logging.DEBUG)
if len(sys.argv) == 1:
    dataset = 'redd'
else:
    dataset = sys.argv[1]


# Edit these to fit your setup.

# The real-time NILM loop runs every 'step' measurements
step = 5

# DEBUG
step = 2*3600
# Whether to display on the screen or save to a video file. Change this to True
# to save a file
# save = True
save = False
# In case save is 'True' specify the output directory (which must exist)
video_out = 'demo_videos/'
inst_id = None
if dataset == 'redd':
    # Replace this with your path to house_1 of the downloaded and extracted
    # redd dataset
    p = 'tests/data/redd/house_1'
    house = os.path.basename(p)
    date_start = '2011-04-18T17:00'
    date_end = '2011-04-20T23:59'
    model_path_r = 'tests/data/model_redd_' + house + '.dill'
    model_path_w = 'tests/data/model_redd_' + house + '.dill'
elif dataset == 'eco':
    # Replace this with your path to the corresponding path to the downloaded
    # and extracted ECO dataset
    p = 'tests/data/eco/01_sm_csv/01'
    date_start = '2012-06-10T20:55'
    date_end = '2012-06-20T23:00'
    model_path_r = 'tests/data/model_eco.dill'
    model_path_w = 'tests/data/model_eco.dill'
# Cenote and eeris are for use with internal datasets
elif dataset == 'cenote':
    p = 'tests/data/pid7870cd27f3c_integration_20200113.csv'
    date_start = '2019-12-01T00:00'
    date_end = '2020-01-08T00:00'
    inst_id = '5e05d5c83e442d4f78db036f'
    model_path_r = 'tests/data/model_cenote.dill'
    model_path_w = 'tests/data/model_cenote.dill'
elif dataset == 'eeris':
    p = 'tests/data/eeris/5e05d5c83e442d4f78db036f_'
    date_start = '2020-12-01T00:00'
    date_end = '2020-12-02T23:59'
    # date_end = '2020-12-04T23:59'
    inst_id = '5e05d5c83e442d4f78db036f'
    model_path_r = 'tests/data/model_eeris.dill'
    model_path_w = 'tests/data/model_eeris.dill'

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ax = plt.subplot(2, 1, 1)
axt = plt.subplot(2, 1, 2)
# plt.ion()
d = Demo(p, date_start, date_end, ax, axt, dataset=dataset, step=step,
         inst_id=inst_id, model_path_r=model_path_r, model_path_w=model_path_w)
ani = animation.FuncAnimation(fig, d, frames=d.data_gen,
                              init_func=d.init, interval=50,
                              fargs=None, blit=False, repeat=False,
                              save_count=sys.maxsize)
if save:
    save_file = video_out + dataset + \
        date_start + '_' + date_end + '.mp4'
    ani.save(save_file,
             progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))
else:
    plt.show()
