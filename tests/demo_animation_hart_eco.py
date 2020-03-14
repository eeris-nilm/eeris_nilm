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
import dill
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.table import table
from matplotlib.font_manager import FontProperties
from eeris_nilm.datasets import eco
from eeris_nilm.algorithms import livehart
import datetime
import logging


class Demo(object):
    TIME_WINDOW = 1200
    MODEL_SAVE_STEP = 20

    def __init__(self, path, date_start, date_end, ax, axt,
                 model_path_r=None, model_path_w=None):
        # Load data
        self.step = 7
        _, self.power = eco.read_eco(path, date_start, date_end)
        self.xdata, self.ydata, self.ydata_r = [], [], []
        self.ymatch = None

        # Prepare model.
        # TODO: Don't keep the files open for writing. Use them when needed.
        self.model_path_r = model_path_r
        new_model = False
        if model_path_r is None:
            new_model = True
        else:
            try:
                with open(self.model_path_r, "rb") as fp_r:
                    self.model = dill.load(fp_r)
                self.start_ts = self.model._last_processed_ts + \
                    datetime.timedelta(seconds=1)
            except IOError:
                print("Warning: Cannot read model file." +
                      "Creating model from scratch.")
                new_model = True
            else:
                new_model = False  # Not needed, for emphasis/readability
        if new_model:
            self.model = livehart.LiveHart(installation_id=1)
            self.start_ts = date_start
        self.model_path_w = model_path_w
        self.current_sec = 0
        self.prev = self.power['active'].iloc[0]

        # Plot parameters
        self.pause = False
        self.ax = ax
        self.line_active, = ax.plot([], [], 'b')
        self.line_reactive, = ax.plot([], [], 'c')
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
        self.line_reactive.set_data([], [])
        self.line_est.set_data([], [])
        return (self.line_active, self.line_reactive, self.line_est)

    def data_gen(self):
        self.power = self.power.loc[self.power.index > self.start_ts]
        end = self.power.shape[0] - self.power.shape[0] % self.step
        for i in range(0, end, self.step):
            data = self.power.iloc[i:i+self.step]
            yield i, data

    def __call__(self, data):
        t, y = data
        self.model.update(y)
        self.current_sec += self.step
        # Update lines
        self.xdata.extend(list(range(t, t + self.step)))
        self.ydata.extend(y['active'].values.tolist())
        self.ydata_r.extend(y['reactive'].values.tolist())
        lim = min(len(self.xdata), self.time_window)
        self.line_active.set_data(self.xdata[-lim:], self.ydata[-lim:])
        self.line_reactive.set_data(self.xdata[-lim:], self.ydata_r[-lim:])
        self.line_est.set_data(self.xdata[-lim:],
                               self.model._yest.tolist()[-lim:])
        self.line_match.set_data(self.xdata[-lim:],
                                 self.model._ymatch.tolist()[-lim:])
        # Update axis limits
        xmin, xmax = self.ax.get_xlim()
        xmin = max(0, t - self.time_window)
        xmax = max(self.time_window, t + self.step)
        ymin = min(self.ydata[-self.time_window:] +
                   self.ydata_r[-self.time_window:])  # List concatenation
        ymax = max(self.ydata[-self.time_window:] +
                   self.ydata_r[-self.time_window:])  # List concatenation
        self.ax.set_xlim(xmin - 100, xmax + 100)
        self.ax.set_ylim(ymin - 50, ymax + 100)
        self.ax.figure.canvas.draw()
        # Add table
        if not self.model.live:
            cell_text = [['None', '-', '-']]
        else:
            cell_text = [[m.name, m.signature[0], m.signature[1]]
                         for m in self.model.live]
        cell_text.append(['Other', self.model.residual_live[0], '-'])
        cell_text.append(['Background', self.model.background_active, '-'])
        # cell_text.append(['Avg Power', self.model.running_avg_power[0], '-'])
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
            self.line_reactive, \
            self.line_est, \
            self.line_match


logging.basicConfig(level=logging.DEBUG)

# Setup
# p = '/media/data/datasets/NILM/ECO/02_sm_csv/02'
p = 'tests/data/01_sm_csv/01'
date_start = '2012-06-10T20:55'
date_end = '2012-06-20T23:00'
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ax = plt.subplot(2, 1, 1)
axt = plt.subplot(2, 1, 2)
model_path_r = 'tests/data/model_eco.dill'
model_path_w = 'tests/data/model_eco.dill'
d = Demo(p, date_start, date_end, ax, axt, model_path_r=model_path_r,
         model_path_w=model_path_w)
ani = animation.FuncAnimation(fig, d, frames=d.data_gen,
                              init_func=d.init, interval=1,
                              fargs=None, blit=False, repeat=False,
                              save_count=sys.maxsize)
plt.show()
