"""
Demo with animation


Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

# Demo of edge detection without REST service implementation
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.table import table
from matplotlib.font_manager import FontProperties
from eeris_nilm.datasets import eco
from eeris_nilm.algorithms import hart
import datetime


class Demo(object):
    TIME_WINDOW = 1200
    MODEL_SAVE_STEP = 20

    def __init__(self, path, date_start, date_end, ax, axt,
                 model_path_r=None, model_path_w=None):
        # Load data
        self.step = 5
        self.phase_list, self.power = eco.read_eco(path, date_start, date_end)
        self.xdata, self.ydata, self.ydata_r = [], [], []
        self.ymatch = None

        # Prepare model.
        # TODO: Don't keep the files open for writing. Use them when needed.
        self.model_path_r = model_path_r
        if model_path_r is None:
            self.model = hart.Hart85eeris(installation_id=1)
            self.start_ts = date_start
        else:
            with open(self.model_path_r, "rb") as fp_r:
                self.model = pickle.load(fp_r)
            self.start_ts = self.model._last_processed_ts + \
                datetime.timedelta(seconds=1)
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
                   self.ydata_r[-self.time_window:])
        ymax = max(self.ydata[-self.time_window:] +
                   self.ydata_r[-self.time_window:])
        self.ax.set_xlim(xmin - 100, xmax + 100)
        self.ax.set_ylim(ymin - 50, ymax + 100)
        self.ax.figure.canvas.draw()
        # Add table
        if not self.model.live:
            cell_text = [['None', '-', '-']]
        else:
            cell_text = [[m.name, m.signature[0], m.signature[1]]
                         for m in self.model.live]
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
                pickle.dump(self.model, fp)
        self.save_counter += 1
        return self.line_active, \
            self.line_reactive, \
            self.line_est, \
            self.line_match


# Setup
# p = '/media/data/datasets/NILM/ECO/02_sm_csv/02'
p = 'tests/data/01_sm_csv/01'
date_start = '2012-06-10T00:00'
date_end = '2012-06-20T23:59'
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ax = plt.subplot(2, 1, 1)
axt = plt.subplot(2, 1, 2)
model_path_r = 'tests/data/model.pickle'
model_path_w = 'tests/data/model.pickle'
d = Demo(p, date_start, date_end, ax, axt, model_path_r=model_path_r,
         model_path_w=model_path_w)
ani = animation.FuncAnimation(fig, d, frames=d.data_gen,
                              init_func=d.init, interval=50,
                              fargs=None, blit=False, repeat=False,
                              save_count=sys.maxsize)
plt.show()
