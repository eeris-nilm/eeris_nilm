"""
Demo with animation


Until we decide on copyright & licensing issues:

Written by Christos Diou <diou@auth.gr>
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

# Demo of edge detection without REST service implementation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import eco
from eeris_nilm.hart85_eeris import Hart85eeris


class Demo(object):
    TIME_WINDOW = 1200

    def __init__(self, ax):
        # Load data
        p = 'tests/data/01_sm_csv/01'
        self.date_start = '2012-06-19T11:00'
        self.date_end = '2012-06-19T23:59'
        self.step = 5
        self.phase_list, self.power = eco.read_eco(p, self.date_start, self.date_end)
        self.xdata, self.ydata = [], []
        self.ymatch = None

        # Prepare model
        self.model = Hart85eeris(installation_id=1)
        self.current_sec = 0
        self.prev = self.power['active'].iloc[0]

        # Plot parameters
        self.ax = ax
        self.line_orig, = ax.plot([], [], 'b')
        self.line_est, = ax.plot([], [], 'r')
        self.line_match, = ax.plot([], [], 'm')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-100, 1000)
        self.ax.grid(True)
        self.time_window = min(self.TIME_WINDOW, self.model.MAX_DISPLAY_SECONDS)

    def init(self):
        self.line_orig.set_data([], [])
        self.line_est.set_data([], [])
        return (self.line_orig, self.line_est)

    def data_gen(self):
        lim = self.power.shape[0] - self.power.shape[0] % self.step
        for i in range(0, lim, self.step):
            # DEBUG
            # print("Seconds %d to %d\n" % (self.current_sec, self.current_sec +
            # self.step))
            data = self.power.iloc[i:i+self.step]
            yield i, data

    def __call__(self, data):
        t, y = data
        self.model.data = y
        self.model.detect_edges_hart()
        self.model._match_edges_hart()
        self.model._update_live()
        self.current_sec += self.step
        # Update lines
        self.xdata.extend(list(range(t, t + self.step)))
        self.ydata.extend(y['active'].values.tolist())
        lim = max(len(self.xdata), self.time_window)
        self.line_orig.set_data(self.xdata[-lim:], self.ydata[-lim:])
        self.line_est.set_data(self.xdata[-lim:], self.model._yest.tolist()[-lim:])
        self.line_match.set_data(self.xdata[-lim:], self.model._ymatch.tolist()[-lim:])
        # Update axis limits
        xmin, xmax = self.ax.get_xlim()
        xmin = max(0, t - self.time_window)
        xmax = max(self.time_window, t + self.step)
        ymin = min(self.ydata[-self.time_window:])
        ymax = max(self.ydata[-self.time_window:])
        self.ax.set_xlim(xmin - 100, xmax + 100)
        self.ax.set_ylim(ymin - 50, ymax + 100)
        self.ax.figure.canvas.draw()
        # TODO (for dates)
        # self.xdata.extend(y.index.strftime('%Y-%m-%d %H:%M:%S').tolist())
        return self.line_orig, self.line_est, self.line_match


fig = plt.figure()
ax = plt.subplot(1, 1, 1)
d = Demo(ax)
ani = animation.FuncAnimation(fig, d, frames=d.data_gen, init_func=d.init, interval=0,
                              fargs=None, blit=False, repeat=False)
plt.show()
