from PySide6.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class FigureComponent(QVBoxLayout):
    def __init__(self, axis_off=True, projection3d=False):
        super().__init__()
        self.axis_off = True
        self.widget()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d') if projection3d else self.fig.add_subplot()
        if axis_off:
            self.ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self.setContentsMargins(0, 0, 0, 0)
        self.canvas = FigureCanvasQTAgg(self.fig)
        tb = NavigationToolbar2QT(self.canvas)
        self.addWidget(self.canvas)
        self.addWidget(tb)

    def reset(self):
        self.ax.cla()
        self.canvas.draw()


class AnimatedComponent(FigureComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ani = None

    def reset(self):
        if self.ani is not None:
            self.ani.event_source.stop()
        super().reset()
        self.ani = None

    def _update_plot(self, num):
        pass

    def update_plot(self, num):
        self.ani.frame_seq = self.ani.new_frame_seq()
        self.ani.event_source.stop()
        self._update_plot(num)
        self.canvas.draw()


class AnimatedLine2D(AnimatedComponent):
    def __init__(self):
        super().__init__(axis_off=False, projection3d=False)
        self.x_data = None
        self.y_data = None
        self.line = None
        self.ani = None
        plt.tight_layout()
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

    def new(self, x_data, y_data):
        self.reset()
        self.x_data = x_data
        self.y_data = y_data
        self.ax.set_aspect('equal')
        self.line, = self.ax.plot([], [])
        self.ax.set_xlabel('Time elapse')
        self.ax.set_ylabel('Crystal area')
        self.ax.set_xlim(0, max(x_data))
        self.ax.set_ylim(0, max(y_data) * 1.25)
        self.ax.set_aspect(1 / self.ax.get_data_ratio(), adjustable='box')

        def init():
            self.line.set_data([], [])
            return self.line,

        self.ani = animation.FuncAnimation(self.fig, self._update_plot, len(x_data), init,
                                           interval=1, repeat=True)
        self.canvas.draw()

    def _update_plot(self, num):
        self.line.set_data(self.x_data[:num], self.y_data[:num])
        return self.line,


class MultiPage(AnimatedComponent):
    def __init__(self):
        super().__init__(axis_off=True, projection3d=False)
        self.stack = None
        self.im = None
        self.ani = None

    def new(self, stack):
        self.reset()
        self.stack = stack
        self.im = self.ax.imshow(stack[0])
        self.ani = animation.FuncAnimation(self.fig, self._update_plot, len(stack),
                                           interval=1, repeat=True)
        self.canvas.draw()

    def _update_plot(self, num):
        self.im.set_array(self.stack[num])
        return [self.im]


class AnimatedLines3D(AnimatedComponent):
    def __init__(self):
        super().__init__(axis_off=False, projection3d=True)
        self.walks = None
        self.timescale = None
        self.lines = None
        self.ani = None

    def new(self, timescale, walks):
        self.reset()
        self.walks = walks
        self.timescale = timescale

        x_max, y_max = 0, 0
        for w in walks:
            x_max = max(x_max, w[0].max())
            y_max = max(y_max, w[1].max())

        # Create lines initially without data
        self.lines = [self.ax.plot([], [], [])[0] for _ in self.walks]

        # Setting the axes properties
        self.ax.set(xlim3d=(0, x_max), xlabel='X')
        self.ax.set(ylim3d=(0, y_max), ylabel='Y')
        self.ax.set(zlim3d=(0, len(timescale) - 1), zlabel='Timestamp')

        self.ani = animation.FuncAnimation(
            self.fig, self._update_plot, len(timescale), interval=1, repeat=True)
        self.canvas.draw()

    def _update_plot(self, num):
        for line, walk in zip(self.lines, self.walks):
            n = int(num - walk[2, 0])
            if n > 0:
                line.set_data(walk[:2, :n])
                line.set_3d_properties(walk[2, :n])
            else:
                line.set_data([], [])
                line.set_3d_properties([])
        return self.lines