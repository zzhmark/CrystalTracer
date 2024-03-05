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
        self.ax.clear()
        if self.axis_off:
            self.ax.axis('off')
        self.canvas.draw()


class AnimatedFigureComponent(FigureComponent):
    def __init__(self):
        super().__init__(axis_off=False, projection3d=True)
        self.walks = None
        self.lines = None
        self.timescale = None
        self.ani = None

    def new(self, timescale, walks):

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
            self.fig, self.update_plot, len(timescale), interval=25, repeat=False)
        self.canvas.draw()

    def update_plot(self, num):
        for line, walk in zip(self.lines, self.walks):
            n = int(num - walk[2, 0])
            if n > 0:
                line.set_data(walk[:2, :n])
                line.set_3d_properties(walk[2, :n])
            else:
                line.set_data([], [])
                line.set_3d_properties([])
        return self.lines
