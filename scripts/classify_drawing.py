import numpy as np
import matplotlib.pyplot as plt

from model.mlp_classifier import mlp_classifier

class drawing_window:
    def __init__(self, model):
        self.model = model

    def prepare_gui(self):
        self.img = np.zeros((28, 28))
        self.drawing = {"pressed": False}

        self.fig, self.ax = plt.subplots()
        plt.title('Digit Recogniser (0-9)')
        self.im = self.ax.imshow(self.img, cmap="gray", vmin=0, vmax=1) # show array as an image

        self.ax.set_xticks(np.arange(-0.5, 28, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, 28, 1), minor=True)
        self.ax.grid(which="minor", linestyle="-", linewidth=0.5) #minor ticks => gridlines

        self.ax.set_xticks([])
        self.ax.set_yticks([]) # remove major ticks

    def set_pixel(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        j = int(event.xdata)  # column index
        i = int(event.ydata)  # row index

        if 0 <= i < 28 and 0 <= j < 28:
            self.img[i, j] = min(self.img[i, j] + 0.7, 1.0)

            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for ni, nj in neighbors:
                if 0 <= ni < 28 and 0 <= nj < 28:
                    self.img[ni, nj] = min(self.img[ni, nj] + 0.4, 1.0)

            self.im.set_data(self.img)
            self.fig.canvas.draw_idle()

    def predict(self):
        x = self.img.ravel()
        x = x.reshape((784, 1))
        y, prob = self.model.predict(x)
        
        yLabel = [str(i) + ": " + str(round(prob[i][0], 2)) for i in range(10)]
        yLabel = "\n".join(yLabel)
        plt.xlabel(f"Predicted digit: {y[0][0]}")
        plt.ylabel(yLabel, rotation=0, labelpad=20)

    def run(self):
        self.prepare_gui()
        
        def on_press(event):
            self.drawing["pressed"] = True
            self.set_pixel(event)

        def on_release(event):
            self.drawing["pressed"] = False

        def on_move(event):
            if self.drawing["pressed"]:
                self.set_pixel(event)
                self.predict()

        self.fig.canvas.mpl_connect("button_press_event", on_press)
        self.fig.canvas.mpl_connect("button_release_event", on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", on_move)

        plt.show()