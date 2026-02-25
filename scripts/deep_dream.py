import numpy as np
import matplotlib.pyplot as plt

class deep_dream:
    def __init__(self, mnist_dreamer, pattern):
        self.mnist_dreamer = mnist_dreamer
        self.pattern = pattern

    def prepare_gui(self):
        self.fig, self.ax = plt.subplots(4, 3, figsize=(7.3, 7.3))
        self.fig.set_facecolor('gray')
        

        for i in range(4):
            for j in range(3):
                self.ax[i][j].set_axis_off()

    def dream(self):
        self.prepare_gui()

        self.ax[0][0].imshow(self.pattern.reshape((28, 28)), cmap='gray', vmin=0, vmax=1)
        self.ax[0][0].set_title('Pattern')

        self.ax[0][1].imshow(np.array([[0.5]]), cmap='gray', vmin=0, vmax=1)

        ax_id = 2

        for i in range(10):
            img = self.mnist_dreamer.generate(i, self.pattern)
            
            working_ax = self.ax[ax_id // 3][ax_id % 3]

            working_ax.imshow(img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            working_ax.set_title(f'Dreamed {i}')
            
            ax_id += 1

        plt.show()