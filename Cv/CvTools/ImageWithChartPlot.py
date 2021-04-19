from Cv.CvTools.ImagePlot import ImageDescriptionToken
from Cv.CvTools.ChartPlot import ChartDescriptionToken
from siki.basics import Exceptions
import matplotlib.pyplot as plt


class PltDiagramPlotCache(object):

    def __init__(self):
        self.tokens = []

    def add_chart(self, chart, title):
        token = ChartDescriptionToken(chart, title)
        self.tokens.append(token)

    def add_image(self, image, title):
        token = ImageDescriptionToken(image, title)
        self.tokens.append(token)

    def _plot_single(self):
        if isinstance(self.tokens[0], ChartDescriptionToken):
            plt.hist(self.tokens[0].chart, self.tokens[0].bins, color='fuchsia', alpha=0.5)
            plt.title(self.tokens[0].title)
        else:
            plt.imshow(self.tokens[0].img, cmap="gray")
            plt.title(self.tokens[0].title)
            plt.axis('off')

        # show images
        plt.show()

    def _plot_multi(self, nrows, ncols):
        if nrows * ncols != len(self.tokens):
            raise Exceptions.ArrayIndexOutOfBoundsException("Dimensions does not match the size of images")

        # iterate each image
        for i in range(nrows * ncols):
            plt.subplot(nrows, ncols, i + 1)

            if isinstance(self.tokens[i], ChartDescriptionToken):
                plt.hist(self.tokens[i].chart, self.tokens[i].bins, color='fuchsia', alpha=0.5)
                plt.title(self.tokens[i].title)
            else:
                plt.imshow(self.tokens[i].img, cmap="gray")
                plt.title(self.tokens[i].title)
                plt.axis('off')

        plt.show()

    def plots(self, nrows=0, ncols=0):
        if len(self.tokens) == 1:
            self._plot_single()

        if len(self.tokens) > 1:
            self._plot_multi(nrows, ncols)