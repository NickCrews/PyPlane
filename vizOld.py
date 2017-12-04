
import graphics as gs
import matplotlib.pyplot as plt

class Visualizer(object):

    DOT_SIZE = 100

    def __init__(self, model):
        self.model = model
        self.dotDict = {}
        self.rows, self.cols = self.model.plane.numRows, self.model.plane.seatsPerRow
        self.fig, self.ax = self.createFigure()

        self.dotSeries = None

    def createFigure(self):
        fig = plt.figure()
        ax = plt.axes(xlim= (-1, self.rows), ylim=(-1.5, self.cols+2), frameon=False)
        ax.autoscale(enable=False)
        ax.set_xticks([]), ax.set_yticks([])
        # add seat squares
        for yy in range(self.cols+1):
            color = 'grey' if yy != self.cols/2 else 'black'
            for xx in range(self.rows):
                    rect = plt.Rectangle((xx-0.5,yy-0.5), 0.95, 0.95, fc=color)
                    ax.add_patch(rect)
        # add overhead bin
        color = 'black'
        for xx in [x/2 for x in range(self.rows*2)]:
                rect = plt.Rectangle((xx-0.5,-1), 0.425, 0.425, fc=color)
                ax.add_patch(rect)
                rect = plt.Rectangle((xx-0.5,self.cols+0.5), 0.425, 0.425, fc=color)
                ax.add_patch(rect)

        '''
        #add jet bridge
        color = 'black'
        for xx in range(-25,0):
            rect = plt.Rectangle((xx-0.5,2.5),0.95,0.95, fc=color)
            ax.add_patch(rect)
            '''

        return fig, ax

    def show(self):
        ppl = [p for p in self.model.people if p.onPlane]
        for p in ppl:
            if p not in self.dotDict:
                self.dotDict[p] = Dot(p, self.model)

        dots = self.dotDict.values()
        xs = [d.row() for d in dots]
        ys = [d.col() for d in dots]
        cs = [d.color for d in dots]
        if self.dotSeries is None:
            self.dotSeries = self.ax.scatter(xs, ys, color=cs, s=self.DOT_SIZE, zorder=2)
        else:
            self.dotSeries.remove()
            self.dotSeries = self.ax.scatter(xs, ys, color=cs, s=self.DOT_SIZE, zorder=2)

        self.fig.show()
        plt.waitforbuttonpress()

class Dot(object):

    colorMaps = {}
    colorIndices = {}

    def __init__(self, person, model):
        self.person = person

        # deal with color
        if model not in type(self).colorMaps:
            # make a rainbow divided into npeople colors
            npeople = len(model.people)
            cmap = plt.cm.get_cmap('hsv', npeople)
            # save that colormap to this model
            type(self).colorMaps[model] = cmap

        # every time we create a dot, get the next color from the colormap
        if model not in type(self).colorIndices:
            type(self).colorIndices[model] = 0
        else:
            type(self).colorIndices[model] += 1

        cmap = type(self).colorMaps[model]
        cindex =type(self).colorIndices[model]
        self.color = cmap(cindex)

    def row(self):
        return self.person.row

    def col(self):
        return self.person.col
