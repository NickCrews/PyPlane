
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView
import numpy as np

import mcm

class Visualizer(object):

    def __init__(self, simulator=None):
        self.sim = simulator
        if self.sim is not None:
            self.attach(self.sim)

        self.dotDict = {}
        self.rows, self.cols = self.sim.plane.numRows, self.sim.plane.seatsPerRow



        app = pg.mkQApp()

        ## Create the widget
        v = RemoteGraphicsView(debug=False)  # setting debug=True causes both processes to print information
                                        # about interprocess communication
        v.show()
        v.setWindowTitle('pyqtgraph example: RemoteGraphicsView')
        ## v.pg is a proxy to the remote process' pyqtgraph module. All attribute
        ## requests and function calls made with this object are forwarded to the
        ## remote process and executed there. See pyqtgraph.multiprocess.remoteproxy
        ## for more inormation.
        self.plt = v.pg.PlotItem()
        v.setCentralItem(self.plt)
        x,y = [1,4,2,3,6,2,3,4,2,3],[1,2,3,5,7,4,3,5,8,9]
        # self.plt.plot(x,y, pen=None, symbol='o',symbolSize=0.2, pxMode=False)
        ## Start Qt event loop unless running in interactive mode or using pyside.
        QtGui.QApplication.instance().exec_()

    def attach(self, simulator):
        '''Attach this Visualizer to a Simulator Object. Whenever that Simulator updates,
        this Visualizer is alerted

        This is done by monkeypatching in a new update() method for the sim
        that includes a call to '''
        def newUpdate():
            result = mcm.Simulator.update(simulator)
            self.show()

        simulator.update = newUpdate

    def show(self):
        print('sjowing!')
        ppl = [p for p in self.sim.passengers if p.onPlane]
        for p in ppl:
            if p not in self.dotDict:
                self.dotDict[p] = Dot(p, self.sim)

        dots = self.dotDict.values()
        xs = [d.row() for d in dots]
        ys = [d.col() for d in dots]
        # cs = [d.color for d in dots]
        self.plt.plot(xs,ys, pen=None, symbol='o',symbolSize=0.2, pxMode=False)


class Dot(object):

    colorMaps = {}
    colorIndices = {}

    def __init__(self, passenger, sim):
        self.passenger = passenger

        # deal with color
        # if sim not in type(self).colorMaps:
        #     # make a rainbow divided into npeople colors
        #     npeople = len(sim.passengers)
        #     cmap = plt.cm.get_cmap('hsv', npeople)
        #     # save that colormap to this model
        #     type(self).colorMaps[model] = cmap
        #
        # # every time we create a dot, get the next color from the colormap
        # if model not in type(self).colorIndices:
        #     type(self).colorIndices[model] = 0
        # else:
        #     type(self).colorIndices[model] += 1
        #
        # cmap = type(self).colorMaps[model]
        # cindex =type(self).colorIndices[model]
        # self.color = cmap(cindex)

    def row(self):
        return self.passenger.row

    def col(self):
        return self.passenger.col
