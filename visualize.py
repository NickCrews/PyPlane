
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView
import numpy as np
import multiprocessing

import mcm

class Visualizer(object):

    def __init__(self, simulator=None):
        self.sim = simulator
        if self.sim is not None:
            self.attach(self.sim)

        self.curves = ([],[])

        self.q = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.run)
        p.start()

    def run(self):
        app = QtGui.QApplication([])
        mw = QtGui.QMainWindow()
        mw.resize(800, 600)
        view = pg.GraphicsLayoutWidget()
        mw.setCentralWidget(view)
        mw.setWindowTitle('Plane')
        plotItem = view.addPlot()
        scatter = pg.ScatterPlotItem(pxMode=False)   ## Set pxMode=False to allow spots to transform with the view
        plotItem.addItem(scatter)
        plotItem.setRange(xRange=(0,self.sim.plane.numRows-1), yRange=(-1,self.sim.plane.seatsPerRow+1))
        mw.show()

        def update():
            if not self.q.empty():
                spots = self.q.get()
                scatter.setData(spots=spots)
                # plotItem.repaint()

        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(1000//300)
        ## Start Qt event loop
        QtGui.QApplication.exec_()


    def attach(self, simulator):
        '''Attach this Visualizer to a Simulator Object. Whenever that Simulator updates,
        this Visualizer is alerted

        This is done by monkeypatching in a new update() method for the sim
        that includes a call to Visualizer.show()'''
        self.sim = simulator
        def newUpdate():
            result = mcm.Simulator.update(simulator)
            self.show()
            return result
        simulator.update = newUpdate

    def detach(self):
        self.sim.update = mcm.Simulator.update(self.sim)

    def show(self):
        spots = []
        for p in self.sim.passengers:
            info = (p.row, p.col,p.isSeated,)
            pos = (p.row, p.col)
            # fillColor = pg.intColor(p.row, 100)
            if p.isSeated:
                fillColor = 'g'
            elif p.is_bag_up:
                fillColor = 'b'
            else:
                fillColor = 'r'
            edgeColor = 'k' if p.isSeated else 'w'
            edgeWidth = max(p.bagTimer, 1)
            spot = {'pos': pos, 'size': 1, 'pen': {'color': edgeColor, 'width': edgeWidth}, 'brush':fillColor}
            spots.append(spot)
        self.q.put(spots)
