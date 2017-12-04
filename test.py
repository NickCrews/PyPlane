from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView
app = pg.mkQApp()

print('here')
## Create the widget
v = RemoteGraphicsView(debug=False)  # setting debug=True causes both processes to print information
                                # about interprocess communication
print('now herers')
v.show()
v.setWindowTitle('pyqtgraph example: RemoteGraphicsView')
print('3')
## v.pg is a proxy to the remote process' pyqtgraph module. All attribute
## requests and function calls made with this object are forwarded to the
## remote process and executed there. See pyqtgraph.multiprocess.remoteproxy
## for more inormation.
plt = v.pg.PlotItem()
print('adfsd')
v.setCentralItem(plt)
print('4')
x,y = [1,4,2,3,6,2,3,4,2,3],[1,2,3,5,7,4,3,5,8,9]
plt.plot(x,y, pen=None, symbol='o',symbolSize=0.2, pxMode=False)
print('5')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
