
import mcm

class Visualizer(object):

    def __init__(simulator=None):
        self.sim = simulator
        if self.sim is not None:
            self.attach(self.sim)

    def attach(self, simulator):
        '''Attach this Visualizer to a Simulator Object. Whenever that Simulator updates,
        this Visualizer is alerted'''
        def newUpdate(sim):
            result = mcm.Simulator.update(sim)
            self.update()

        simulator.update = newUpdate

    def update(self):
        print('i need to revisualize!')
