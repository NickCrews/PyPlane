#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:37:27 2017

@author: MacobJiller
# and nick crews
"""
#import libs
import random
import numpy as np
# import pandas as pd
import visualize
'''
Airbus320NEO cabin length = 90ft
numSeats = 150
90/25 = 3.6 ft/row

one passenger: mu = 14s, sigma = 1.96
two ppl: mu = 17s, sigma = 2.62

passenger velocity = 4.6 ft/s OR 1.28 row/s

bins = 49 bins / 90 ft

1.8 bins per ft, 2 bins per row


'''
# import numpy as np
# random.seed(42)
class Plane(object):

    def __init__(self, numRows=25, seatsPerRow=6):
        self.numRows = numRows
        self.seatsPerRow = seatsPerRow
        self.reset()

    def reset(self):
        self.seats = [[None]*(self.seatsPerRow+1) for i in range(self.numRows)]
        self.overhead = [[False]*self.numRows, [False]*self.numRows]
        self.nonSeated = []

    def generateAllSeats(self):
        listTuples = []
        letters = [chr(i) for i in range(ord('a'),ord('z')+1)][:self.seatsPerRow]
        for seatNum in range(self.numRows):
            for letter in letters:
                seat_append = (seatNum,letter)
                listTuples.append(seat_append)
        return listTuples

    def __repr__(self):
        seats = ""
        for i, row in enumerate(self.seats):
            seats += str(i)
            seats += str(row)
            seats += '\n'
        seats = seats[:-1]
        return "Plane with the layout \n{}".format(seats)

    def overheadEmpty(self,row):
        isLeftFull =  self.overhead[0][row]
        isRightFull = self.overhead[1][row]
        return not isLeftFull or not isRightFull

    def stowBag(self, row):
        if not self.overhead[0][row]:
            self.overhead[0][row] = True
        elif not self.overhead[0][row]:
            self.overhead[1][row] = True
        else:
            pass
            # raise ValueError('no space to add that bag')

    def isBoardingSpotEmpty(self):
        firstRow = self.seats[0]
        return firstRow[self.seatsPerRow//2] == None

    def inAisle(self, passenger):
        seatNum = self.letter2number(passenger.seat[1])
        aisle = self.seatsPerRow//2
        if seatNum < aisle:
            numPpl = 0
            for spot in self.seats[passenger.row][seatNum+1:aisle]:
                if spot is not None:
                    numPpl+=1
            return numPpl
        else:
            numPpl = 0
            for spot in self.seats[passenger.row][aisle+1:seatNum]:
                if spot is not None:
                    numPpl+=1
            return numPpl


    def addPassenger(self, passenger):
        firstRow = self.seats[0]
        firstRow[self.seatsPerRow//2] = passenger
        self.nonSeated.append(passenger)

    def findPassenger(self, passenger):
        for row in self.seats:
            if passenger in row:
                return (self.seats.index(row), row.index(passenger))

    def isEmptyInFront(self, rowNumber):
        if rowNumber+1 < self.numRows:
            row = self.seats[rowNumber+1]
            return row[self.seatsPerRow//2] == None
        else:
            return True

    def moveForward(self, passenger):
        row = passenger.row
        self.seats[row][self.seatsPerRow//2] = None
        if row+1 < self.numRows:
            self.seats[row+1][self.seatsPerRow//2] = passenger

    def sitDown(self, passenger):
        rowNumber, letter = passenger.seat
        row = self.seats[rowNumber]
        row[self.seatsPerRow//2] = None
        seatNumber = self.letter2number(letter)
        row[seatNumber] = passenger
        self.nonSeated.remove(passenger)

    def letter2number(self,letter):
        raw = ord(letter) - ord('a')
        if raw >= self.seatsPerRow/2:
            raw += 1
        return raw

    def isFull(self):
        spr = self.seatsPerRow
        for row in self.seats:
            if row[spr//2] is not None:
                return False
            if None in row[:spr//2]:
                return False
            if None in row[spr//2+1:]:
                return False
        return True

class BasePassenger(object):
    #need better numbers here
    # TIME_PER_BAG_MU    = 8
    # TIME_PER_BAG_SIGMA = 2
    #
    # TIME_AISLE_ONE_MU = 14
    # TIME_AISLE_ONE_SIGMA = 1.96
    # TIME_AISLE_TWO_MU = 17
    # TIME_AISLE_TWO_SIGMA = 2.62

    def __init__(self, plane, seat):
        self.plane = plane
        self.seat = seat

        self.TIME_PER_BAG = random.normalvariate(self.TIME_PER_BAG_MU, self.TIME_PER_BAG_SIGMA)
        self.TIME_PER_AISLE_ONE = random.normalvariate(self.TIME_AISLE_ONE_MU,self.TIME_AISLE_ONE_SIGMA)
        self.TIME_PER_AISLE_TWO = random.normalvariate(self.TIME_AISLE_TWO_MU,self.TIME_AISLE_TWO_SIGMA)

        self.reset()

    def reset(self):
        self.row = 0
        self.col = self.plane.seatsPerRow//2
        self.isSeated = False
        self.onPlane = False
        self.numBags = 2
        self.satisfaction = 100

        self.bagTimer = -1
        self.is_bag_up = False

        self.aisleTimer = None

    def __repr__(self):
        return "Passenger: Seat {} at {}".format(self.seat, self.row)

    def atRow(self):
        return self.seat[0] == self.row

    def act(self):
        if self.isSeated:
            return False
        if self.atRow():
            if self.is_bag_up:
                if self.aisleTimer == None:
                    self.aisleTimer = (self.TIME_PER_AISLE_ONE if self.plane.inAisle(self) == 1 else self.TIME_PER_AISLE_TWO if self.plane.inAisle(self) == 2 else 0)
                #    print(self.aisleTimer)
                elif self.aisleTimer <= 0:
                    self.sitDown()
                    return True
                else:
                    self.aisleTimer-=1
            if not self.is_bag_up:
            #    print(self, 'at row!')
                if self.numBags > 0:
                    if self.plane.overheadEmpty(self.row):
                        self.plane.stowBag(self.row)
                        self.numBags = 0
                        self.bagTimer = self.TIME_PER_BAG
                if self.bagTimer <= 0:
                    self.is_bag_up = True
                    return True
                else:
                    self.bagTimer -= 1
        elif self.plane.isEmptyInFront(self.row):
            self.plane.moveForward(self)
            self.row += 1
            return True
        return False

    def sitDown(self):
        self.isSeated = True
        self.col = self.plane.letter2number(self.seat[1])
        self.plane.sitDown(self)

class DefaultPassenger(BasePassenger):
    #need better numbers here
    TIME_PER_BAG_MU    = 8
    TIME_PER_BAG_SIGMA = 2

    TIME_AISLE_ONE_MU = 14
    TIME_AISLE_ONE_SIGMA = 1.96
    TIME_AISLE_TWO_MU = 17
    TIME_AISLE_TWO_SIGMA = 2.62

class Simulator(object):

    DEFAULT_PARAMS = {'planeType':Plane, 'passengerType':DefaultPassenger, 'boardingStrategy':'random'}

    def __init__(self, params=DEFAULT_PARAMS):
        self.PLANE_TYPE = params['planeType']
        self.PASSENGER_TYPE = params['passengerType']

        self.plane = self.PLANE_TYPE()
        seats = self.plane.generateAllSeats()
        self.passengers = [self.PASSENGER_TYPE(self.plane, seat) for seat in seats]
        self.boardingStrategy = params['boardingStrategy']

        self.ticks = 0
        self.queue = list(self.passengers)
        self.order()

    def reset(self):
        self.plane.reset()
        seats = self.plane.generateAllSeats()
        self.passengers = [self.PASSENGER_TYPE(self.plane, seat) for seat in seats]
        self.ticks = 0
        self.queue = self.passengers.copy()
        self.order()

    def run(self):
        # print('starting run with')
        # print(self.plane)
        # print(self.passengers)
        # print(self.queue)
        # print(self.ticks)
        while not self.plane.isFull():
            self.ticks+=1
            self.update()

        # print("done.")
        # print(self.plane)
        return self.ticks

    def update(self):
        # print('updating!')
        for p in self.plane.nonSeated:
            p.act()

        if self.plane.isBoardingSpotEmpty():
            if len(self.queue) > 0:
                p = self.queue.pop(0)
                p.onPlane = True
                self.plane.addPassenger(p)

    #defining various boarding procedures
    def order(self):
        if self.boardingStrategy == 'random':
            random.shuffle(self.queue)
        if self.boardingStrategy == 'backFirst':
            self.queue.sort(key = lambda passenger : passenger.seat[0], reverse=True)
        if self.boardingStrategy == 'frontFirst':
            self.queue.sort( key = lambda passenger : passenger.seat[0])

class Model(object):

    SENSITIVITY_FILENAME = 'sensitivity.txt'

    def runSensitivityAnalysis(self):
        passTypes = self.createPassengerTypes()
        inputs = [{'planeType':Plane, 'passengerType':pt, 'boardingStrategy':'random'} for pt in passTypes]
        # resLists = [['BagTime'],['AisleOne'],['AisleTwo'],['BoardingTime']]
        variables = ['BagTime', 'AisleOne', 'AisleTwo','BoardingTime']
        resultDict = {var:[] for var in variables}
        print('testing on {} inputs'.format(len(inputs)))
        for i, inp in enumerate(inputs):
            print('running input number {}'.format(i))
            results = self.test(params=inp, convergenceThreshold=.01)
            avgResult = avg(results)
            resultDict['BoardingTime'].append(avgResult)
            pt = inp['passengerType']
            resultDict['BagTime'].append(pt.TIME_PER_BAG_MU)
            resultDict['AisleOne'].append(pt.TIME_AISLE_ONE_MU)
            resultDict['AisleTwo'].append(pt.TIME_AISLE_TWO_MU)
            # variables = [pt.TIME_PER_BAG_MU, pt.TIME_AISLE_ONE_MU, pt.TIME_AISLE_TWO_MU]
            # # make our input dict hashable
            # list_data = [[x]*len(result) for x in list_inputs]
            # list_data.append(result)
            # for num,data in enumerate(list_data):
            #     resLists[num] += data
        return resultDict

    def getAnalysisData(self):
        try:
            data = self.openSensitivityAnalysis()
            return data
        except:
            data = self.runSensitivityAnalysis()
            self.saveSensitivityAnalysis(data)
            return data

    @staticmethod
    def saveSensitivityAnalysis(resultDict):
        with open(Model.SENSITIVITY_FILENAME, 'w') as f:
            for varName, series in resultDict.items():
                f.write(varName + '\t')
                for val in series:
                    f.write(str(val) + '\t')
                f.write('\n')

    @staticmethod
    def openSensitivityAnalysis():
        with open(Model.SENSITIVITY_FILENAME, 'r') as f:
            lines = f.readlines()
        dataDict = {}
        for line in lines:
            data = line.strip().split()
            dataDict[data[0]] = [float(x) for x in data[1:]]

        return dataDict

    @staticmethod
    def viewSensitivityAnalysis(dataDict):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D

        # data = [np.array(x) for x in dataDict.values()]
        keys = sorted(dataDict.keys())
        vals = [dataDict[key] for key in keys]
        # >>> print(keys)
        # ['AisleOne', 'AisleTwo', 'BagTime', 'BoardingTime']
        arr = np.array(vals)
        inputs = keys.copy()
        inputs.remove('BoardingTime')
        from itertools import combinations
        for var1, var2 in combinations(inputs, 2):
            # need to only look at constant values of all other variables.
            # use the median as a good one to slice through
            otherVars = [var for var in inputs if var not in (var1, var2)]
            medDict = {var:median(dataDict[var]) for var in otherVars}
            # print(medDict)
            usedVals = arr.copy()
            # print(usedVals)
            # slice along unused variables
            for var, medianVal in medDict.items():
                # which row is this variable stored in our array
                irow = keys.index(var)
                row = usedVals[irow]
                # keep the entries equal to median of that unused variable
                goodEntryIndices = np.where(row==medianVal)
                usedVals = usedVals[:, goodEntryIndices]
                # print(goodEntryIndices)
            # print(usedVals)

            # extract used data series
            x1 = usedVals[keys.index(var1)][0]
            x2 = usedVals[keys.index(var2)][0]
            Z = usedVals[keys.index('BoardingTime')][0]

            def dev(data):
                # convert a series of data to deviation from mean, in percent of mean
                data = np.array(data)
                m = np.mean(data)
                percentDeviances = (data-m)/m
                return percentDeviances
            # percent = deviances/
            print(var1, var2)
            print(x1,x2,Z)

            # X, Y =  np.meshgrid(x1,x2)
            fig = plt.figure('{} and {}'.format(var1,var2))
            ax = fig.gca(projection='3d')
            # ax.plot_surface(x1,x2,Z, cmap=cm.coolwarm)
            ax.plot_trisurf(dev(x1),dev(x2),dev(Z), cmap=cm.coolwarm)
            ax.set_xlabel('{} (mean {} seconds)'.format(var1, avg(x1)))
            ax.set_ylabel('{} (mean {} seconds)'.format(var2, avg(x2)))
            m, s = toMinutesAndSeconds(round(avg(Z)))
            ax.set_zlabel('Average Boarding Time (mean {}:{:.0f})'.format(m, s))

        plt.show()

        #
        # X,Y = np.meshgrid(bagTimes,sitDownTimes1)
        # fig = plt.figure('bagTimes and sitDown1')
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(X,Y, results, cmap=cm.coolwarm)
        #
        # X,Y = np.meshgrid(bagTimes,sitDownTimes2)
        # fig = plt.figure('bagTimes and sitDown2')
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(X,Y, results, cmap=cm.coolwarm)
        #
        # X,Y = np.meshgrid(sitDownTimes1,sitDownTimes2)
        # fig = plt.figure('sitDown1 and sitDown2')
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(X, Y, results, cmap=cm.coolwarm)
        #
        # plt.show()

    @staticmethod
    def createPassengerTypes():
        bagTimes = np.linspace(5,20,4)
        sitDown1Times = np.linspace(10,25,4)
        sitDown2Times = np.linspace(15,30,4)

        from itertools import combinations
        passTypes = []
        combos = combinations((bagTimes, sitDown1Times, sitDown2Times))
        print('generated {} combos'.format(len(combos)))
        for bt, sd1, sd2 in combos:
            class CustomPassenger(BasePassenger):
                TIME_PER_BAG_MU    = bt
                TIME_PER_BAG_SIGMA = bt/6

                TIME_AISLE_ONE_MU = sd1
                TIME_AISLE_ONE_SIGMA = sd1/6
                TIME_AISLE_TWO_MU = sd2
                TIME_AISLE_TWO_SIGMA = sd2/6
            passTypes.append(CustomPassenger)
        return passTypes

    def test(self, params=Simulator.DEFAULT_PARAMS, minRuns=10, convergenceThreshold=.01, minConvergenceCount=5):
        assert minRuns > 1
        results = []
        convergenceCount = 0

        sim = Simulator(params)
        print("running the model with params", params)
        while True:
            results.append(sim.run())
            avg = sum(results)/len(results)
            pretty = toMinutesAndSeconds(avg)
            print('epoch {} has average {}:{}'.format(len(results), *pretty))
            if len(results) > 1:
                # did this run change the average by more than cutoffChange deviations?
                prevResults = results[:-1]
                prevAverage = sum(prevResults)/len(prevResults)
                diff = abs(avg-prevAverage)
                dev = stdDev(results)
                # print(results, diff, dev)

                # did we converge this time?
                if dev == 0.0 or diff/dev <= convergenceThreshold:
                    # we need to converge multiple times in a row to be sure
                    convergenceCount += 1
                    if convergenceCount >= minConvergenceCount and len(results) >= minRuns:
                        #convergence
                        print('CONVERGENCE')
                        break
                else:
                    convergenceCount = 0
            sim.reset()
        return results

def stdDev(nums):
    n = len(nums)
    mean = sum(nums)/n
    var_sum = sum([(val - mean)**2 for val in nums])

    #sample variance is defined as sum(x - avg) / (n - 1)
    #as n approaches inf, sample var will = pop var
    var = var_sum/(n - 1)
    std = var**.5
    return std

def runningAvgs(results):
    runningAvgs = []
    for i in range(1, len(results)):
        part = results[:i]
        avg = sum(part)/len(part)
        runningAvgs.append(avg)
    return runningAvgs

def avg(iterable):
    return sum(iterable)/len(iterable)

def median(iterable):
    s = sorted(iterable)
    index = len(iterable) // 2
    return s[index]

def toMinutesAndSeconds(seconds):
    return (int(seconds//60), seconds%60)

def sensitivityAnalysis():
    m = Model()
    data = m.getAnalysisData()
    m.viewSensitivityAnalysis(data)

def visualizeRun(boardingStrategy):
    # strategies = ['random', 'backFirst', 'frontFirst']
    p = {'planeType':Plane, 'passengerType':DefaultPassenger, 'boardingStrategy':boardingStrategy}
    sim = Simulator(params=p)
    viz = visualize.Visualizer(sim)
    res = sim.run()
    # print(res)
    print('time needed for {} strategy was {}:{}'.format(boardingStrategy, *toMinutesAndSeconds(res)))

def visualizeConvergence():
    # run model
    boardingStrategy = 'backFirst'
    p = {'planeType':Plane, 'passengerType':DefaultPassenger, 'boardingStrategy':boardingStrategy}
    m = Model()
    results = m.test(params=p, convergenceThreshold=.01)

    # display
    avgs = runningAvgs(results)
    std = stdDev(results)
    minutes, sec = toMinutesAndSeconds(avg(results))
    print('Average boarding time was {}:{}  and StdDev was {} seconds'.format(minutes, sec, std))
    import matplotlib.pyplot as plt
    plt.plot(results)
    plt.plot(avgs)
    plt.xlabel('Number of Simulation Runs')
    plt.ylabel('Boarding Time (seconds)')
    plt.legend(('individual results', 'running average'))
    plt.show()

if __name__== "__main__":
    visualizeRun('random')
    # visualizeConvergence()
    # sensitivityAnalysis()
