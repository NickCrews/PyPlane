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
        if seatNum < 3:
            numPpl = 0
            for spot in self.seats[passenger.row][:seatNum]:
                if spot is not None:
                    numPpl+=1
            return numPpl
        else:
            numPpl = 0
            for spot in self.seats[passenger.row][seatNum:]:
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
                p = self.queue.pop()
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

    SENSITIVITY_FILENAME = 'sensitivity.pkl'

    def runSensitivityAnalysis(self):
        passTypes = self.createPassengerTypes()
        inputs = [{'planeType':Plane, 'passengerType':pt, 'boardingStrategy':'random'} for pt in passTypes]
        resultDict = {}
        for inp in inputs:
            result = self.test(params=inp)
            # make our input dict hashable
            key = tuple(sorted(inp.items()))
            resultDict[key] = result
        return resultDict

    def getAnalysisData(self):
        try:
            return openSensitivityAnalysis()
        except:
            data = self.runSensitivityAnalysis()
            lists = self.convertToLists(data)
            self.saveSensitivityAnalysis(lists)
            return lists

    @staticmethod
    def saveSensitivityAnalysis(resultLists):
        with open(Model.SENSITIVITY_FILENAME, 'wb') as f:
            import pickle
            pickle.dump(resultLists, f)

    @staticmethod
    def openSensitivityAnalysis():
        with open(Model.SENSITIVITY_FILENAME, 'rb') as f:
            import pickle
            return pickle.load(f)

    @staticmethod
    def convertToLists(resultDict):
        results = bagTimes = sitDownTimes1 = sitDownTimes2 = []
        for inputs, result in resultDict.items():
            results.append(result)
            for key, val in inputs:
                if key == 'passengerType':
                    pt = val
                    break
            bagTimes.append(pt.TIME_PER_BAG_MU)
            sitDownTimes1.append(pt.TIME_AISLE_ONE_MU)
            sitDownTimes2.append(pt.TIME_AISLE_TWO_MU)
        return bagTimes, sitDownTimes1, sitDownTimes2, results

    @staticmethod
    def viewSensitivityAnalysis(resultLists):
        results = bagTimes = sitDownTimes1 = sitDownTimes2 = resultLists

        import matplotlib.pyplot as plt
        fig = plt.figure('bagTimes and sitDown1')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(bagTimes, sitDownTimes1, results)

        fig = plt.figure('bagTimes and sitDown2')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(bagTimes, sitDownTimes2, results)

        fig = plt.figure('sitDown1 and sitDown2')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(sitDownTimes1, sitDownTimes2, results)

        plt.show()

    def sensitivityAnalysis(self):
        data = self.getAnalysisData()
        self.viewSensitivityAnalysis(data)

    @staticmethod
    def createPassengerTypes():
        bagTimes = np.linspace(5,20,5)
        sitDown1Times = np.linspace(10,25,5)
        sitDown2Times = np.linspace(15,30,5)

        from sklearn.utils.extmath import cartesian
        passTypes = []
        for bt, sd1, sd2 in cartesian((bagTimes, sitDown1Times, sitDown2Times)):
            class CustomPassenger(BasePassenger):
                TIME_PER_BAG_MU    = bt
                TIME_PER_BAG_SIGMA = bt/6

                TIME_AISLE_ONE_MU = sd1
                TIME_AISLE_ONE_SIGMA = sd1/6
                TIME_AISLE_TWO_MU = sd2
                TIME_AISLE_TWO_SIGMA = sd2/6
            passTypes.append(CustomPassenger)
        return passTypes


    def test(self, params=Simulator.DEFAULT_PARAMS, minRuns=10, convergenceThreshold=.05, minConvergenceCount=5):
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

    # def genData(self, modelResults, modelNum):
    #     self.listData.append(modelResults[self.lastIndex:self.index])
    #     self.listModel.append([modelNum] * (len(self.listData[modelNum])))
    #     self.lastIndex = self.index
    #     return self.listData,self.listModel

def runningAvgs(results):
    runningAvgs = []
    for i in range(1, len(results)):
        part = results[:i]
        avg = sum(part)/len(part)
        runningAvgs.append(avg)
    return runningAvgs

def avg(iterable):
    return sum(iterable)/len(iterable)

def toMinutesAndSeconds(seconds):
    return (int(seconds//60), seconds%60)

#run simulation
if __name__== "__main__":
    # sim = Simulator()
    # viz = visualize.Visualizer(sim)
    # import time
    # time.sleep(2)
    # sim.run()

    # print('Running')
    m = Model()
    m.sensitivityAnalysis()
    # res = m.test()
    # print(res),
    # print('time needed was {}:{}'.format(*toMinutesAndSeconds(avg(res))))
    # avgs = runningAvgs(res)
    # import matplotlib.pyplot as plt
    # plt.plot(res)
    # plt.plot(avgs)
    # plt.show()
    # print(('Average was %s seconds and StdDev was %s seconds') % (res[0],res[1]))

# raw_seconds, raw_model = [],[]
# for w, z in zip(res[2],res[3]):
#     for x,y in zip(w,z):
#         raw_seconds.append(x)
#         raw_model.append(y)

# data = pd.DataFrame({'Model':raw_model,'Seconds':raw_seconds})
# data = data.join(pd.get_dummies(data['Model'],prefix='Model'))
