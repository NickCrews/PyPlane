#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:37:27 2017

@author: MacobJiller
"""
#import libs
import random
import pandas as pd
'''
Airbus320NEO cabin length = 90ft
numSeats = 150
90/25 = 3.6 ft/row

one person: mu = 14s, sigma = 1.96
two ppl: mu = 17s, sigma = 2.62

person velocity = 4.6 ft/s OR 1.28 row/s

bins = 49 bins / 90 ft

1.8 bins per ft, 2 bins per row


'''
# import numpy as np
random.seed(42)
class Plane(object):

    def __init__(self, params, numRows=25, seatsPerRow=6):
        self.numRows = numRows
        self.seatsPerRow = seatsPerRow
        self.seats = [[None]*(self.seatsPerRow+1) for i in range(self.numRows)]
        self.overhead = [[False]*self.numRows, [False]*self.numRows]

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

    def inAisle(self, person):
        seatNum = self.letter2number(person.seat[1])
        if seatNum < 3:
            numPpl = 0
            for spot in self.seats[person.row][:seatNum]:
                if spot is not None:
                    numPpl+=1
            return numPpl
        else:
            numPpl = 0
            for spot in self.seats[person.row][seatNum:]:
                if spot is not None:
                    numPpl+=1
            return numPpl


    def addPerson(self, person):
        firstRow = self.seats[0]
        firstRow[self.seatsPerRow//2] = person

    def findPerson(self, person):
        for row in self.seats:
            if person in row:
                return (self.seats.index(row), row.index(person))

    def isEmptyInFront(self, rowNumber):
        if rowNumber+1 < self.numRows:
            row = self.seats[rowNumber+1]
            return row[self.seatsPerRow//2] == None
        else:
            return True

    def moveForward(self, person):
        row = person.row
        self.seats[row][self.seatsPerRow//2] = None
        if row+1 < self.numRows:
            self.seats[row+1][self.seatsPerRow//2] = person

    def allNonseated(self):
        aisleSpots = [row[self.seatsPerRow//2] for row in self.seats]
        return reversed([p for p in aisleSpots if p is not None])

    def sitDown(self, person):
        rowNumber, letter = person.seat
        row = self.seats[rowNumber]
        row[self.seatsPerRow//2] = None
        seatNumber = self.letter2number(letter)
        row[seatNumber] = person

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

class Person(object):
    #need better numbers here
    TIME_PER_BAG_MU    = 8
    TIME_PER_BAG_SIGMA = 2

    TIME_AISLE_ONE_MU = 14
    TIME_AISLE_ONE_SIGMA = 1.96
    TIME_AISLE_TWO_MU = 17
    TIME_AISLE_TWO_SIGMA = 2.62

    def __init__(self, plane, seat, params):
        self.plane = plane
        self.speed = 1.0
        self.seat = seat

        self.row = 0
        self.col = self.plane.seatsPerRow//2
        self.isSeated = False
        self.onPlane = False
        self.numBags = 2
        self.satisfaction = 100

        self.TIME_PER_BAG = random.normalvariate(self.TIME_PER_BAG_MU, self.TIME_PER_BAG_SIGMA)
        self.bagTimer = -1
        self.is_bag_up = False

        self.TIME_PER_AISLE_ONE = random.normalvariate(self.TIME_AISLE_ONE_MU,self.TIME_AISLE_ONE_SIGMA)
        self.TIME_PER_AISLE_TWO = random.normalvariate(self.TIME_AISLE_TWO_MU,self.TIME_AISLE_TWO_SIGMA)
        self.aisleTimer = None


    def __repr__(self):
        return "Person: Seat {} at {}".format(self.seat, self.row)

    def atRow(self):
        return self.seat[0] == self.row

    def act(self):
        if self.isSeated:
            return False
        if self.atRow():
            if self.is_bag_up:
                if self.aisleTimer == None:
                    self.aisleTimer = (self.TIME_PER_AISLE_ONE if self.plane.inAisle(self) == 1 else self.TIME_PER_AISLE_TWO if self.plane.inAisle(self) == 2 else 0)
#                    print(self.aisleTimer)
                elif self.aisleTimer <= 0:
                    self.sitDown()
                    return True
                else:
                    self.aisleTimer-=1
            if not self.is_bag_up:
#                print(self, 'at row!')
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

class Model(object):

    DEFAULT_PARAMS = None,None
    listData = []
    listModel = []
    index, lastIndex = 0, 0

    def test(self, params=DEFAULT_PARAMS, minRuns=10, cutoffChange=1):
        assert minRuns > 1
        results = []
        lastAverage = None
        dev = 0
        modelCount = 0

        while True:
            print("running the sim with params", params)
            sim = Simulator(*params, modelNum = modelCount)
            # order = Simulator(*params, modelNum = modelCount).startingOrder()
            results.append(sim.run())
            avg = sum(results)/len(results)
            print("average this time was", avg)
            self.index += 1
            if len(results) > minRuns:

                diff = abs(avg-lastAverage)
                dev = self.stdDev(results)
                print("dev and diff was", dev, diff)

                if abs(dev - lastDev) < cutoffChange:
                    print('Reached set convergence with', len(results), 'trials.')
                    # print(order)

                    self.genData(results, modelCount)
                    modelCount +=1

                    #convergence
                    if modelCount == 3:
                        break
            lastAverage = avg
            lastDev = dev

        print('Last StdDev was', lastDev)
        print(modelCount)
        avg = sum(results)/len(results)
        return avg, dev, self.listData, self.listModel, results, order

    def stdDev(self,nums):
        n = len(nums)
        mean = sum(nums)/n
        var_sum = sum([(val - mean)**2 for val in nums])

        #sample variance is defined as sum(x - avg) / (n - 1)
        #as n approaches inf, sample var will = pop var
        var = var_sum/(n - 1)
        std = var**.5
        return std

    def genData(self, modelResults, modelNum):
        self.listData.append(modelResults[self.lastIndex:self.index])
        self.listModel.append([modelNum] * (len(self.listData[modelNum])))
        self.lastIndex = self.index
        return self.listData,self.listModel



class Simulator(object):

    def __init__(self, planeParams, personParams, boardingStrategy):
        self.plane = Plane(planeParams)
        seats = self.plane.generateAllSeats()
        self.people = [Person(self.plane, seat, personParams) for seat in seats]

        self.queue = [p for p in self.people]
#        random.shuffle(self.queue)
        self.ticks = 0
        self.count = modelNum

        self.order()

#        sorted(self.queue, key = lambda p : p.seat[0], reverse=True)

    def reset(self):
        self.queue = [p for p in self.people]
#        random.shuffle(self.queue)
        self.ticks = 0
        self.order()


    def run(self):
        while not self.plane.isFull():
            self.ticks+=1
            self.update()

        print("done.")
        print(self.plane)
        print(self.ticks)
        self.count += 1
        return self.ticks

fsdf
    def update(self):
        for p in self.plane.allNonseated():
            p.act()

        if self.plane.isBoardingSpotEmpty():
            if len(self.queue) > 0:
                p = self.queue.pop()
                p.onPlane = True
                self.plane.addPerson(p)


    #defining various boarding procedures
    def order(self):
        if self.count == 0:
            random.shuffle(self.queue)
        if self.count == 1:
            self.queue = sorted(self.queue, key = lambda person : person.seat[0], reverse=True)
        if self.count == 2:
            self.queue = sorted(self.queue, key = lambda person : person.seat[0])

        return self.queue


#run simulation
if __name__== "__main__":
    print('Running')
    m = Model()
    res = m.test()
    print(('Average was %s seconds and StdDev was %s seconds') % (res[0],res[1]))

raw_seconds, raw_model = [],[]
for w, z in zip(res[2],res[3]):
    for x,y in zip(w,z):
        raw_seconds.append(x)
        raw_model.append(y)

data = pd.DataFrame({'Model':raw_model,'Seconds':raw_seconds})
data = data.join(pd.get_dummies(data['Model'],prefix='Model'))
