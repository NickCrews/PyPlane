#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:37:27 2017

@author: MacobJiller
"""
#import libs
import random
import matplotlib.pyplot as plt
import pyautogui as auto
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

    def __init__(self):
        self.numRows = 25
        self.seatsPerRow = 6
        self.seats = [[None]*(self.seatsPerRow+1) for i in range(self.numRows)]
        self.overhead = [[False]*self.numRows, [False]*self.numRows]

    def generateAllSeats(self):
        '''generate all the seat tuples (0,'a'), (0, 'b'), ...(self.numSeats, 'f')'''
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
        spr = Plane().seatsPerRow
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
    
    def __init__(self, plane, seat):
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

class Controller(object):
    def __init__(self):
        self.plane = Plane()
        seats = self.plane.generateAllSeats()
        self.people = [Person(self.plane, seat) for seat in seats]

        self.queue = [p for p in self.people]
        random.shuffle(self.queue)
        self.ticks = 0

        self.DO_VIZ = True
        if self.DO_VIZ:
            self.viz = Visualizer(self)
            
        self.count = 0

    def run(self):
        while not self.plane.isFull():
            self.ticks+=1
            changed = self.update()
            if self.DO_VIZ and changed:
                self.viz.show()
                #to automate our model
                auto.click()
        print(self.plane)
        print(self.ticks)


    def update(self):
#        self.ticks+=1
        vizChanged = False
        s = [x for x in self.viz.ax.patches if x.xy[1] == -1.0 or x.xy[1] == 6.5]
        for p in self.plane.allNonseated():
            if p.act():
                vizChanged = True
                if p.isSeated:
                    #put in this try/except block solely for the color change to work
                    try:
                        s[self.count].set_facecolor('green')
                    except:
                        pass
                    for patch in self.viz.ax.patches:
                        if patch.xy == (p.row - 0.5, p.col - 0.5):
                            patch.set_facecolor('green')
                            self.count+=1
                    
        if self.plane.isBoardingSpotEmpty():
            if len(self.queue) > 0:
                p = self.queue.pop()
                p.onPlane = True
                self.plane.addPerson(p)
        return vizChanged

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


#run simulation
if __name__== "__main__":
    print('Running')
    c = Controller()
    c.run()
    print('Total boarding time:', c.ticks/60, 'minutes.')

