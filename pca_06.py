
from PIL import Image
import random
import numpy
import pdb

from PIL import Image

import array
import logging

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math


class Cluster(object):
    # Constructor for cluster object
    def __init__(self):
        self.pixels = []  # intialize pixels into a list
        self.centroid = None  # set the number of centro
        # ids to none

    def addPoint(self, pixel):  # add pixels to the pixel list
        self.pixels.append(pixel)


class fcm(object):
    # __inti__ is the constructor and self refers to the current object.
    def __init__(self, k=3, max_PCA_iterations=15, min_distance=5.0, size=300, m=1.5, epsilon=.5, max_FCM_iterations=5):
        self.k = k  # initialize k clusters

        # intialize max_iterations
        self.max_PCA_iterations = max_PCA_iterations
        self.max_FCM_iterations = max_FCM_iterations

        self.min_distance = min_distance  # intialize min_distance
        self.degree_of_membership = []
        self.s = size ** 2
        self.size = (size, size)  # intialize the size
        self.m = m
        self.epsilon = 0.01
        self.max_diff = 0.0
        self.image = 0

    # Takes in an image and performs FCM Clustering.
    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)
        # self.beta = self.calculate_beta(self.image)

        print "********************************************************************"

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        for i in range(self.s):
            self.degree_of_membership.append(numpy.random.dirichlet(numpy.ones(self.k), size=1))

        for i in range(self.s):
            num_1 = random.randint(1, 5) * 0.1
            num_2 = random.randint(1, 5) * 0.1
            num_3 = 1.0 - (num_1 + num_2)
            degreelist = [num_1, num_2, num_3]
            self.degree_of_membership[i] = degreelist

        randomPixels = random.sample(self.pixels, self.k)
        print"INTIALIZE RANDOM PIXELS AS CENTROIDS"
        print randomPixels
        #    print"================================================================================"
        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]
            # if(i ==0):
        for cluster in self.clusters:
            for pixel in self.pixels:
                cluster.addPoint(pixel)

        print "________", self.clusters[0].pixels[0]
        iterations = 0

        # FCM
        while self.shouldExitFCM(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print "HELLO I A AM ITERATIONS:", iterations
            print"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            self.calculate_centre_vector()

            self.update_degree_of_membershipFCM()

            iterations += 1

        iterations = 0

        # PCA
        while self.shouldExitPCA(iterations) is False:
            self.oldClusters = [cluster.centroid for cluster in self.clusters]
            print "HELLO I A AM ITERATIONS:", iterations
            print"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            self.calculate_centre_vector()

            self.update_degree_of_membershipPCA()

            iterations += 1

        for cluster in self.clusters:
            print cluster.centroid
        return [cluster.centroid for cluster in self.clusters]


    def selectSingleSolution(self):
        self.max_PCA_iterations = 10
        self.max_FCM_iterations=5


    def getClusterCentroid(self):
        centroid = []
        for cluster in self.clusters:
            centroid.append(cluster.centroid);

        return centroid

    def printClustorCentroid(self):
        for cluster in self.clusters:
            print cluster.centroid

    def shouldExitFCM(self, iterations):
        if iterations <= self.max_FCM_iterations:
            return False
        return True

    def shouldExitPCA(self, iterations):

        if iterations <= self.max_PCA_iterations:
            return False
        return True

    # Euclidean distance (Distance Metric).
    def calcDistance(self, a, b):
        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    # Calculates the centroids using degree of membership and fuzziness.
    def calculate_centre_vector(self):
        for cluster in range(self.k):
            sum_numerator = 0.0
            sum_denominator = 0.0
            for i in range(self.s):
                pow_uij= pow(self.degree_of_membership[i][cluster], self.m)
                sum_denominator +=pow_uij
                num= pow_uij * self.pixels[i]

                sum_numerator+=num

            updatedcluster_center = sum_numerator/sum_denominator

            self.clusters[cluster].centroid = updatedcluster_center


    # Updates the degree of membership for all of the data points.
    def update_degree_of_membershipFCM(self):
        self.max_diff = 0.0

        for idx in range(self.k):
            for i in range(self.s):
                new_uij = self.get_new_value(self.pixels[i], self.clusters[idx].centroid)
                if (i == 0):
                    print "This is the Updatedegree centroid number:", idx, self.clusters[idx].centroid
                diff = new_uij - self.degree_of_membership[i][idx]
                if (diff > self.max_diff):
                    self.max_diff = diff
                self.degree_of_membership[i][idx] = new_uij
        return self.max_diff

    def get_new_value(self, i, j):
        sum = 0.0
        val = 0.0
        p = (2 * (1.0) / (self.m - 1))  # cast to float value or else will round to nearst int
        for k in self.clusters:
            num = self.calcDistance(i, j)
            denom = self.calcDistance(i, k.centroid)
            val = num / denom
            val = pow(val, p)
            sum += val
        return (1.0 / sum)

    def getBeta(self):
        sum_pixel = 0.0
        

        for i in range(self.s):
            sum_pixel+= self.pixels[i]

        mean = sum_pixel/self.s

        sum_distance = 0.0
        for i in range(self.s):
            sum_distance+= pow(self.calcDistance(self.pixels[i], mean),2.0)

        beta = sum_distance/self.s
            

        return beta

    # update the degree of membership for PCA
    def update_degree_of_membershipPCA(self):

        beta = self.getBeta()
       
        for idx in range(self.k):
            
            for i in range(self.s):
                if (i == 0):
                    print "This is the Update degree centroid number:", idx, self.clusters[idx].centroid

                dis = pow(self.calcDistance(self.clusters[idx].centroid, self.pixels[i]), 2.0)

                factor = dis * self.m * math.sqrt(self.k)
                factor = factor/beta

                factor = factor * -1.0

                updated_membership_degree = math.exp(factor)

                diff = updated_membership_degree - self.degree_of_membership[i][idx]
                if (diff > self.max_diff):
                    self.max_diff = diff
                    print "max_diff ------------> ", self.max_diff

                self.degree_of_membership[i][idx] = updated_membership_degree

    def I_index(self):


        result = 0;
        Ek = 0.0
        E1 = 0.0
        for i in range(self.s):
            for j in range(self.k):
                # print "membership ",self.degree_of_membership[i][0][j]
                x = self.degree_of_membership[i][j]
                y = self.calcDistance(self.pixels[i], self.clusters[j].centroid)
                mul = x * y
                Ek += mul

                if j == 0:
                    E1 += mul

                distance_list = []
        distance_list = []
        for x in range(self.k):
            if x + 1 < self.k:
                distance_list.append(self.calcDistance(self.clusters[x].centroid, self.clusters[x + 1].centroid))
            else:
                distance_list.append(self.calcDistance(self.clusters[x].centroid, self.clusters[0].centroid))

            # print "distance list " , distance_list

        distance_list.sort()

            # sort the list
        print "distance list ", distance_list
        print "E1 ", E1, "Ek", Ek

        result = (1.0 / self.k) * (E1 / Ek) * distance_list[self.k - 1]
        print "result of pb before power", result
        result = pow(result, 2.0)
        print "PB", result
        return result

    def normalizemembership(self):

        sumofzero = 0
        for i in range(self.s):
            # print " *************** ", i, self.degree_of_membership[i][0], self.degree_of_membership[i][1], self.degree_of_membership[i][2]

            sum = 0.00000

            for j in range(self.k):
                sum +=self.degree_of_membership[i][j]




            for j in range(self.k):
                if sum > 0.000000:
                    self.degree_of_membership[i][j] = self.degree_of_membership[i][j]/sum
                else:
                    sumofzero += 1

        print "sum of zero pixels ", sumofzero


    def JmFunction(self):
        # PCA 96

        sum = 0


        for j in range(self.k):

            sum1 = 0;
            sum2 = 0;

            for i in range(self.s):
                # print "membership ",self.degree_of_membership[i][0][j]
                x = pow(self.degree_of_membership[i][j], self.m)
                y = pow(self.calcDistance(self.pixels[i], self.clusters[j].centroid), 2.0)
                mul = x * y
                sum1 += mul

            eta = self.getEta(j)

            for i in range(self.s):
                x = self.degree_of_membership[i][j]
                y = (x * math.log(x)) - x
                sum2 += y

            sum2 = (sum2 * eta)

            sum += (sum1 + sum2)

        print "JM1 ", sum
        return sum

    def XBindex(self):
        sum = 0;
        distance_list = []
        for x in range(self.k):
            if x + 1 < self.k:
                distance_list.append(self.calcDistance(self.clusters[x].centroid, self.clusters[x + 1].centroid))
            else:
                distance_list.append(self.calcDistance(self.clusters[x].centroid, self.clusters[0].centroid))

        # print "distance list " , distance_list

        distance_list.sort()
        # sort the list
        # print "distance list ", distance_list

        for i in range(self.s):
            for j in range(self.k):
                x = pow(self.degree_of_membership[i][j], 2.0)
                y = pow(self.calcDistance(self.pixels[i], self.clusters[j].centroid), 2.0)

                mul = x * y
                sum += mul

        result = sum / (self.s * distance_list[0])

        print "XB ", result
        return result


    # Shows the image.
    def showImage(self):
        self.image.show()

    def showClustering(self):
        localPixels = [None] * len(self.image.getdata())
        for idx, pixel in enumerate(self.pixels):
            shortest = float('Inf')
            for cluster in self.clusters:
                distance = self.calcDistance(cluster.centroid, pixel)
                if distance < shortest:
                    shortest = distance
                    nearest = cluster

            if nearest == self.clusters[0]:
                localPixels[idx]=[0,0,0]
            elif nearest == self.clusters[1]:
                localPixels[idx] = [0,125,125]
            elif nearest == self.clusters[2]:
                localPixels[idx] = [0,0,255]
            # localPixels[idx] = nearest.centroid

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels) \
            .astype('uint8') \
            .reshape((h, w, 3))
        colourMap = Image.fromarray(localPixels)
        # colourMap.show()

        plt.imsave("PCA.png", colourMap)


if __name__ == "__main__":
    image = Image.open("Lenna.png")
    f = fcm()
    result = f.run(image)

    f.showClustering()
    f.normalizemembership()
    print f.I_index()
    # print f.JmFunction()
    print f.XBindex()
