import math
import os
import shutil
import sys

import numpy as np
from pyspark import SparkContext
from resources.PlotUtil import PLotUtil

sc = SparkContext('local', 'k-means-app')
sc.setLogLevel('WARN')


def closest_mean(point, means):
    j = 0
    shortest_dist = math.pow(np.linalg.norm(np.subtract(means[j], point)), 2)
    nearest_index = j
    j = j + 1
    while j < len(means):
        distance = math.pow(np.linalg.norm(np.subtract(means[j], np.array(point))), 2)
        if shortest_dist > distance:
            nearest_index = j
            shortest_dist = distance
        j = j + 1
    #print("the nearest index", nearest_index)
    return nearest_index


def shortest_distance(point, means):
    j = 0
    shortest_dist = math.pow(np.linalg.norm(np.subtract(means[j], point)),2)
    j = j + 1
    while j < len(means):
        distance = math.pow(np.linalg.norm(np.subtract(means[j], point)), 2)
        if shortest_dist > distance:
            shortest_dist = distance
        j += 1
    return shortest_dist


def return_y(x,y):
    if x == 0:
        return y


def find_average(x, y):
    total = np.array(x) + np.array(y)
    return total / 2


def find_list_average(x):
    tot = np.array([0, 0], dtype=float)
    for i in x:
        tot = np.add(tot, np.array(i))
    return tot/len(x)


def main():
    if len(sys.argv) == 1:
        input_f = "./points.txt"
        mean_number = 4
    elif len(sys.argv) == 3:
        input_f = sys.argv[1]
        mean_number = int(sys.argv[2])
    else:
        print("usage: python </path/to/inputfile.txt> <number_of_means> \n or no arguments")
        exit(0)

    if os.path.exists("./output/"):
        shutil.rmtree("./output/")

    err_distance = float('inf')
    stop_err_level = 0.0000001
    iteration_max = 20
    iteration_min = 10

    pointstxt = sc.textFile(input_f)
    points = pointstxt.map(lambda x: x.split(",")).map(lambda x: np.array(x, dtype=float))
    starting_means = points.takeSample(num=mean_number, withReplacement=False, seed=1)
    print("starting means size ", len(starting_means))

    iteration = 0
    errs = []
    interm_means = starting_means
    mean_count = sc.broadcast(mean_number)
    print("starting means ", interm_means)

    if len(starting_means[0]) == 2:
        # plot points and initial means with black if the dimension is 2
        PLotUtil.plot_list(points.collect())
        PLotUtil.plot_list(starting_means, col='black', sz=80)

    while iteration < iteration_max:
        prev_errdist = err_distance
        temp = points.map(lambda x: (closest_mean(x, interm_means), x))
        aTuple = (0, 0)
        temp2 = temp.aggregateByKey(aTuple, lambda a, b: (a[0] + b, a[1] + 1), lambda a, b: (a[0] + b[0], a[1] + b[1]))
        temp3 = temp2.mapValues(lambda v: v[0]/v[1]).values().collect()
        print("mean values: ", temp3)
        #print("iteration ", iteration, " keys ", temp.keys().takeSample(withReplacement=False, num=10, seed=1))
        print("key 0 values: ", temp.count())
        x = []
        for i in temp.collect():
            print(i[0], i[1][0], i[1][1])

            #if i[0] == 2:
                #x.append(i[1])

        #print("list mean of ", len(x), " is ", find_list_average(x))
        '''
        for i in new_means_2.collect():
            print("key ", i[0])
            for j in i[1]:
                print(j)
        '''
        new_means = temp3
        err_distance = points.map(lambda x: shortest_distance(x, interm_means)).sum()

        errs.append(err_distance)
        iteration += 1
        if iteration > 0:
            del interm_means
            interm_means = new_means

        del new_means
        print("Means ", interm_means, " iteration ", iteration, " error ", err_distance)
        #interm_means.destroy()
        if (iteration > iteration_min) and (math.fabs(prev_errdist - err_distance) < (stop_err_level * prev_errdist)):
            break

    print("Final Means")
    for mean in interm_means:
        print(mean)

    '''plotting the final means if the dimension is 2'''
    if len(starting_means[0]) == 2:
        PLotUtil.plot_list(interm_means, col='red', sz=80)
        PLotUtil.show()

        '''plotting the line graph of errors'''
        PLotUtil.plot(errs)

        '''plotting the scatter plot of the cluster'''
        PLotUtil.clustering_plot(points.collect(), interm_means, closest_mean)

    '''Saving the output clusters'''
    sc.parallelize(interm_means).saveAsTextFile("./output/")
    sc.cancelAllJobs()
    sc.stop()


if __name__ == '__main__':
    main()
