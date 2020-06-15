"""
when running in local mode the algorithm displays the means in the progress and the errors.
Also it plots the resulting clustered points graph
"""

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
    distance = np.sum((np.asarray(means) - np.array(point)) ** 2, axis=1)
    return np.argmin(distance)


def shortest_distance(point, means):
    distance = np.sum((np.asarray(means) - np.array(point)) ** 2, axis=1)
    return np.amin(distance)


def main():
    if len(sys.argv) == 1:
        inpute_file = "./points.txt"
        cluster_number = 4
        dimension = 2
    elif len(sys.argv) == 4:
        inpute_file = sys.argv[1]
        cluster_number = int(sys.argv[2])
        dimension = int(sys.argv[3])
    else:
        print("usage: python </path/to/inputfile.txt> <number_of_means> <dimension_of_points>\n or no arguments")
        exit(0)

    if os.path.exists("./output/"):
        shutil.rmtree("./output/")

    error_distance = float('inf')
    stop_err_level = 0.0000001
    iteration_max = 20
    iteration_min = 5

    pointstxt = sc.textFile(inpute_file)
    points = pointstxt.map(lambda x: x.split(",")).map(lambda x: np.array(x, dtype=float))
    starting_means = points.takeSample(num=cluster_number, withReplacement=False)
    if sc.master == 'local':
        print("starting means size ", len(starting_means))

    iteration = 0
    errors = []
    intermediate_means = sc.broadcast(starting_means)
    print("starting means ", intermediate_means.value)

    if dimension == 2 and sc.master == 'local':
        """ plot points and initial means with black if the dimension is 2
        for debugging purpose"""
        PLotUtil.plot_list(points.collect())
        PLotUtil.plot_list(starting_means, col='black', sz=80)

    while iteration < iteration_max:
        previous_error_distance = error_distance
        accumulator = (np.zeros(shape=(dimension,), dtype=float), 0)
        new_means = points.map(lambda x: (closest_mean(x, intermediate_means.value), x))\
            .aggregateByKey(accumulator, lambda a, b: (a[0] + b, a[1] + 1), lambda a, b: (a[0] + b[0], a[1] + b[1])) \
            .mapValues(lambda v: v[0] / v[1]).values().collect()

        error_distance = points.map(lambda x: shortest_distance(x, intermediate_means.value)).sum()

        intermediate_means = sc.broadcast(new_means)
        iteration += 1

        # display debugging information if it is in local node
        if sc.master == 'local':
            print("Means ", intermediate_means.value, " iteration ", iteration, " error ", error_distance)
            # collect the errors in a list to plot the error trend at the end
            errors.append(error_distance)

        if (iteration > iteration_min) and (math.fabs(previous_error_distance - error_distance) < (stop_err_level * previous_error_distance)):
            break

    print("Final Means")
    for mean in intermediate_means.value:
        print(mean)
        
    if dimension == 2 and sc.master == 'local':
        '''plotting the final means if the dimension is 2'''
        if len(starting_means[0]) == 2:
            PLotUtil.plot_list(intermediate_means.value, col='red', sz=80)
            PLotUtil.show()

            '''plotting the line graph of errors'''
            PLotUtil.plot(errors)

            '''plotting the scatter plot of the cluster'''
            PLotUtil.clustering_plot(points.collect(), intermediate_means.value, closest_mean)

    '''Saving the output cluster centroids'''
    sc.parallelize(intermediate_means.value).saveAsTextFile("./output/")
    sc.cancelAllJobs()
    sc.stop()


if __name__ == '__main__':
    main()
