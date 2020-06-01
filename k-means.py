from pyspark import SparkContext, Broadcast

import sys
import matplotlib.pyplot as plt
import math
import numpy as np

sc = SparkContext('local', 'k-means-app')
sc.setLogLevel('WARN')


def closest_mean(point, means, repetition):
    j = 0
    shortest_dist: float = np.linalg.norm(np.subtract(means[j], point))
    nearest_index = j
    j = j + 1
    while j < repetition:
        distance = np.linalg.norm(np.subtract(means[j], np.array(point)))
        if shortest_dist > distance:
            nearest_index = j
            shortest_dist = distance
        j = j + 1
    return nearest_index


def shortest_distance(point, means, repetition):
    j = 0
    shortest_dist = np.linalg.norm(np.subtract(means[j], point))
    j = j + 1
    while j < repetition:
        distance = np.linalg.norm(np.subtract(means[j], point))
        if shortest_dist > distance:
            shortest_dist = distance
        j += 1
    return shortest_dist


def plot_list(points, col='blue'):
    # plotting list of points
    x, y = [], []
    for pt in points:
        x.append(pt[0])
        y.append(pt[1])
    plt.scatter(x, y, color=col)


def clustering_plot(points, means):
    for pt in points:
        if closest_mean(pt, means, len(means)) == 0:
            plt.scatter(pt[0], pt[1], color='red')
        elif closest_mean(pt, means, len(means)) == 1:
            plt.scatter(pt[0], pt[1], color='blue')
        elif closest_mean(pt, means, len(means)) == 2:
            plt.scatter(pt[0], pt[1], color='green')
        elif closest_mean(pt, means, len(means)) == 3:
            plt.scatter(pt[0],pt[1], color='yellow')

    plt.show()


def main():
    if len(sys.argv) == 1:
        input_f = "./points.txt"
        mean_number = 4
    elif len(sys.argv) == 3:
        input_f = sys.argv[1]
        mean_number = sys.argv[2]
    else:
        print("usage: python </path/to/inputfile.txt> <number_of_means> \n or no arguments")
        exit(0)

    err_distance = float('inf')
    stop_err_level = 0.0000001
    iteration_max = 20

    pointstxt = sc.textFile(input_f)
    points = pointstxt.map(lambda x: x.split(",")).map(lambda x: np.array(x, dtype=float))
    starting_means = points.takeSample(num=mean_number, withReplacement=False)
    print("starting means size ", len(starting_means))

    iteration = 0
    errs = []
    interm_means = sc.broadcast(starting_means)
    mean_count = sc.broadcast(mean_number)
    print("starting means ", interm_means.value)

    # plot points and initial means with black
    plot_list(points.collect())
    plot_list(starting_means, col='black')

    while iteration < iteration_max:
        prev_errdist = err_distance
        new_means = points.map(lambda x: (closest_mean(x, interm_means.value, mean_count.value), x))\
            .reduceByKey(lambda x, y: np.average(np.array([x, y]), axis=0)).values().collect()

        iteration += 1
        if iteration > 0:
            interm_means = sc.broadcast(new_means)

        err_distance = points.map(lambda x: shortest_distance(x, interm_means.value, mean_count.value)).sum()
        errs.append(err_distance)

        print(" means num ", len(interm_means.value), " iteration ", iteration, " error ", err_distance)

        if (iteration > 1) and (math.fabs(prev_errdist - err_distance) < stop_err_level):
            break

    print("Final Means")
    for mean in interm_means.value:
        print(mean)

    plot_list(interm_means.value, col='red')
    plt.show()

    '''plotting the line graph of errors'''
    plt.plot(errs)
    plt.show()

    '''plotting the scatter plot of the cluster'''
    clustering_plot(points.collect(), interm_means.value)
    sc.cancelAllJobs()
    sc.stop()


if __name__ == '__main__':
    main()
