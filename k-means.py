from pyspark import SparkContext
sc = SparkContext(appName = "k-means", master='local[*]')

import matplotlib.pyplot as plt
import math
import numpy as np
mean_number = 4
err_distance = np.inf


def cast_list(x):
    return np.array(x, dtype=float)

def closest_mean(point, means):
    j = 0
    for mean in means:
        if j == 0:
            shortest_distance = np.linalg.norm(np.subtract(mean,point))
            nearest_index = j
            j = j + 1
        else:
            distance = np.linalg.norm(np.subtract(mean,point))
            if(shortest_distance > distance):
                nearest_index = j
            j = j + 1
    return nearest_index



def shortest_distance(point, means):
    j = 0
    for mean in means:
        if j == 0:
            shortest_distance = np.linalg.norm(np.subtract(mean, point))
            j = j + 1
        else:
            distance = np.linalg.norm(np.subtract(mean, point))
            if (shortest_distance > distance):
                short_distance = distance
            j = j + 1
    return shortest_distance

if __name__ == "__main__":
    pointstxt = sc.textFile("/home/bruk/projects/cloud/k-means/generate_point/points.txt")
    points = pointstxt.map(lambda x: x.split(",")).map(lambda x: cast_list(x)).persist()
    starting_means = points.takeSample(num=mean_number, withReplacement=False)

    i = 0
    interm_means = sc.broadcast(starting_means)
    while True:
        prev_errDist = err_distance
        new_means = points.keyBy(lambda x: closest_mean(x, interm_means.value)) \
                                 .reduceByKey(lambda x, y: np.average(np.array([x, y]), axis=0)).values().collect()

        err_distance = points.map(lambda x: shortest_distance(x, interm_means.value)).reduce(lambda x, y: x + y)

        i = i + 1
        print(" means num ", len(interm_means.value), " iteration ", i)
        if i > 0:
            interm_means = sc.broadcast(new_means)
        if (i > 1 and math.fabs(prev_errDist - err_distance) < 0.01 * prev_errDist):
            #plt.plot(new_means, color='red')
            for i in new_means:
                print(i)
            break

    plt.show()