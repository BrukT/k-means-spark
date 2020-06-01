from pyspark import SparkContext
sc = SparkContext(appName = "k-means", master='local[1]')

import matplotlib.pyplot as plt
import math
import numpy as np
mean_number = 4
err_distance = np.inf


def cast_list(x):
    return np.array(x, dtype=float)

def closest_mean(point, means, repetition):
    j = 0
    shortest_distance = np.linalg.norm(np.subtract(means[j], point))
    nearest_index = j
    j = j + 1
    while j < repetition:
        distance = np.linalg.norm(np.subtract(means[j], point))
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
        temp = points.keyBy(lambda x: closest_mean(x, interm_means.value, mean_number)).sortByKey()
        #print("temp keys", temp.keys().distinct().collect())
        x = temp.keys().distinct().count()
        if( x < mean_number):
            print("   error dumping      ")
            print(temp.collect())
            exit(0)
        temp2 = temp.reduceByKey(lambda x, y: np.average(np.array([x, y]), axis=0)).cache()
        print("temp2 size ", temp2.count())
        new_means = temp2.values().collect()

        err_distance = points.map(lambda x: shortest_distance(x, interm_means.value)).reduce(lambda x, y: x + y)

        print(" means num ", len(interm_means.value), " iteration ", i)
        i = i + 1
        if i > 0:
            interm_means = sc.broadcast(new_means)
        if (i > 1 and math.fabs(prev_errDist - err_distance) < 0.001 * prev_errDist):
            #plt.plot(new_means, color='red')
            break

print("Final Means")
for i in interm_means.value:
    print(i)