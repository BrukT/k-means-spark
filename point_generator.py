import numpy as np

import random
import math
from resources.PlotUtil import PLotUtil
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

class Circle:
    def __init__(self, origin, radius):
        self.origin = origin
        self.radius = radius


if __name__ == '__main__':
    num_samples = 200
    center_x, center_y = 20, 80
    origin = Point(0, 0)
    radius = 20
    circle = Circle(origin, radius)
    values = np.empty(shape=(num_samples, 2), dtype=float)
    for i in range(0, int(num_samples/2)):
        p = random.random() * 2 * math.pi
        r = circle.radius * math.sqrt(random.random())
        x, y = math.cos(p) * r + center_x, math.sin(p) * r + center_y
        values[i] = np.array([x, y])

    origin = Point(0, 0)
    circle = Circle(origin, radius)
    center_x, center_y = 80, 20

    for j in range(int(num_samples/2), num_samples):
        p = random.random() * 2 * math.pi
        r = circle.radius * math.sqrt(random.random())
        x, y = math.cos(p) * r + center_x, math.sin(p) * r + center_y
        values[j] = np.array([x, y])

    #print(values)
    PLotUtil.plot_list(values)
    PLotUtil.show()
    np.savetxt("pts_in_circles.txt", values, delimiter=",", fmt="%s")