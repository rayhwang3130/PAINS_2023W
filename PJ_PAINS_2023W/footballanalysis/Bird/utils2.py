import numpy as np


def euclidean(x, y):

    x = np.array(x)
    y = np.array(y)

    dist = np.sqrt(np.sum(np.square(x - y), 1))

    return dist


def near_player(dist, dist_param):

    count = sum(dist <= dist_param)

    return count


def angle_dist(ball, goal_line):

    ball, goal_line = np.array(ball), np.array(goal_line)
    x = ball[0] - goal_line[0]
    y = np.abs(ball[1] - goal_line[1])

    dist = np.sqrt(x**2 + y**2)
    angle = np.rad2deg(np.arcsin(x / dist))

    return dist, angle
