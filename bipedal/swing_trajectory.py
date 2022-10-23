import numpy as np
import scipy.linalg as la


def swingTrajectoryGenerator(startPointVector, endPointVector, startVelocityVector_xy, endVelocityVector, zheight,
                             startTime, endTime, dt):
    x = trajectoryGenerator_xy(startPointVector[0], endPointVector[0], startVelocityVector_xy[0], endVelocityVector[0],
                               startTime, endTime, dt)
    y = trajectoryGenerator_xy(startPointVector[1], endPointVector[1], startVelocityVector_xy[1], endVelocityVector[1],
                               startTime, endTime, dt)
    z = trajectoryGenerator_z(zheight, startPointVector[2], endPointVector[2], endVelocityVector[2], startTime, endTime,
                              dt)

    return np.vstack((x, y, z)).T


def trajectoryGenerator_xy(startPoint, endPoint, startVelocity, endVelocity, startTime, endTime, dt):
    # 三次多项式
    # to time
    time_v = np.arange(startTime, endTime, dt)

    A = np.array([[startPoint], [endPoint], [startVelocity], [endVelocity]])

    B = np.array([[startTime ** 3, startTime ** 2, startTime, 1],
                  [endTime ** 3, endTime ** 2, endTime, 1],
                  [3 * (startTime ** 2), 2 * startTime, 1, 0],
                  [3 * (endTime ** 2), 2 * endTime, 1, 0]])

    C = np.dot(la.inv(B), A)

    x = (time_v ** 3) * C[0] + (time_v ** 2) * C[1] + time_v * C[2] + C[3]
    v = (time_v ** 2) * C[0] * 3 + time_v * C[1] * 2 + C[2]
    a = time_v * C[0] * 6 + 2 * C[2]

    return x


def trajectoryGenerator_z(zheight, startPoint, endPoint, endVelocity, startTime, endTime, dt):
    heightTime = ((endTime - startTime) / 2) + startTime
    zh = startPoint + zheight
    # print(zh)
    # print(startPoint)
    # print(endPoint)
    time_v = np.arange(startTime, endTime, dt)

    A = np.array([[zh], [endPoint], [startPoint], [endVelocity]])

    B = np.array([[heightTime ** 3, heightTime ** 2, heightTime, 1],
                  [endTime ** 3, endTime ** 2, endTime, 1],
                  [startTime ** 3, startTime ** 2, startTime, 1],
                  [3 * (endTime ** 2), 2 * endTime, 1, 0]])

    C = np.dot(la.inv(B), A)

    z = (time_v ** 3) * C[0] + (time_v ** 2) * C[1] + time_v * C[2] + C[3]  # z position
    v = (time_v ** 2) * C[0] * 3 + time_v * C[1] * 2 + C[2]  # velocity
    a = time_v * C[0] * 6 + 2 * C[2]

    return z
