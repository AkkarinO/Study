import csv
from matplotlib.pyplot import figure, show, title
import numpy as np
import pyransac3d as py
import random
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def points_r():
    with open('LidarData_all.xyz', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for x, y, z in reader:
            yield x, y, z


points_all = []
for line in points_r():
    points_all.append(line)


# 3D DRAWING OF POINTS CLOUD

def draw_3d(points):
    fig = figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')

    x, y, z = zip(*points)

    # Changing tuple type to the float type in order to plot 3d
    X = np.array(x, dtype=float)
    Y = np.array(y, dtype=float)
    Z = np.array(z, dtype=float)
    ax.scatter(X, Y, Z)
    title('Points cloud representation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    show()


# 3D DRAWING OF POINTS CLOUD - CLUSTERS KMEANS METHOD

def drawKMeans_3d(points):
    clusterer = KMeans(n_clusters=3)

    X_clus = np.array(points, dtype=float)
    y_pred = clusterer.fit_predict(X_clus)

    red = y_pred == 0
    blue = y_pred == 1
    cyan = y_pred == 2

    fig = figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X_clus[red, 0], X_clus[red, 1], X_clus[red, 2], c="r")
    ax.scatter(X_clus[blue, 0], X_clus[blue, 1], X_clus[blue, 2], c="b")
    ax.scatter(X_clus[cyan, 0], X_clus[cyan, 1], X_clus[cyan, 2], c="c")
    title('KMeans algorithm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    show()


drawKMeans_3d(points_all)


# RANSAC ALGORITHM BASED ON THE POINTS CLOUD

def ransac(points, threshold):
    X = np.array(points, dtype=float)
    A = random.choice(X)
    B = random.choice(X)
    C = random.choice(X)

    vec_A = np.subtract(A, C)
    vec_B = np.subtract(B, C)

    vec_Ua = vec_A / np.linalg.norm(vec_A)
    vec_Ub = vec_B / np.linalg.norm(vec_B)

    n = np.cross(vec_Ua, vec_Ub)

    d = -np.sum(np.multiply(C, n))

    distance_all_p = (n[0] * X[:, 0] + n[1] * X[:, 1] + n[2] * X[:, 2]) + d

    inliers = np.where(np.abs(distance_all_p) <= threshold)[0]

    print('Plane coefficients are: ', '\na: ', n[0], '\nb: ', n[1], '\nc: ', n[2], '\nd: ', d,
          '\n\nNumber of inliers equals to: ', inliers)

    if n[1] == 0 and n[0] == 0 and n[2] != 0:
        print('Plane is horizontal.', end='')
    elif (n[0] != 0 or n[1] != 0) and n[2] == 0:
        print('Plane is vertical.', end='')
    else:
        print('Plane is not vertical neither horizontal.')


ransac(points_all, 0.01)


# 3D DRAWING OF POINTS CLOUD - CLUSTERS DBSCAN METHOD

def drawDBSCAN_3d(points, eps, min_samples):
    X_clus = np.array(points, dtype=float)
    clusterer = DBSCAN(eps, min_samples=min_samples)
    y_pred = clusterer.fit_predict(X_clus)

    # black = y_pred == -1  # outliers - uncomment if desired on plot
    red = y_pred == 0
    blue = y_pred == 1
    cyan = y_pred == 2

    fig = figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')

    # ax.scatter(X_clus[black, 0], X_clus[black, 1], X_clus[black, 2], c="black") # outliers
    ax.scatter(X_clus[red, 0], X_clus[red, 1], X_clus[red, 2], c="r")
    ax.scatter(X_clus[blue, 0], X_clus[blue, 1], X_clus[blue, 2], c="b")
    ax.scatter(X_clus[cyan, 0], X_clus[cyan, 1], X_clus[cyan, 2], c="c")
    title('DBSCAN algorithm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    show()


drawDBSCAN_3d(points_all, 40, 500)


# RANSAC ALGORITHM BASED ON THE POINTS CLOUD - ransac3d package

def ransac3d_plane(points):
    X = np.array(points, dtype=float)
    plane = py.Plane()
    best_eq, best_inliers = plane.fit(X, 0.01)
    print('\n\nThese are plane coefficients:', '\na: ', best_eq[0], '\nb: ', best_eq[1], '\nc: ', best_eq[2],
          '\nd: ', best_eq[3], '\n')


ransac3d_plane(points_all)


# ATTENTION! Cylinder function seems to be not fully functional yet - error occurs

# def ransac3d_cylinder(points):
#     X = np.array(points)
#     cylinder = py.Cylinder
#     center, axis, radius, inliers = cylinder.fit(X, 0.01)
#     print('These are cylinder parameters:', '\nCenter: ', center, '\nAxis: ', axis,
#           '\nRadius: ', radius, '\nInliers: ', inliers, '\n')
#
#
# ransac3d_cylinder(points_all)
