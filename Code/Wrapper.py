import cv2
import numpy as np
import matplotlib.pyplot as plt
from GetInliersRANSAC import RANSAC, DrawMatches
from ExtractCameraPose import ComputeCameraPose
from EssentialMatrixFromFundamentalMatrix import *
from LinearTriangulation import ComputeLinearTriangulation
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from EstimateFundamentalMatrix import FindCorrespondence, ComputeFundamental
from PnPRANSAC import *
from BundleAdjustment import *
from BuildVisibilityMatrix import *
from NonlinearPnP import *
import sys
import pry

K= np.array([[ 531.122155322710, 0,                407.192550839899],
             [ 0,                531.541737503901, 313.308715048366],
             [ 0,                0,                1]])

def main():
    path = ("../Data/P3Data/")
    a = 1
    b = 2

    pts1,pts2,rgb_list = FindCorrespondence(a, b, path = path) 
    F = ComputeFundamental(pts1, pts2)  

    F_best = RANSAC(pts1, pts2)  
    DrawMatches(path, a, b, pts1, pts2)

    # E_cv2, mask_E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=0.5)
    E = ComputeEssential(K, F_best)

    R_, C_ = ComputeCameraPose(E)
    # pry()

    color = ['r', 'g', 'b', 'k']

    X_ = []

    for n in range(0, 4):
        X1 = ComputeLinearTriangulation(K, np.zeros((3, 1)), np.identity(3),
                                         C_[n].T, R_[n], np.float32(pts1),
                                         np.float32(pts2))
        # plt.scatter(X1[:, 0], X1[:, 2], c='r', s=4)
        # ax = plt.gca()
        print("X1", X1)
        # ax.set_xlabel('x')
        # ax.set_ylabel('z')

        # ax.set_xlim([-0.5, 0.5])
        # ax.set_ylim([-0.1, 2])

        # plt.show()
        X_.append(X1)

    R_best, C_best, X_best = GetBestPose(R_, C_, X_)
    pry()

    # X_nlt = NonLinearTriangulation(K, np.float32(pts1), np.float32(pts2), X_best,
    #                            np.eye(3), np.zeros((3, 1)), R_best, C_best)

    # plt.scatter(X_best[:, 0], X_best[:, 2], c='r', s=4)
    # ax = plt.gca()
    # ax.set_xlabel('x')
    # ax.set_ylabel('z')

    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([-0.1, 2])

    # plt.show()
    # pry()

    # X.append(X1)

    print(X_best)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_values = []
    y_values = []
    z_values = []

    for i in X_:
        x_values.append(X_best[0])
        y_values.append(X_best[1])
        z_values.append(X_best[2])

    ax.scatter(x_values, z_values, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


if __name__ == '__main__':
    main()








# from EstimateFundamentalMatrix import *
# from EssentialMatrixFromFundamentalMatrix import *
# from ExtractCameraPose import *
# from LinearTriangulation import *
# import matplotlib.pyplot as plt

# path = ("../Data/P3Data/")
# pts1,pts2,c=FindCorrespondence(1,2,path)

# K= np.array([[ 531.122155322710, 0,                407.192550839899],
#     [ 0,                531.541737503901, 313.308715048366],
#     [ 0,                0,                1]])

# print("pts1", pts1)
# F = compute_fundamental(pts1 , pts2)
# print(F)

# E = compute_essential(K, F)
# print(E)

# R_, C_ = compute_camerapose(E)
# # color = ['r', 'g', 'b', 'k']

# # for n in range(0, 4):
# #     X1 = compute_lineartriangulation(K, np.zeros((3, 1)), np.identity(3),
# #                                 C_[n].T, R_[n], np.float32(pts1),
# #                                 np.float32(pts2))

#     plt.scatter(X1[:, 0], X1[:, 2], c='r', s=4)
#     ax = plt.gca()
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')

#     ax.set_xlim([-0.5, 0.5])
#     ax.set_ylim([-0.1, 2])

#     plt.show()

    # X.append(X1)

# # Plotting non linear triangulation output

