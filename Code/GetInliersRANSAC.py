import numpy as np
from EstimateFundamentalMatrix import *
import cv2

def RANSAC_filter(pts1, pts2):
    # print("pts1 shape: = ", pts1.shape)
    # print("pts2 shape: = ", pts2.shape)
    num_iterations = 1000
    num_points = pts1.shape[0]
    max_count = 0
    F_best = None

    for i in range(num_iterations):
        random_index = np.random.randint(num_points, size=8)
        pts1_new = pts1[random_index, :] 
        pts2_new = pts2[random_index, :] 
        # print("pts1_new shape: = ", pts1_new.shape)
        # print("pts2_new shape: = ", pts2_new.shape)
        F = compute_fundamental(pts1_new, pts2_new)
        # print("F: = ", F)
        # inlier_indices = []
        count = 0

        for j in range(num_points):
            x1tmp=np.array([pts1[j, :][0], pts1[j, :][1], 1])
            x2tmp=np.array([pts2[j, :][0], pts2[j, :][1], 1]).T
            error = abs(np.dot(x2tmp, np.dot(F, x1tmp)))
            # print("Error: = ", error)
            if error < 0.001:
                # inlier_indices.append()
                print("Error: = ", error)
                count = count + 1

            if count > max_count:
                max_count = count
                F_best = F

    return F_best