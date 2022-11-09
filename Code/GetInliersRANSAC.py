import numpy as np
from EstimateFundamentalMatrix import *
import cv2

def RANSAC(pts1, pts2):

    num_points = pts1.shape[0]
    max_count = 0
    F_best = None

    for i in range(1000):
        random_index = np.random.randint(num_points, size=8)
        pts1_new = pts1[random_index, :] 
        pts2_new = pts2[random_index, :] 

        F = ComputeFundamental(pts1_new, pts2_new)

        count = 0

        for j in range(num_points):
            x1tmp=np.array([pts1[j, :][0], pts1[j, :][1], 1])
            x2tmp=np.array([pts2[j, :][0], pts2[j, :][1], 1]).T
            error = abs(np.dot(x2tmp, np.dot(F, x1tmp)))
            # print("Error: = ", error)
            if error < 0.01:
                count = count + 1

            if count > max_count:
                max_count = count
                F_best = F

    return F_best


def DrawMatches(path, a, b, inliers_a, inliers_b):
    img1 = cv2.imread(path + str(a) + ".png")
    img2 = cv2.imread(path + str(b) + ".png")

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1
    out[:rows2, cols1:cols1 + cols2, :] = img2
    radius = 6
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    thickness = 1

    for m in range(0, len(inliers_a)):
        # Draw small circle on image 1
        cv2.circle(out, (int(inliers_a[m][0]), int(inliers_a[m][1])),
            radius, RED, thickness)

        # Draw small circle on image 2
        cv2.circle(out, (int(inliers_b[m][0])
            + cols1, int(inliers_b[m][1])), radius, RED, thickness)

        # Draw line connecting circles
        cv2.line(out, (int(inliers_a[m][0]), int(inliers_a[m][1])),
            (int(inliers_b[m][0])+cols1, int(inliers_b[m][1])), GREEN, thickness)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000, 600)
    cv2.imshow('image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

