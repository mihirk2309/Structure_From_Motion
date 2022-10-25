from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
import matplotlib.pyplot as plt

path = ("../Data/P3Data/")
pts1,pts2,c=FindCorrespondence(1,2,path)

K= np.array([[ 531.122155322710, 0,                407.192550839899],
    [ 0,                531.541737503901, 313.308715048366],
    [ 0,                0,                1]])

print("pts1", pts1)
F = compute_fundamental(pts1 , pts2)
print(F)

E = compute_essential(K, F)
print(E)

R_, C_ = compute_camerapose(E)
color = ['r', 'g', 'b', 'k']

for n in range(0, 4):
    X1 = compute_lineartriangulation(K, np.zeros((3, 1)), np.identity(3),
                                C_[n].T, R_[n], np.float32(pts1),
                                np.float32(pts2))

    plt.scatter(X1[:, 0], X1[:, 2], c='r', s=4)
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.1, 2])

    plt.show()

    X.append(X1)

# # Plotting non linear triangulation output

