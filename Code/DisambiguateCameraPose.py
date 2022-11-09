import numpy as np

def GetBestPose(R_, C_, X_):

#     i_best = 0
#     max_pos_depth = 0
#     for i in range(4):
#         X = X_[i] zs0p44444443e56
#         X = X / X[:, 3].reshape(-1, 1)
#         X = X[:, :3]
#         pos_depth = 0
#         for x in X:
#             x = x.reshape(-1, 1)
#             if np.dot(R_[i][2, :], x - C_[i]) > 0 and x[2] > 0:
#                 pos_depth += 1
#         if pos_depth > max_pos_depth:
#             i_best = i
#             max_pos_depth = pos_depth   
    
#     R_best, C_best, X_best = R_[i_best], C_[i_best], X_[i_best]
    
#     return R_best, C_best, X_best



# def DisambiguatePose(r_set, c_set, x3D_set):
    best_i = 0
    max_positive_depths = 0
    
    for i in range(len(R_)):
        R, C = R_[i],  C_[i].reshape(-1,1) 
        r3 = R[2, :].reshape(1,-1)
        x3D = X_[i]
        x3D = x3D / x3D[:,3].reshape(-1,1)
        x3D = x3D[:, 0:3]
        n_positive_depths = DepthPositivityConstraint(x3D, r3,C)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths
#         print(n_positive_depths, i, best_i)

    R, C, x3D = R_[best_i], C_[best_i], X_[best_i]

    return R, C, x3D 

def DepthPositivityConstraint(x3D, r3, C):
    # r3(X-C) alone doesnt solve the check positivity. z = X[2] must also be +ve 
    n_positive_depths=  0
    for X in x3D:
        X = X.reshape(-1,1) 
        if r3.dot(X-C)>0 and X[2]>0: 
            n_positive_depths+=1
    return n_positive_depths

    # best = 0
    # for i in range(4):

    #     #         Cset[i] = np.reshape(Cset[i],(-1,-1))
    #     N = X_[i].shape[0]
    #     n = 0
    #     for j in range(N):
    #         if ((np.dot(R_[i][2, :], (X_[i][j, :] - C_[i])) > 0)
    #                 and X_[i][j, 2] >= 0):
    #             n = n + 1
    #     if n > best:
    #         C = C_[i]
    #         R = R_[i]
    #         X = X_[i]
    #         best = n

    # return X, R, C