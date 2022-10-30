from scipy import linalg
import numpy as np


def normalized(uv):
    uv_ = np.mean(uv, axis=0)
    u_,v_ = uv_[0], uv_[1]
    u_cap, v_cap = uv[:,0] - u_, uv[:,1] - v_

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_],[0,1,-v_],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T


def FindCorrespondence(a,b,path):
    matching_list = []

    if 1<= a <=5:
        with open(path + "matching" + str(a) + ".txt") as f:
            line_no = 1
            for line in f:
                if line_no == 1:
                    line_no += 1
                    nfeatures = line[11:15]
                    nfeatures = int(nfeatures)

                else:
                    matching_list.append(line.rstrip('\n'))
    final_list = []   

    for i in range(0, len(matching_list)):
          current_row = matching_list[i]
          splitStr = current_row.split()
          current_row = []
          for j in splitStr:
              current_row.append(float(j))
          final_list.append(np.transpose(current_row))
    rgb_list = []
    image1_points = []
    image2_points = []
    
    for i in range(0, len(final_list)):
        rgb_row = []
        P_1 = []
        P_2 = []
        current_row = final_list[i]
        current_row = current_row[1:len(current_row)]
        
        res = np.where(current_row == b)
        
        P_1.append((current_row[3],current_row[4]))
        rgb_row.append(current_row[0])
        rgb_row.append(current_row[1])
        rgb_row.append(current_row[2])
        
        if (len(res[0]) != 0):
            index = res[0][0]
            P_2.append((current_row[index + 1],current_row[index + 2]))
            
        else:
            P_1.remove((current_row[3],current_row[4]))
            
        if (len(P_1) != 0):
            image1_points.append((P_1))
            image2_points.append((P_2))
            rgb_list.append(np.transpose(rgb_row))
        
    image1_points = np.array(image1_points).reshape(-1,2)
    image2_points = np.array(image2_points).reshape(-1,2)
                    
    return image1_points,image2_points,rgb_list


def compute_fundamental(x1 , x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    x1, T1 = normalized(x1)
    x2, T2 = normalized(x2)
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))

    if True:
        F = np.dot(T2.T, np.dot(F, T1))

    return F/F[2,2]















































# from scipy import linalg
# import numpy as np

# #finding mateches between two images 
# def FindCorrespondence(a,b,path):
#     matching_list = []

#     if 1<= a <=5:
#         with open(path + "matching" + str(a) + ".txt") as f:
#             line_no = 1
#             for line in f:
#                 if line_no == 1:
#                     line_no += 1
#                     nfeatures = line[11:15]
#                     nfeatures = int(nfeatures)
#                 else:
#                     matching_list.append(line.rstrip('\n'))
#     final_list = []   

#     for i in range(0, len(matching_list)):
#           current_row = matching_list[i]
#           splitStr = current_row.split()
#           current_row = []
#           for j in splitStr:
#               current_row.append(float(j))
#           final_list.append(np.transpose(current_row))
#     rgb_list = []
#     image1_points = []
#     image2_points = []
    
#     for i in range(0, len(final_list)):
#         rgb_row = []
#         P_1 = []
#         P_2 = []
#         current_row = final_list[i]
#         current_row = current_row[1:len(current_row)]
        
#         res = np.where(current_row == b)
        
#         P_1.append((current_row[3],current_row[4]))
#         rgb_row.append(current_row[0])
#         rgb_row.append(current_row[1])
#         rgb_row.append(current_row[2])
        
#         if (len(res[0]) != 0):
#             index = res[0][0]
#             P_2.append((current_row[index + 1],current_row[index + 2]))
            
#         else:
#             P_1.remove((current_row[3],current_row[4]))
            
#         if (len(P_1) != 0):
#             image1_points.append((P_1))
#             image2_points.append((P_2))
#             rgb_list.append(np.transpose(rgb_row))
        
#     image1_points = np.array(image1_points).reshape(-1,2)
#     image2_points = np.array(image2_points).reshape(-1,2)
                    
#     return image1_points,image2_points,rgb_list


# def compute_fundamental(x1 , x2):
#     n = x1.shape[1]
#     if x2.shape[1] != n:
#         raise ValueError("Number of points don't match.")
    
#     # build matrix for equations
#     A = np.zeros((n,9))
#     for i in range(n):
#         A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
#                 x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
#                 x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
#     # compute linear least square solution
#     U,S,V = linalg.svd(A)
#     F = V[-1].reshape(3,3)
        
#     # constrain F
#     # make rank 2 by zeroing out last singular value
#     U,S,V = linalg.svd(F)
#     S[2] = 0
#     F = np.dot(U,np.dot(np.diag(S),V))
    
#     return F/F[2,2]

    
# # path = ("./Data/P3Data/")
# # pts1,pts2,c=FindCorrespondence(1,2,path)
# # F = compute_fundamental(pts1, pts2)






