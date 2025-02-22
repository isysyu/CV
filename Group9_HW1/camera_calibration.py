import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

USE_OPENCV = False # True: use cv2.calibrateCamera, False: use our own calibration function
DATA_PATH = 'my_data' # ['data', 'my_data']

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers. -> (10,7)
corner_x = 10 if DATA_PATH == 'my_data' else 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob(os.path.join(DATA_PATH, '*.jpg'))

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)

print("objpoints shape:", np.array(objpoints).shape)
print("imgpoints shape:", np.array(imgpoints).shape)

#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = img[0].shape
# You need to comment these functions and write your own calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.

cv_ret, cv_mtx, cv_dist, cv_rvecs, cv_tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
cv_rvecs = np.array(cv_rvecs)
cv_tvecs = np.array(cv_tvecs)

# 將 cv_extrinsics 轉換為 (n_images, 3, 4) 的形狀，方便後續比較
cv_extrinsics = np.zeros((len(cv_rvecs), 3, 4))
for i in range(len(cv_rvecs)):
    R, _ = cv2.Rodrigues(cv_rvecs[i])
    cv_extrinsics[i] = np.hstack((R, cv_tvecs[i].reshape(3, 1)))
print("cv_extrinsics shape:", cv_extrinsics.shape)

#########################################
##### START OF CALIBRATION FUNCTION #####
#########################################
def calculate_homography(obj_points, img_points):
    """
    計算單張圖片的 Homography Matrix

    s * [u, v, 1]^T = H * [X, Y, 1]^T

    每個 corner point 提供兩個方程式，總共有 corner_x * corner_y * 2 個方程式
    X*h11 + Y*h12 + h13 - X*u*h31 - Y*u*h32 - u*h33 = 0
    X*h21 + Y*h22 + h23 - X*v*h31 - Y*v*h32 - v*h33 = 0
    """
    H = np.zeros((corner_x * corner_y * 2, 9))
    img_points = img_points.reshape(-1, 2) # from (70, 1, 2) to (70, 2), corner_x * corner_y = 70
    for idx, (obj, img) in enumerate(zip(obj_points, img_points)):
        X_i, Y_i, _ = obj
        u_i, v_i = img
        H[idx*2, :] = [X_i, Y_i, 1, 0, 0, 0, -X_i*u_i, -Y_i*u_i, -u_i]
        H[idx*2+1, :] = [0, 0, 0, X_i, Y_i, 1, -X_i*v_i, -Y_i*v_i, -v_i]
    
    # 使用 SVD 分解求解 H
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    # H 的解是與最小奇異值相對應的右奇異向量，即 vh 的最後一行，這裡使用 argmin 確保取到最小奇異值
    # H = vh[-1]
    H = vh[np.argmin(s), :]
    # Normalize H
    H *= np.sign(H[-1]) # Ensure H[-1] is positive
    H /= np.abs(H[-1]) # Normalize H[-1] to 1
    return H.reshape(3, 3) # reshape to 3x3

def v_pq(H, p, q):
    """
        v_pq 是 H 的元素組成的向量，用於計算 B 矩陣
    """
    v_pq = np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])
    return v_pq

def calibrate_camera(objpoints, imgpoints):
    n = len(objpoints) # number of views
    """
        Compute homography matrix for each image
    """
    H_list = [calculate_homography(obj, img) for obj, img in zip(objpoints, imgpoints)]

    """
        Find intrinsic matrix K
    """
    V = np.zeros((2 * n, 6))
    for i, h in enumerate(H_list):
        V[2*i, :] = v_pq(h, 0, 1)
        V[2*i+1, :] = v_pq(h, 0, 0) - v_pq(h, 1, 1)

    u, s, vh = np.linalg.svd(V)
    b_val = vh[np.argmin(s), :]
    # Make B positive definite
    if b_val[0] < 0 or b_val[2] < 0 or b_val[5] < 0:
        b_val = -b_val
    # B = L * L.T
    B = np.array([
        [b_val[0], b_val[1], b_val[3]],
        [b_val[1], b_val[2], b_val[4]],
        [b_val[3], b_val[4], b_val[5]]
    ])
    # Use Cholesky Decomposition to get L
    L = np.linalg.cholesky(B)
    # K = L^{-T}
    K = L[2, 2] * np.linalg.inv(L.T)

    print("-" * 75)
    print("L:")
    print(L)
    print("inv(L):")
    print(np.linalg.inv(L))
    print("inv(L).T:")
    print(np.linalg.inv(L).T)
    print("inv(L.T):")
    print(np.linalg.inv(L.T))
    print("K:")
    print(K)
    print("-" * 75)

    """
        Find extrinsic matrix
    """
    extrinsics = []
    for i, h in enumerate(H_list):
        K_inv = np.linalg.inv(K)    
        h0, h1, h2 = h[:, 0], h[:, 1], h[:, 2]
        lambda_ = 1 / np.linalg.norm(np.dot(K_inv, h0)) # scale_factor (λ), p79
        # Make r0 and r1 to orthonormal
        r0 = lambda_ * np.dot(K_inv, h0)
        r1 = lambda_ * np.dot(K_inv, h1)
        r2 = np.cross(r0, r1) # Because r0 and r1 are orthonormal, so r2 = r0 x r1
        t = lambda_ * np.dot(K_inv, h2).reshape(-1, 1)
        R_init = np.array([r0, r1, r2]).T
        u, s, vh = np.linalg.svd(R_init)
        R = np.dot(u, vh)
        W = np.hstack((R, t))
        extrinsics.append(W)
    return K, np.array(extrinsics)

#########################################
#####  END OF CALIBRATION FUNCTION  #####
#########################################

# Use our calibration function
K, extrinsics = calibrate_camera(objpoints, imgpoints)
print("K shape:", K.shape)
print("Extrinsics shape:", extrinsics.shape)

# Display results and compare with OpenCV
print("-" * 75)
print('Camera matrix K:')
print(K)
print('Camera matrix K (cv2):')
print(cv_mtx)
print("-" * 75)
print("Camera matrix difference:")
print(np.abs(cv_mtx - K))
print("-" * 75)

# show the camera extrinsics
print("-" * 75)
print('Show the camera extrinsics')
print(K)
print("-" * 75)

# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# camera setting
camera_matrix = cv_mtx if USE_OPENCV else K
extrinsics = cv_extrinsics if USE_OPENCV else extrinsics

cam_width = 0.064/0.1 * 1.5
cam_height = 0.032/0.1 * 1.5
scale_focal = 1600

# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min, Y_min, Z_min = min_values
X_max, Y_max, Z_max = max_values
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title(f'Extrinsic Parameters Visualization, use_opencv={USE_OPENCV}')

# Save the plot as a PNG file
os.makedirs("./output", exist_ok=True)
plt.savefig(f"./output/{DATA_PATH}_{'cv2' if USE_OPENCV else 'ours'}.png")
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""