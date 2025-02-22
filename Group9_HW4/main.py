
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import sample
import random


def feature_matching(img1, img2, kp1, kp2, threshold=0.5):
    matches = []
    img1 = np.array(img1)
    img2 = np.array(img2)

    for i, f1 in enumerate(img1):
        ssd_distances = np.sum((img2 - f1) ** 2, axis=1)
        idx = np.argsort(ssd_distances)
        ratio = ssd_distances[idx[0]] / ssd_distances[idx[1]
                                                      ] if ssd_distances[idx[1]] != 0 else float('inf')

        if ratio <= threshold:
            matches.append([i, idx[0]])

    point1 = []
    point2 = []
    for idx1, idx2 in matches:
        p1 = tuple(map(round, kp1[idx1].pt))
        p2 = tuple(map(round, kp2[idx2].pt))

        point1.append([p1[0], p1[1]])
        point2.append([p2[0], p2[1]])

    return np.array(point1), np.array(point2)


def ransac(img1, img2, point1, point2, sample=8, num_iterations=5000, threshold=1.5):
    best_F = np.zeros((3, 3))
    best_inlier1 = []
    best_inlier2 = []
    min_error = np.inf
    for _ in range(num_iterations):
        idx = random.sample(range(len(point1)), sample)

        sample_point1 = np.array([point1[i] for i in idx])
        sample_point2 = np.array([point2[i] for i in idx])

        F = fundamental_matrix(sample_point1, sample_point2, img1, img2)

        inlier1 = []
        inlier2 = []
        error = 0
        for i in range(len(point1)):
            p1 = np.array([*point1[i], 1.0])
            p2 = np.array([*point2[i], 1.0])
            projected_p1 = np.dot(F, p1)
            dis = np.abs(np.dot(p2.T, projected_p1)) / \
                np.sqrt(projected_p1[0]**2 + projected_p1[1]**2)
            error += dis

            if dis <= threshold:
                inlier1.append(point1[i])
                inlier2.append(point2[i])

        if error < min_error:
            min_error = error
            best_F = F
            best_inlier1 = inlier1
            best_inlier2 = inlier2

    return best_F, np.array(best_inlier1), np.array(best_inlier2)


def normalize_points(points, img_width, img_height):
    norm_matrix = np.array([
        [2.0 / img_width, 0, -1],
        [0, 2.0 / img_height, -1],
        [0, 0, 1]
    ])

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized_points = (norm_matrix @ points_homogeneous.T).T

    return normalized_points[:, :2], norm_matrix


def fundamental_matrix(point1, point2, img1, img2):

    norm_point1, norm_mat1 = normalize_points(
        point1, img1.shape[1], img1.shape[0])
    norm_point2, norm_mat2 = normalize_points(
        point2, img2.shape[1], img2.shape[0])

    A = []
    for p1, p2 in zip(norm_point1, norm_point2):
        x1, y1 = p1
        x2, y2 = p2
        A.append(np.asarray([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]))

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F_normalized)
    S[2] = 0
    F_normalized = U @ np.diag(S) @ Vt

    F = norm_mat2.T @ F_normalized @ norm_mat1

    F /= F[-1, -1]

    return F


def draw_epipolar_lines(F, img1, img2, point1, point2):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    match_lines_1 = np.zeros(shape=img1.shape, dtype='uint8')
    match_lines_2 = np.zeros(shape=img2.shape, dtype='uint8')

    match_lines_1[:, :] = img1
    match_lines_2[:, :] = img2

    for pt_1, pt_2 in zip(point1, point2):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        vec_2 = np.array([pt_2[0], pt_2[1], 1])
        L = np.dot(F, vec_2)
        line_pt_1 = (0, -L[2] / L[1])
        line_pt_2 = (height1, (-L[2] - (L[0] * height1)) / L[1])

        pt = (int(pt_1[0]), int(pt_1[1]))
        line_1 = (int(line_pt_1[0]), int(line_pt_1[1]))
        line_2 = (int(line_pt_2[0]), int(line_pt_2[1]))

        cv2.circle(match_lines_1, pt, 4, color)
        cv2.line(match_lines_1, line_1, line_2, color)

        pt2 = (int(pt_2[0]), int(pt_2[1]))
        cv2.circle(match_lines_2, pt2, 4, color, -1)

    match_lines_blend = np.zeros(
        (max(height1, height2), width1+width2, 3), dtype='uint8')

    match_lines_blend[0:height1, 0:width1] = match_lines_1
    match_lines_blend[0:height2, width1:] = match_lines_2

    return match_lines_blend


def compute_essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1

    U, S, Vt = np.linalg.svd(E)

    S_corrected = np.diag([S[0], S[1], 0])

    E_corrected = U @ S_corrected @ Vt

    U, _, Vt = np.linalg.svd(E_corrected)

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    poses = [
        np.hstack((U @ W @ Vt, U[:, 2:3])),
        np.hstack((U @ W @ Vt, -U[:, 2:3])),
        np.hstack((U @ W.T @ Vt, U[:, 2:3])),
        np.hstack((U @ W.T @ Vt, -U[:, 2:3]))
    ]

    return E_corrected, poses


def plot_3d_points(points_3d):

    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=z, cmap='viridis', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def triangulate_points(P1, P2, points1, points2):
    num_points = points1.shape[0]
    points_3D = []

    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]

        A = np.zeros((4, 4))
        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[-1]

        points_3D.append(X[:3])

    return np.array(points_3D)


def find_best_pose(E, poses, K1, K2, points1, points2):
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K1 @ P1

    max_positive_depth = 0
    best_3D_points = None
    best_pose = None

    for P2 in poses:
        P2 = K2 @ P2
        points_3D = triangulate_points(
            P1, P2, points1, points2)

        positive_depth = np.sum(points_3D[:, 2] > 0)

        if positive_depth > max_positive_depth:
            max_positive_depth = positive_depth
            best_3D_points = points_3D
            best_pose = P2

    return best_pose, best_3D_points


if __name__ == '__main__':
    data = [['Mesona1.JPG', 'Mesona2.JPG'], ['Statue1.bmp', 'Statue2.bmp'], ['test1.JPG', 'test2.JPG'], ['test3.JPG', 'test4.JPG']

            ]
    data_path = './my_data'

    img_choose = 2

    img1 = cv2.imread(data_path+'/'+data[img_choose][0])
    img2 = cv2.imread(data_path+'/'+data[img_choose][1])
    # while (img2.shape[0] > 1000):
    #     if img1.shape == img2.shape:
    #         img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
    #     img2 = cv2.resize(img2, None, fx=0.5, fy=0.5)
    # print(img1.shape)
    # print(img2.shape)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)


    p1, p2 = feature_matching(des1, des2, kp1, kp2)
    
    F, inlier1, inlier2 = ransac(img1, img2,
                                 p1, p2)

    epipolar_image = draw_epipolar_lines(F, img1, img2, inlier1, inlier2)
    cv2.imwrite('./output/result.jpg', epipolar_image)

    # K = np.array([[1.4219, 0.0005, 0.5092],
    #               [0, 1.4219, 0.3802],
    #               [0, 0, 0.0010]])
    # K = np.array([[5426.566895, 0.678017, 330.096680],
    #               [0.000000, 5423.133301, 648.950012],
    #               [0.000000, 0.000000, 1.000000]])
    K = np.array([[6659.18470, 0.0000, 1105.86479],
                  [0.0000, 6659.18470, 953.87568],
                  [0.0000, 0.0000, 1.0000]])

    E, poses = compute_essential_matrix(F, K, K)

    E, points_3d = find_best_pose(E, poses, K, K, inlier1, inlier2)
    plot_3d_points(points_3d)

    with open('./output/3d_coord.txt', 'w') as f:
        for coord in points_3d:
            f.write("{} {} {};\n".format(coord[0], coord[1], coord[2]))

    with open('./output/2d_coord.txt', 'w') as f:
        for coord in inlier1:
            f.write("{} {};\n".format(coord[1], coord[0]))

