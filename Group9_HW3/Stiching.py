import os
import random
from random import sample
import numpy as np
import cv2 as cv2


def feature_matching(src1, src2, threshold=0.5):
    matches = []
    src1 = np.array(src1)
    src2 = np.array(src2)

    for i, f1 in enumerate(src1):
        ssd_distances = np.sum((src2 - f1) ** 2, axis=1)
        idx = np.argsort(ssd_distances)
        ratio = ssd_distances[idx[0]] / ssd_distances[idx[1]
                                                      ] if ssd_distances[idx[1]] != 0 else float('inf')

        if ratio <= threshold:
            matches.append([i, idx[0]])

    return np.array(matches)


def draw_match_line(img1, img2, kp1, kp2, matches):
    width = img1.shape[1]

    combined_image = cv2.hconcat([img1, img2])

    point1 = []
    point2 = []
    for idx1, idx2 in matches:
        color = tuple(random.randint(0, 255) for _ in range(3))
        p1 = tuple(map(round, kp1[idx1].pt))
        p2 = (round(kp2[idx2].pt[0])+width, round(kp2[idx2].pt[1]))

        point1.append([p1[1], p1[0]])
        point2.append([p2[1], p2[0]-width])

        cv2.circle(combined_image, p1, 5, color, -1)
        cv2.line(combined_image, p1, p2, color, 1)
        cv2.circle(combined_image, p2, 5, color, -1)

    return combined_image, point1, point2


def ransac(point1, point2, sample=32, num_iterations=300, threshold=5.0):
    max_inliers = 0
    best_H = np.zeros((3, 3))

    for _ in range(num_iterations):
        idx = random.sample(range(len(point1)), sample)

        sample_point1 = np.array([point1[i] for i in idx])
        sample_point2 = np.array([point2[i] for i in idx])

        H = homomat(sample_point1, sample_point2)

        inliers_count = 0
        for i in range(len(point1)):
            p2 = np.array([*point2[i], 1.0])
            projected_p2 = np.dot(H, p2)
            projected_p2 /= projected_p2[2]

            error = np.linalg.norm(point1[i] - projected_p2[:2])

            if error < threshold:
                inliers_count += 1

        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_H = H

    return best_H


def homomat(point1, point2):
    A = []

    for p1, p2 in zip(point1, point2):
        u, v = p1
        x, y = p2

        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

    _, _, Vt = np.linalg.svd(np.array(A))

    H = Vt[-1].reshape(3, 3)
    H /= H[-1, -1]

    return H


def warp(img1, img2, H):
    height, width, _ = img1.shape
    H_inv = np.linalg.inv(H)

    corners = np.array([[0, 0, 1], [0, width, 1], [
                       height, 0, 1], [height, width, 1]])
    transformed_corners = np.dot(H, corners.T).T
    transformed_corners /= transformed_corners[:, -1][:, np.newaxis]

    y_offset = int(round(np.min(transformed_corners[0:2, 0])))
    y_bound = int(round(np.max(transformed_corners[2:4, 0]))) - y_offset
    x_offset = int(round(np.min(transformed_corners[[0, 2], 1])))
    x_offset = min(0, x_offset)
    x_bound = int(round(np.max(transformed_corners[[1, 3], 1]))) - x_offset

    result = np.zeros((y_bound, x_bound, 3), dtype='uint8')

    overlap_img1 = np.zeros((y_bound, x_bound, 3), dtype='uint8')
    overlap_img2 = np.zeros((y_bound, x_bound, 3), dtype='uint8')

    y_start = int(round(np.min(transformed_corners[0:2, 0])))
    y_end = int(round(np.max(transformed_corners[2:4, 0])))
    x_start = int(round(np.min(transformed_corners[[0, 2], 1])))
    x_end = int(round(np.max(transformed_corners[[1, 3], 1])))

    for i in range(height):
        for j in range(width):
            x, y = i - y_offset, j - x_offset
            coord = np.dot(H_inv, np.array([i, j, 1]))
            coord /= coord[-1]
            u, v = int(round(coord[0])), int(round(coord[1]))

            if u >= 0 and u < height and v >= 0 and v < width:
                overlap_img1[x, y, :] = img1[i, j, :]
            else:
                result[x, y, :] = img1[i, j, :]

    x_min, x_max = float('inf'), float('-inf')
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            coord = (H_inv @ np.array([i, j, 1]))
            coord /= coord[-1]
            u, v = int(round(coord[0])), int(round(coord[1]))
            if u >= 0 and u < height and v >= 0 and v < width:
                x = i - y_offset
                y = j - x_offset
                if i >= 0 and i < height and j >= 0 and j < width:
                    overlap_img2[x, y, :] = img2[u, v, :]
                    x_min = min(x_min, j)
                    x_max = max(x_max, j)
                else:
                    result[x, y, :] = img2[u, v, :]

    x_range = (x_max - x_min)

    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            coord = np.dot(H_inv, np.array([i, j, 1]))
            coord /= coord[-1]

            u = int(round(coord[0]))
            v = int(round(coord[1]))

            if u >= 0 and u < height and v >= 0 and v < width:
                x = i - y_offset
                y = j - x_offset
                if i >= 0 and i < height and j >= 0 and j < width:
                    weight = (j - x_min) / x_range
                    result[x, y] = (1-weight)*img1[i, j] + weight*img2[u, v]

    return result


if __name__ == '__main__':
    data = [['S1.JPG', 'S2.JPG'],
            ['TV1.JPG', 'TV2.JPG'],
            ['hill1.JPG', 'hill2.JPG'],
            ['me1.JPG', 'me2.JPG'],
            ['wall1.JPG', 'wall2.JPG'],
            ]
    data_path = './data'

    result_path = './result'

    img_choose = 0

    img1 = cv2.imread(data_path+'/'+data[img_choose][0])
    img2 = cv2.imread(data_path+'/'+data[img_choose][1])

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = feature_matching(des1, des2)
    img_matches, p1, p2 = draw_match_line(img1, img2, kp1, kp2, matches)

    cv2.imshow("Manual Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    H = ransac(p1, p2)

    result = warp(img1, img2, H)

    cv2.imwrite('./result/test.jpg', result)
