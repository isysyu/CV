import numpy as np
import os
import os.path as path
import cv2
import matplotlib.pyplot as plt
from math import log2, ceil


def split_image(src):
    """將輸入圖像分割為三個part，分別對應B,G,R三個channel"""
    height, width = src.shape
    split_height = height // 3

    # 由上至下是藍,綠,紅
    b = src[0:split_height, :]
    g = src[split_height:2 * split_height, :]
    r = src[2 * split_height:3 * split_height, :]
    return b, g, r


def padding_zero(img, space):
    """在圖像周圍補0"""
    height, width = img.shape
    temp_image = np.zeros([height + space * 2, width + space * 2])
    temp_image[space:-space, space:-space] = img
    return temp_image


def filter_2D(src, kernel):
    """實現2D filter"""
    height, width = src.shape
    kernel_size = kernel.shape[0]
    space = kernel_size // 2

    dst = np.zeros((height, width))
    pad_image = padding_zero(src, space)

    # 對整張圖片做計算
    for i in range(height):
        for j in range(width):
            region = pad_image[i:i + kernel_size, j:j + kernel_size]
            dst[i, j] = np.sum(region * kernel)

    return dst


def sobel(src):
    """用Sobel算子計算graph的gradient以檢測邊緣"""
    # 轉float64不然一直overflow
    src = src.astype(np.float64)

    # Sobel算子, 一個垂直一個水平
    so_y = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    so_x = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

    grad_x = filter_2D(src, so_x)
    grad_y = filter_2D(src, so_y)

    dst = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return dst


def resize(img, size):
    """用Bilinear Interpolation來Resize"""
    src_h, src_w = img.shape
    dst_h, dst_w = size

    scale_x = src_w / dst_w
    scale_y = src_h / dst_h

    dst = np.zeros((dst_h, dst_w))

    for y in range(dst_h):
        for x in range(dst_w):
            src_x = x * scale_x
            src_y = y * scale_y

            # X1,y1是為了防止越界
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, src_w - 1)
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, src_h - 1)

            wx = src_x - x0
            wy = src_y - y0
            dst[y, x] = (1 - wx) * (1 - wy) * img[y0, x0] + \
                        wx * (1 - wy) * img[y0, x1] + \
                        (1 - wx) * wy * img[y1, x0] + \
                        wx * wy * img[y1, x1]
    return dst


def NCC_score(a, b):
    """用NCC來計算兩個graph的相似度"""
    # 一樣轉float64 prevent overfit
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_centered = a - a_mean
    b_centered = b - b_mean

    a_norm = np.sqrt(np.sum(a_centered ** 2))
    b_norm = np.sqrt(np.sum(b_centered ** 2))
    epsilon = 1e-10
    if a_norm < epsilon or b_norm < epsilon:
        return 0

    correlation = np.sum(a_centered * b_centered) / (a_norm * b_norm)  # 這是相似度

    return np.clip(correlation, -1, 1)


def image_shift(src, idx):
    """對圖像進行平移操作"""
    height, width = src.shape
    dst = np.zeros((height, width))
    x_shift, y_shift = idx

    if x_shift >= 0:
        src_x_s = 0
        src_x_e = width - x_shift
        dst_x_s = x_shift
        dst_x_e = width
    else:
        src_x_s = -x_shift
        src_x_e = width
        dst_x_s = 0
        dst_x_e = width + x_shift

    if y_shift >= 0:
        src_y_s = 0
        src_y_e = height - y_shift
        dst_y_s = y_shift
        dst_y_e = height
    else:
        src_y_s = -y_shift
        src_y_e = height
        dst_y_s = 0
        dst_y_e = height + y_shift

    dst[dst_y_s:dst_y_e, dst_x_s:dst_x_e] = \
        src[src_y_s:src_y_e, src_x_s:src_x_e]

    return dst


def align(a, b, n=15):
    """by 圖像大小來動態調整範圍，並用中心的部分來對齊"""
    best_score = -1
    best_index = [0, 0]

    if n is None:
        n = min(a.shape) // 20

    h, w = a.shape
    margin_h = h // 4
    margin_w = w // 4
    a_center = a[margin_h:-margin_h, margin_w:-margin_w]
    b_center = b[margin_h:-margin_h, margin_w:-margin_w]

    for y_shift in range(-n, n + 1):
        for x_shift in range(-n, n + 1):
            shifted_b = image_shift(b_center, [x_shift, y_shift])
            score = NCC_score(a_center, shifted_b)
            if score > best_score:
                best_score = score
                best_index = [x_shift, y_shift]

    return best_index


def downsample_image(src, down_size=500):
    """downsample->加速"""
    height, width = src.shape
    size = max(height // 3, width)
    dst = src
    scale = 1
    level = 1

    if size > down_size:
        level = ceil(log2(size / down_size))
        scale = 2 ** level
        dst = resize(src, (height // scale, width // scale))

    return dst, scale


def align_channels(src):
    """對齊三個channel(綠跟紅對齊藍)"""
    if src is None or len(src.shape) != 2:
        raise ValueError("輸入圖片有誤")

    resize_src, scale = downsample_image(src)
    resize_b, resize_g, resize_r = split_image(resize_src)

    edge_b = sobel(resize_b)
    edge_g = sobel(resize_g)
    edge_r = sobel(resize_r)

    g_idx = align(edge_b, edge_g)
    r_idx = align(edge_b, edge_r)

    print('green channel shift:', g_idx)
    print('red channel shift:', r_idx)

    scaled_g_idx = [g_idx[0] * scale, g_idx[1] * scale]
    scaled_r_idx = [r_idx[0] * scale, r_idx[1] * scale]

    b, g, r = split_image(src)
    align_g = image_shift(g, scaled_g_idx)
    align_r = image_shift(r, scaled_r_idx)

    dst = np.dstack((align_r, align_g, b)).astype(np.uint8)
    return dst


def combine_channels_no_align(src):
    """直接疊加三個channel(不做對齊)"""
    if src is None or len(src.shape) != 2:
        raise ValueError("輸入圖片有誤")

    # 直接分割三個channel
    b, g, r = split_image(src)

    # 直接疊加，不做任何shift
    dst = np.dstack((r, g, b)).astype(np.uint8)
    return dst


if __name__ == '__main__':
    #建立兩個資料夾，分別放有align跟沒有align的
    cur = os.getcwd()
    aligned_folder = path.join(cur, 'output/task3_aligned')
    no_align_folder = path.join(cur, 'output/task3_no_align')

    for folder in [aligned_folder, no_align_folder]:
        if not path.isdir(folder):
            os.makedirs(folder)

    dir_path = './data/task3_colorizing'
    for f in os.listdir(dir_path):
        img_path = path.join(dir_path, f)
        if not path.isfile(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print('處理圖片:', f)
        print('圖片大小: %d x %d' % (img.shape[1], img.shape[0]))

        # 處理對齊版本
        aligned_image = align_channels(img)
        aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR)
        aligned_save_path = path.join(aligned_folder, "%s_align.png" % os.path.splitext(f)[0])
        cv2.imwrite(aligned_save_path, aligned_image)

        # 處理不對齊版本
        no_align_image = combine_channels_no_align(img)
        no_align_image = cv2.cvtColor(no_align_image, cv2.COLOR_RGB2BGR)
        no_align_save_path = path.join(no_align_folder, "%s_no_align.png" % os.path.splitext(f)[0])
        cv2.imwrite(no_align_save_path, no_align_image)

        print("\n")

    print("對齊版本儲存至:", aligned_save_path)
    print("不對齊版本儲存至:", no_align_save_path)
    print('finish:>')