import cv2
import os
import numpy as np
from numpy.fft import fft2, ifft2
import math
import argparse


def shift_frequency(img):
    img_shift = np.zeros(img.shape)
    for c in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_shift[i, j, c] = img[i, j, c] * ((-1)**(i+j))
    return img_shift


def ideal_low_filter(img, cutoff_ratio):
    h, w, c = img.shape
    mid_h = h // 2
    mid_w = w // 2
    cutoff_frequency = math.ceil((min(h, w) * cutoff_ratio) / 2)
    y, x = np.ogrid[:h, :w]

    distance = np.sqrt((x - mid_w)**2 + (y - mid_h)**2)
    low_filter = np.where(distance <= cutoff_frequency, 1, 0)
    return low_filter


def gaussian_low_filter(img, cutoff_ratio):
    h, w, c = img.shape
    mid_h = h // 2
    mid_w = w // 2
    cutoff_frequency = math.ceil(min(h, w) * cutoff_ratio/2)

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - mid_w)**2 + (y - mid_h)**2)
    low_filter = np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))

    return low_filter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_choose', type=int, default=0)
    parser.add_argument('--filter_type', type=str, default="Gaussian",
                        choices=["Gaussian", "Ideal"], help='Type of filter to apply')
    parser.add_argument('--low_cutoff', type=float, default=0.05,
                        help='Low-pass filter cutoff ratio (D0)')
    parser.add_argument('--high_cutoff', type=float, default=0.1,
                        help='High-pass filter cutoff ratio (D1)')
    args = parser.parse_args()

    data = [["0_Afghan_girl_after.jpg", "0_Afghan_girl_before.jpg"],
            ["1_bicycle.bmp", "1_motorcycle.bmp"],
            ["2_bird.bmp", "2_plane.bmp"],
            ["3_cat.bmp", "3_dog.bmp"],
            ["4_einstein.bmp", "4_marilyn.bmp"],
            ["5_fish.bmp", "5_submarine.bmp"],
            ["6_makeup_after.jpg", "6_makeup_before.jpg"],
            ["test2.jpg", "test1.jpg"]
            ]

    data_path = "./data/task1and2_hybrid_pyramid"
    result_path = "./output/task1"
    os.makedirs(result_path, exist_ok=True)

    high_img_path = data[args.img_choose][0]
    low_img_path = data[args.img_choose][1]

    low_img = cv2.imread(os.path.join(data_path, low_img_path))
    high_img = cv2.imread(os.path.join(data_path, high_img_path))

    if low_img.shape[0] != high_img.shape[0] or low_img.shape[1] != high_img.shape[1]:
        dsize = (min(low_img.shape[1], high_img.shape[1]), min(
            low_img.shape[0], high_img.shape[0]))
        low_img = cv2.resize(low_img, dsize)
        high_img = cv2.resize(high_img, dsize)

    if args.filter_type == "Gaussian":
        low_filter = gaussian_low_filter(low_img, args.low_cutoff)
        high_filter = 1 - gaussian_low_filter(high_img, args.high_cutoff)
    else:
        low_filter = ideal_low_filter(low_img, args.low_cutoff)
        high_filter = 1 - ideal_low_filter(high_img, args.high_cutoff)

    low_img_shift = shift_frequency(low_img.astype(np.float32))
    high_img_shift = shift_frequency(high_img.astype(np.float32))

    low_result = np.zeros(low_img_shift.shape)
    high_result = np.zeros(high_img_shift.shape)

    for c in range(low_img_shift.shape[2]):
        low_img_fft = fft2(low_img_shift[:, :, c])
        high_img_fft = fft2(high_img_shift[:, :, c])
        low_img_filter = low_img_fft * low_filter
        high_img_filter = high_img_fft * high_filter

        low_result[:, :, c] = ifft2(low_img_filter).real
        high_result[:, :, c] = ifft2(high_img_filter).real

    hybrid_img = (low_result+high_result).real
    hybrid_result = shift_frequency(hybrid_img)

    cv2.imwrite(os.path.join(
        result_path, f"low_{args.img_choose}_{args.filter_type}_{args.low_cutoff}.jpg"), low_result)
    cv2.imwrite(os.path.join(
        result_path, f"high_{args.img_choose}_{args.filter_type}_{args.high_cutoff}.jpg"), high_result)
    cv2.imwrite(os.path.join(
        result_path, f"hybrid_{args.img_choose}_{args.filter_type}.jpg"), hybrid_result)