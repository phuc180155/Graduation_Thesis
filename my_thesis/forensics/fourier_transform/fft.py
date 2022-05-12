# import library
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import math
# Python >= 3.8


def dft_1d(img_arr):
    """perform the naive fourier transform on 1d array"""
    N = img_arr.shape[0]
    values = []
    for k in range(N):
        temp = []
        for n in range(N):
            temp.append(np.exp(-2j * np.pi * n * k / N))
        values.append(temp)
    values = np.array(values)
    return img_arr @ values


def dft_1d_inverse(img_arr):
    """perform the naive inverse fourier transform on 1d array"""
    N = img_arr.shape[0]
    values = []
    for k in range(N):
        temp = []
        for n in range(N):
            temp.append(np.exp(2j * np.pi * n * k / N))
        values.append(temp)
    values = np.array(values)
    return (img_arr @ values) / N


def fft_1d(img_arr):
    """perform the fast fourier transform on 1d array"""
    img_arr = np.asarray(img_arr, dtype=float)
    N = img_arr.shape[0]

    if not math.log2(N).is_integer():
        print("Size of x must be a power of 2")
        exit(0)
    if N <= 32:
        return dft_1d(img_arr)
    else:
        even = fft_1d(img_arr[::2])  # even indices
        odd = fft_1d(img_arr[1::2])  # odd indices
        factor = []
        for i in range(N):
            factor.append(np.exp(-2j * np.pi * i / N))
        factor = np.array(factor)
        term1 = even + factor[:N//2] * odd
        term2 = even + factor[N//2:] * odd
        return np.concatenate([term1, term2])


def fft_1d_inverse(img_arr):
    """perform the fast inverse fourier transform on 1d array"""
    img_arr = np.asarray(img_arr, dtype=complex)
    N = img_arr.shape[0]

    if not math.log2(N).is_integer():
        print("Size of x must be a power of 2")
        exit(0)
    if N <= 32:
        return dft_1d_inverse(img_arr)
    else:
        even = fft_1d_inverse(img_arr[::2])  # even indices
        odd = fft_1d_inverse(img_arr[1::2])  # odd indices
        factor = []
        for i in range(N):
            factor.append(np.exp(2j * np.pi * i / N))
        factor = np.array(factor)
        term1 = even + factor[:N//2] * odd
        term2 = even + factor[N//2:] * odd
        return np.concatenate([term1, term2]) / N


def dft_2d_helper(img_arr):
    """perform helper of the naive fourier transform on 2d array"""
    result = []
    for i in range(img_arr.shape[0]):
        result.append(dft_1d(img_arr[i, :]))
    return np.array(result)


def dft_2d(img_arr):
    """perform the naive fourier transform on 2d array"""
    img_arr = np.asarray(img_arr, dtype=float)
    return dft_2d_helper(dft_2d_helper(img_arr).T).T


def dft_2d_inverse_helper(img_arr):
    """perform helper of the inverse fourier transform on 2d array"""
    result = []
    for i in range(img_arr.shape[0]):
        result.append(dft_1d_inverse(img_arr[i, :]))
    return np.array(result)


def dft_2d_inverse(img_arr):
    """perform the inverse fourier transform on 2d array"""
    img_arr = np.asarray(img_arr, dtype=complex)
    return dft_2d_inverse_helper(dft_2d_inverse_helper(img_arr).T).T


def fft_2d_helper(img_arr):
    """perform helper of the fast fourier transform on 2d array"""
    M, N = img_arr.shape
    if not math.log2(N).is_integer():
        print("Size of x must be a power of 2")
        exit(0)
    if N <= 32:
        return dft_2d_helper(img_arr)
    else:
        even = fft_2d_helper(img_arr[:, ::2])
        odd = fft_2d_helper(img_arr[:, 1::2])
        factor = []
        for m in range(M):
            temp = []
            for n in range(N):
                temp.append(np.exp(-2j * np.pi * n / N))
            factor.append(temp)
        factor = np.array(factor)
        term1 = even + np.multiply(factor[:, :N//2], odd)
        term2 = even + np.multiply(factor[:, N//2:], odd)
        return np.hstack([term1, term2])


def fft_2d(img_arr):
    """perform the fast fourier transform on 2d array"""
    img_arr = np.asarray(img_arr, dtype=float)
    return fft_2d_helper(fft_2d_helper(img_arr).T).T


def fft_2d_inverse_helper(img_arr):
    """perform helper of the fast inverse fourier transform on 2d array"""
    M, N = img_arr.shape
    if not math.log2(N).is_integer():
        print("Size of x must be a power of 2")
        exit(0)
    if N <= 32:
        return dft_2d_inverse_helper(img_arr)
    else:
        even = fft_2d_inverse_helper(img_arr[:, ::2])
        odd = fft_2d_inverse_helper(img_arr[:, 1::2])
        factor = []
        for m in range(M):
            temp = []
            for n in range(N):
                temp.append(np.exp(2j * np.pi * n / N))
            factor.append(temp)
        factor = np.array(factor)
        term1 = even + np.multiply(factor[:, :N//2], odd)
        term2 = even + np.multiply(factor[:, N//2:], odd)
        return np.hstack([term1, term2]) / (M * N)


def fft_2d_inverse(img_arr):
    """perform the fast inverse fourier transform on 2d array"""
    img_arr = np.asarray(img_arr, dtype=complex)
    return fft_2d_inverse_helper(fft_2d_inverse_helper(img_arr).T).T


def resize_and_fft(path):
    """resize image and apply fast fourier transform on image"""
    # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(path, flags=0)
    new_dim = (512, 256)
    resize_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    fft_img = fft_2d(resize_img)
    np_fft_img = np.fft.fft2(resize_img)  # use built-in function
    print("FFT on image {} done.".format(path))
    return resize_img, fft_img, np_fft_img


def mode1(path):
    img, fft_img, np_fft_img = resize_and_fft(path)
    fig, axs = plt.subplots(1, 3, figsize=(8, 6))
    # plot original image
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original image')
    # plot fft image
    axs[1].imshow(np.abs(fft_img), norm=LogNorm())
    axs[1].set_title('FFT on image')
    # plot numpy fft image
    axs[2].imshow(np.abs(np_fft_img), norm=LogNorm())
    axs[2].set_title("Numpy FFT on image")
    fig.suptitle('Mode 1')
    plt.show()


def mode2(path):
    img, fft_img, _ = resize_and_fft(path)
    print(fft_img)
    keep = 0.05  # init keep ratio
    keep_r = np.sqrt(keep) / 2
    # get rows and columns of fft image
    r, c = fft_img.shape
    fft_img[int(r*keep_r):int(r*(1-keep_r)), :] = 0.
    fft_img[:, int(c*keep_r):int(c*(1-keep_r))] = 0.
    print("Fraction of non-zeros: ", keep)
    # get number of non-zeros
    num_nonzero = fft_img[np.nonzero(fft_img)].shape[0]
    print("Number of non-zeros {} out of {}".format(num_nonzero, r*c))
    # use fft_2d_inverse to reconstruct the image
    denoised_img = fft_2d_inverse(fft_img).real
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    # plot original image
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original image')
    # plot denoise image
    axs[1].imshow(denoised_img, cmap='gray')
    axs[1].set_title('Denoise image with {}% high and {}% low frequency removed'.format(keep*100, keep*100))
    fig.suptitle("Mode 2")
    plt.show()


def compress_func(img, level):
    """perform compress function"""
    # get dim of image
    h, w = img.shape
    size = h * w
    threshold = int(level * size) // 100
    temp = np.abs(img)
    # get the indices with threshold
    indices = np.argpartition(temp, threshold, axis=None)
    img_temp = img.flatten()  # flatten the image
    for i in range(threshold):
        img_temp[indices[i]] = 0
    compressed_img = np.reshape(img_temp, (h, w))
    # save matrix
    np.savez_compressed("compression_level_{}".format(level), compressed_img)
    return fft_2d_inverse(compressed_img)


def mode3(path):
    img, fft_img, _ = resize_and_fft(path)
    r, c = fft_img.shape
    level_list = [[0, 10, 20], [40, 65, 99.9]]

    # plot compress images
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    for i in range(2):
        for j in range(3):
            level = level_list[i][j]
            print("Compression level {} - Number of non-zeros on image ({}x{}): {}".format(level, r, c, r * c * (100 - level) // 100))

            compressed_img = compress_func(fft_img, level)
            axs[i, j].imshow(compressed_img.real, cmap='gray')
            axs[i, j].set_title("Level: {}".format(level))
    fig.suptitle('Mode 3')
    plt.show()


def mode4():
    plot, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('size')
    ax.set_ylabel('runtime (s)')
    methods = [fft_2d, dft_2d]
    names = ["Fast fourier transform", "Naive fourier transform"]
    colors = ['green', 'blue']

    # loop with 2 algorithms
    for i in range(2):
        print("Algorithm: {} ({})".format(names[i], colors[i]))

        x_data = []
        y_data = []
        std_list = []
        sizes = [32, 64, 128, 256, 512, 1024]
        for size in sizes:
            x_data.append(size)

            # create random 2d array
            r_img = np.random.rand(size, size)

            rt_list = []
            for j in range(10):  # 10 times
                start = time.time()
                methods[i](r_img)
                end = time.time()
                rt_list.append(end - start)

            rt_mean = np.mean(rt_list)
            rt_std = np.std(rt_list, ddof=1)
            std_list.append(rt_std)
            print("Size: {}, mean: {}, std: {}".format(size, rt_mean, rt_std))
            y_data.append(rt_mean)

        plt.errorbar(x_data, y_data, yerr=std_list, fmt=colors[i])
    plt.show()


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', dest='mode', help="Mode (1-4)", type=int, default=1)
    parser.add_argument('-i', action='store', dest='image', help="Image name", type=str, default="moonlanding.png")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli()
    mode = args.mode
    img_path = args.image
    if not os.path.exists(img_path):
        print("The path of image does not exist")
        exit(0)
    if mode == 1:
        mode1(img_path)
    elif mode == 2:
        mode2(img_path)
    elif mode == 3:
        mode3(img_path)
    elif mode == 4:
        mode4()
    else:
        print("Mode must be between 1 and 4.")
