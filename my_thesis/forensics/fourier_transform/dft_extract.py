# https://github.com/cc-hpc-itwm/DeepFakeDetection
import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import griddata
import argparse
import traceback
import os.path as osp

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    print("x = ", x)
    print("y = ", y)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
        print("Center = ", center)

    r = np.hypot(x - center[0], y - center[1])
    print("r = ", r.shape)

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    print("Sorted: ", r_sorted)
    print("i_sorted", i_sorted)

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def get_interpolated(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon

    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    print("magnitude_spectrum shape: ", magnitude_spectrum.shape)
    psd1D = azimuthalAverage(magnitude_spectrum)

    # Calculate the azimuthally averaged 1D power spectrum
    points = np.linspace(0, N, num=psd1D.size)  # coordinates of a
    xi = np.linspace(0, N, num=N)  # coordinates for interpolation

    interpolated = griddata(points, psd1D, xi, method='cubic')
    interpolated /= interpolated[0]
    return interpolated

data = {}
epsilon = 1e-8
N = 80
y = []
error = []

if __name__ == "__main__":
    img_path = "test/face_test.jpg"
    img = cv2.imread(osp.join(osp.dirname(__file__), img_path), flags=0)
    interpolated = get_interpolated(img)
    print("Interpolated shape: ", interpolated.shape)
