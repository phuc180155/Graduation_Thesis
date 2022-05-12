import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import os.path as osp

img_path = "test/dft_test.jpg"
img = cv2.imread(osp.join(osp.dirname(__file__), img_path), flags=0)

# DFT image:
# @param: src: cần chuyển image sang float, flags=DFT_COMPLEX_OUTPUT để trả về kết quả là ma trận DFT cả real và imaginary
# @return: [H, W, 2] (2: real và imaginary)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Chú ý rằng theo công thức: X_{k, l} = sigma_{n=0}{W-1}sigma_{m=0}{H-1} x_{n, m}. e^[-i*2pi*(k*n/W + l*m/H)]
# Điểm (k, l) trên ảnh spectrum (trên miền tần số) sẽ là: 
# tổng hợp các thành phần tần số (k*n/W + l*m/H) với n, m chạy từ 0->W-1, 0->H-1
# nên tại (k, l) càng bé (ví dụ top-left) thì tần số sẽ là 0
# càng ra xa dần thì tần số sẽ càng lớn
# Cho nên: ta mong muốn transform spectrum image này sao cho ở tâm ảnh tần số càng bé (ví dụ 0)
# càng ra xa dần tần số càng lớn
# => np.fft.shift
dft_shift = np.fft.fftshift(dft)

# Cộng với 1 lượng epsilon để có thể tính log:
dft_shift += 1e-5

# dft_shift vẫn đang là complex => Ta tìm biên độ của phổ của ảnh:
magnitude_spectrum_1 = cv2.magnitude(dft_shift[:,:,0], dft_shift[:, :, 1])
magnitude_spectrum_2 = 2000 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:, :, 1]))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original image")
plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum_1, cmap="gray")
plt.title('Spectrum image 1')
plt.subplot(1, 3, 3)
plt.imshow(magnitude_spectrum_2, cmap="gray")
plt.title('Spectrum image 2')
plt.savefig(osp.join(osp.dirname(__file__), "test/DFT_image.png"), dpi=300, bbox_inches="tight")

"""
    Ảnh sẽ như thế nào nếu lọc đi tần số thấp?
"""
# High-pass filter mask: Lọc freq thấp (ở giữa tâm ảnh spectrum)
H, W = img.shape[:2]
crow, ccol = H//2, W//2

# Create a mask with two channels because DFT result in 2D
mask = np.ones((H, W, 2), dtype=np.uint8)

# Circular radius from center
r = H // 20
center = [crow, ccol]

# @return x: np.ndarray([[0], [1]...[H-1]], shape=(H, 1))
# @return y: np.ndarray([[0, 1... W-1]], shape=(1, W))
x, y = np.ogrid[:H, :W]

# Tạo mask_area: Lọc các tần số thấp <=> mask = 0
mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
mask[mask_area] = 0

f_shift = dft_shift * mask
f_shift += 1e-5

filtered_magnitude_spectrum = 20 * np.log(cv2.magnitude(np.array(f_shift[:, :, 0], dtype=np.float32), \
                                                        np.array(f_shift[:, :, 1], dtype=np.float32)))

f_ishift = np.fft.ifftshift(f_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(np.array(img_back[:, :, 0], dtype=np.float32), \
                         np.array(img_back[:, :, 1], dtype=np.float32))

# Low-pass filtering
lowpass_mask = np.ones((H, W, 2), dtype=np.uint8)
mask_area = (x - center[0])**2 + (y - center[1])**2 >= r*r
lowpass_mask[mask_area] = 0
f_shift_2 = dft_shift * lowpass_mask
f_shift_2 += 1e-5

low_pass_filtered_magnitude_spectrum = 20 * np.log(cv2.magnitude(np.array(f_shift_2[:, :, 0], dtype=np.float32), \
                                                                   np.array(f_shift_2[:, :, 1], dtype=np.float32)))
f_ishift_2 = np.fft.ifftshift(f_shift_2)
img_back_2 = cv2.idft(f_ishift_2)
img_back_2 = cv2.magnitude(np.array(img_back_2[:, :, 0], dtype=np.float32), \
                         np.array(img_back_2[:, :, 1], dtype=np.float32))

# Plot:
plt.figure(figsize=(20, 20))
plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.subplot(3, 2, 2)
plt.imshow(magnitude_spectrum_2)
plt.title('Spectrum image (FFT of image)')
plt.subplot(3, 2, 3)
# High-pass filtering: Edge detector
plt.imshow(filtered_magnitude_spectrum)
plt.title('Mask of high filtered spectrum image')
plt.subplot(3, 2, 4)
plt.imshow(img_back, cmap='gray')
plt.title('Image after high-pass filtered')
# Low-pass filtering: Blur image
plt.subplot(3, 2, 5)
plt.imshow(low_pass_filtered_magnitude_spectrum)
plt.title('Mask of low filtered spectrum image')
plt.subplot(3, 2, 6)
plt.imshow(img_back_2, cmap='gray')
plt.title('Image after low-pass filtered')
# plt.savefig(osp.join(osp.dirname(__file__), "test/Image_After_Freq_Filter.png"), dpi=300, bbox_inches="tight")
plt.show()





