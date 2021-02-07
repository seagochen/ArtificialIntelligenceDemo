import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image as a gray scale image
img_gray = cv2.imread("Data/cv_logo.png", cv2.IMREAD_GRAYSCALE)
img_copy = img_gray.copy()

# concatenate two images by horizontally and vertically
horizon = np.concatenate((img_gray, img_copy), axis=1)
vertical = np.concatenate((img_gray, img_copy), axis=0)

# plot images
plt.subplot(1, 2, 1); plt.imshow(horizon, cmap="gray"); plt.axis('off'); plt.title('horizontally')
plt.subplot(1, 2, 2); plt.imshow(vertical, cmap="gray"); plt.axis('off'); plt.title('vertically')

plt.show()