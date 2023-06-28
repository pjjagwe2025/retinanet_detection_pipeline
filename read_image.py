import cv2

from PIL import Image

# img = cv2.imread('data/STDW-main/STDW-main/images/108000_1.PNG')
# print(img.shape)

img = Image.open('data/STDW-main/STDW-main/images/108000_1.PNG')
print(img.size)