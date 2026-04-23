import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. 이미지 불러오기
img = cv2.imread('apples.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 전처리 (그레이 + 블러)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Sobel (보정 포함)
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel) 

# 4. Canny
canny = cv2.Canny(blur, 100, 200)

# 5. Contour 
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contour = img_rgb.copy()

# 작은 잡음 제거
filtered_contours = []
for cnt in contours:
    if cv2.contourArea(cnt) > 200: 
        filtered_contours.append(cnt)

cv2.drawContours(img_contour, filtered_contours, -1, (0, 255, 0), 2)

# 6. 결과 출력
plt.figure(figsize=(10, 8))

plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(sobel, cmap='gray')
plt.title("Sobel")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(img_contour)
plt.title("Contour (Filtered)")
plt.axis('off')

plt.tight_layout()
plt.show()