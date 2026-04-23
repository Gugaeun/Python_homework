import cv2 as cv
import numpy as np

# 1. 이미지 읽기
img = cv.imread('rose.png')

# 2. 이미지 크기 50% 축소
resized = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

# 3. 기하학 변환 (시계 방향 30도 회전)
(h, w) = resized.shape[:2]
center = (w // 2, h // 2)

# 시계 방향은 -각도
M = cv.getRotationMatrix2D(center, -30, 1.0)
rotated = cv.warpAffine(resized, M, (w, h))

# 4. 모폴로지 변환 (예: 침식 + 팽창)
kernel = np.ones((5, 5), np.uint8)

erosion = cv.erode(resized, kernel, iterations=1)
dilation = cv.dilate(resized, kernel, iterations=1)

# 5. 결과 출력
cv.imshow('Original', img)
cv.imshow('Resized (50%)', resized)
cv.imshow('Rotated (-30 deg)', rotated)
cv.imshow('Erosion', erosion)
cv.imshow('Dilation', dilation)

cv.waitKey(0)
cv.destroyAllWindows()

