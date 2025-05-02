import cv2
import numpy as np
from video_path import input_video_path
# 載入MP4文件並提取第一幀
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, img = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

cap.release()

# 定義原始圖像中場地的四個角點 (這些點需要根據實際情況調整)
pts1 = np.float32([[782, 175], [1151, 184],[1465, 984], [427, 957]])


# 定義變換後圖像中場地的四個角點
pts2 = np.float32([[782, 175], [1151, 184],[1465, 984], [427, 957]])
# 計算透視變換矩陣
M = cv2.getPerspectiveTransform(pts1, pts2)

# 進行透視變換
dst = cv2.warpPerspective(img, M, (800, 600))

# 顯示原始圖像和變換後的圖像
cv2.imshow('Original', img)
cv2.imshow('Corrected', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

