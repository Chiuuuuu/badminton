import cv2
import numpy as np
from tqdm.auto import tqdm
from video_path import input_video_path, output_video_path
# 定義座標點
left_line_start = (785, 0)
left_line_end = (273, 1000)
right_line_start = (1156, 0)
right_line_end = (1622, 1000)

# 讀取影片
cap = cv2.VideoCapture(input_video_path)

# 取得影片的幀率和幀大小
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定影片寫入參數
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 創建一個黑色遮罩
    mask = np.zeros_like(frame)

    # 填充左邊線以左的區域
    left_polygon = np.array([[0, 0], [left_line_start[0], left_line_start[1]], [left_line_end[0], left_line_end[1]], [0, height]], dtype=np.int32)
    cv2.fillPoly(mask, [left_polygon], (255, 255, 255))

    # 填充右邊線以右的區域
    right_polygon = np.array([[width, 0], [right_line_start[0], right_line_start[1]], [right_line_end[0], right_line_end[1]], [width, height]], dtype=np.int32)
    cv2.fillPoly(mask, [right_polygon], (255, 255, 255))

    # 創建反轉遮罩來保留中間區域
    mask_inv = cv2.bitwise_not(mask)

    # 應用反轉遮罩來保留中間區域
    result = cv2.bitwise_and(frame, mask_inv)

    # 寫入處理後的幀
    out.write(result)

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
