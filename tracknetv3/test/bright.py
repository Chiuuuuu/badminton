import cv2
from video_path import input_video_path, output_video_path
# 讀取影片
cap = cv2.VideoCapture(input_video_path)

brighter = 40
contrast = 1.5
# 取得影片的幀率和大小
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定輸出影片的編碼和參數
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_brightened_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 增加影片亮度，這裡的 beta 值越高，亮度增加越多
    brightened_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brighter)
    
    # 寫入處理後的幀
    out.write(brightened_frame)
    
    # 顯示幀，方便檢查
    cv2.imshow('Brightened Frame', brightened_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()