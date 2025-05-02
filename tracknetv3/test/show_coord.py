import cv2
from video_path import input_video_path, output_video_path

def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img = param.copy()
        cv2.putText(img, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("First Frame", img)

# 讀取影片
cap = cv2.VideoCapture(input_video_path)

if cap.isOpened():
    ret, frame = cap.read()  # 讀取第一個 frame
    if ret:
        cv2.imshow("First Frame", frame)
        cv2.setMouseCallback("First Frame", show_coordinates, frame)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 鍵退出
                break
    else:
        print("無法讀取第一個 frame")
else:
    print("無法開啟影片")

cap.release()
cv2.destroyAllWindows()
