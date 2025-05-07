import cv2
import argparse

coordinates = []

def click_event(event, x, y, flags, param):
    global coordinates, image, output_path

    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        print(f"({x}, {y})")

        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Court Image', image)

        if len(coordinates) == 4:
            save_coordinates(output_path)
            print(f"Court coordinates saved to {output_path}")
            cv2.destroyAllWindows()

def save_coordinates(output_path):
    """將四個角點儲存到 output_path"""
    with open(output_path, 'w') as file:
        for coord in coordinates:
            file.write(f"{coord[0]};{coord[1]}\n")

def main():
    global image, output_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='video.mp4', help='Path to the input video')
    parser.add_argument('--output', type=str, default='court.txt', help='Path to save the court coordinates')
    args = parser.parse_args()

    output_path = args.output

    # 讀取影片第一幀作為標記圖片
    cap = cv2.VideoCapture(args.video)

    ret, image = cap.read()
    cap.release()

    cv2.imshow('Court Image', image)
    cv2.setMouseCallback('Court Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()