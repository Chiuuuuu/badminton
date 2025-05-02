import cv2
import numpy as np
from tqdm.auto import tqdm


def get_image_midpoints(width, height):
    return (width // 2, 0), (width // 2, height - 1), (0, height // 2), (width - 1, height // 2)

def precompute_maps(h, w, alpha, border_size):
    new_h = h + 2 * border_size
    new_w = w + 2 * border_size
    
    map_x, map_y = np.meshgrid(np.arange(new_w), np.arange(new_h))
    
    center_x = new_w / 2
    center_y = new_h / 2
    
    radius = np.sqrt((map_x - center_x) ** 2 + (map_y - center_y) ** 2)
    max_radius = np.max(radius)
    
    map_x_new = map_x + alpha * (map_x - center_x) * (radius / max_radius)
    map_y_new = map_y + alpha * (map_y - center_y) * (radius / max_radius)
    
    return map_x_new.astype(np.float32), map_y_new.astype(np.float32), radius, max_radius, center_x, center_y

def defisheye_coordinates(coord, radius, max_radius, center_x, center_y, alpha, border_size):
    x, y = coord
    fisheye_x = x + border_size
    fisheye_y = y + border_size
    
    defisheye_x = center_x + (fisheye_x - center_x) / (1 + alpha * (radius[fisheye_y, fisheye_x] / max_radius))
    defisheye_y = center_y + (fisheye_y - center_y) / (1 + alpha * (radius[fisheye_y, fisheye_x] / max_radius))
    
    return defisheye_x, defisheye_y

def defisheye(img, map_x, map_y, border_size):
    h, w = img.shape[:2]
    new_h = h + 2 * border_size
    new_w = w + 2 * border_size

    padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    padded_image[border_size:border_size+h, border_size:border_size+w, :] = img

    corrected_image = cv2.remap(padded_image, map_x, map_y, cv2.INTER_LINEAR)
    
    return corrected_image

def crop_image_by_points(img, top_point=None, bottom_point=None, left_point=None, right_point=None):
    h, w = img.shape[:2]

    top_y = 0 if top_point is None else max(0, int(top_point[1]))
    bottom_y = h if bottom_point is None else min(h, int(bottom_point[1]))
    left_x = 0 if left_point is None else max(0, int(left_point[0]))
    right_x = w if right_point is None else min(w, int(right_point[0]))

    return img[top_y:bottom_y, left_x:right_x]

def process_video(input_path, output_path, alpha, border_size):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 预计算映射
    map_x, map_y, radius, max_radius, center_x, center_y = precompute_maps(height, width, alpha, border_size)

    # 预计算中点坐标
    m_top, m_bot, _, _ = get_image_midpoints(width, height)
    
    # 预计算校正后的中点坐标
    dfe_m_top = defisheye_coordinates(m_top, radius, max_radius, center_x, center_y, alpha, border_size)
    dfe_m_bot = defisheye_coordinates(m_bot, radius, max_radius, center_x, center_y, alpha, border_size)

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        dfe_frame = defisheye(frame, map_x, map_y, border_size)
        cut_dfe_frame = crop_image_by_points(img=dfe_frame, top_point=dfe_m_top, bottom_point=dfe_m_bot)
        processed_frame = cv2.resize(cut_dfe_frame,  (width, height)) #WARNING: The frame size must be the same size as the video stream.
        out.write(processed_frame)

    cap.release()
    out.release()

def generate_defisheye_frames(video_file, alpha=-0.33, border_size=200):
    """ Sample and defisheye frames from the video.

        Args:
            video_file (str): The file path of the video file
            alpha (float): The defisheye distortion parameter
            border_size (int): The size of the border added for defisheye correction

        Returns:
            frame_list (List[numpy.ndarray]): A list of defisheye-corrected frames
            fps (int): The frame rate of the video
            (w, h) (Tuple[int, int]): The width and height of the video
    """

    assert video_file[-4:] == '.mp4', 'Invalid video file format.'

    # Get video properties
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_list = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count

    # Precompute the maps for defisheye correction
    map_x, map_y, radius, max_radius, center_x, center_y = precompute_maps(h, w, alpha, border_size)

    # Precompute midpoints
    m_top, m_bot, _, _ = get_image_midpoints(w, h)
    
    # Precompute corrected midpoints
    dfe_m_top = defisheye_coordinates(m_top, radius, max_radius, center_x, center_y, alpha, border_size)
    dfe_m_bot = defisheye_coordinates(m_bot, radius, max_radius, center_x, center_y, alpha, border_size)

    # Sample frames until the end of the video
    for _ in tqdm(range(total_frames), desc="Reading frames"):
        success, frame = cap.read()
        if success:
            dfe_frame = defisheye(frame, map_x, map_y, border_size)
            cut_dfe_frame = crop_image_by_points(img=dfe_frame, top_point=dfe_m_top, bottom_point=dfe_m_bot)
            processed_frame = cv2.resize(cut_dfe_frame, (w, h))  # WARNING: The frame size must match the video stream size.
            frame_list.append(processed_frame)
            
    cap.release()  # Ensure video resource is released
    return frame_list, fps, (w, h)
