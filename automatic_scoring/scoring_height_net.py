import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from three_d_coordinate import get_all_3d_coordinates
import argparse

# 定義參數
DEVIATION_THRESHOLD = 100  # 偏差篩選閾值
Z_RANGE = (0, 300)  # Z_avg 的得分範圍
FIELD_LIMITS = {'x_min': 460, 'x_max': 5640, 'y_min': 0, 'y_max': 13400}
Y_DIVIDER = 6700  # 場地上下半場分界
SCORE_COOLDOWN_SECONDS = 8  # 計分冷卻時間
previous_frames = [] # 保存三幀以補幀
fps = 50

def plot_data(processed_df, start_frame, end_frame):
    """繪製 Y_avg 和 Z_avg 在同一張圖上，使用雙座標軸"""
    filtered_df = processed_df[(processed_df['Frame'] >= start_frame) & (processed_df['Frame'] <= end_frame)]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # 建立第二個座標軸

    ax1.plot(filtered_df['Frame'], filtered_df['Y_avg'], marker='o', label='Y_avg', color='b')
    ax2.plot(filtered_df['Frame'], filtered_df['Z_avg'], marker='s', label='Z_avg', color='r')

    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Y_avg', color='b')
    ax2.set_ylabel('Z_avg', color='r')

    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f'Filtered Y_avg and Z_avg vs Frame ({start_frame} - {end_frame})')
    fig.tight_layout()
    plt.grid()
    plt.savefig('yz_avg_slice.png', dpi=300)
    plt.show()

def filter_outliers(row, data, median, Z):
    """過濾每個 Frame 中偏差過大的攝影機數據"""
    x_vals, y_vals, z_vals = [], [], []

    for col in data.columns:
        if '_X' in col:
            cam_name = col.split('_X')[0]
            x, y, z = row[col], row.get(f"{cam_name}_Y"), row.get(f"{cam_name}_Z")

            if Z:
                if pd.notna(x) and pd.notna(y) and pd.notna(z) and 0 <= z <= 4000: #新增Z範圍限制
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
            elif pd.notna(x) and pd.notna(y) and pd.notna(z):
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)

    # 計算中位數作為基準
    median_x, median_y, median_z = np.median(x_vals), np.median(y_vals), np.median(z_vals)

    # 根據偏差篩選出有效數據
    valid_x, valid_y, valid_z = [], [], []
    if median:
        for x, y, z in zip(x_vals, y_vals, z_vals):
            if abs(x - median_x) <= DEVIATION_THRESHOLD and \
            abs(y - median_y) <= DEVIATION_THRESHOLD and \
            abs(z - median_z) <= DEVIATION_THRESHOLD:
                valid_x.append(x)
                valid_y.append(y)
                valid_z.append(z)
    else:
        for x, y, z in zip(x_vals, y_vals, z_vals):
            valid_x.append(x)
            valid_y.append(y)
            valid_z.append(z)

    return valid_x, valid_y, valid_z

def add_to_previous_frames(x, y, z, frame_number):
    """將當前幀的數據添加到 previous_frames 中，並確保只保留最新的三幀"""
    # 當前幀的數據
    frame_data = {'X': x, 'Y': y, 'Z': z, 'Frame': frame_number}

    # 若已經有三個幀，刪除最舊的幀
    if len(previous_frames) >= 3:
        previous_frames.pop(0)

    # 添加當前幀的數據
    previous_frames.append(frame_data)

def predict_position(frame_number):
    """基於已知速度和加速度預測下一幀的位置"""
    # 計算速度
    vx1 = (previous_frames[1]['X'] - previous_frames[0]['X']) / (previous_frames[1]['Frame'] - previous_frames[0]['Frame'])
    vx2 = (previous_frames[2]['X'] - previous_frames[1]['X']) / (previous_frames[2]['Frame'] - previous_frames[1]['Frame'])
    vy1 = (previous_frames[1]['Y'] - previous_frames[0]['Y']) / (previous_frames[1]['Frame'] - previous_frames[0]['Frame'])
    vy2 = (previous_frames[2]['Y'] - previous_frames[1]['Y']) / (previous_frames[2]['Frame'] - previous_frames[1]['Frame'])
    vz1 = (previous_frames[1]['Z'] - previous_frames[0]['Z']) / (previous_frames[1]['Frame'] - previous_frames[0]['Frame'])
    vz2 = (previous_frames[2]['Z'] - previous_frames[1]['Z']) / (previous_frames[2]['Frame'] - previous_frames[1]['Frame'])

    # 計算加速度
    ax = (vx2 - vx1) / (previous_frames[2]['Frame'] - previous_frames[0]['Frame'])
    ay = (vy2 - vy1) / (previous_frames[2]['Frame'] - previous_frames[0]['Frame'])
    az = (vz2 - vz1) / (previous_frames[2]['Frame'] - previous_frames[0]['Frame'])

    delta_t = frame_number - previous_frames[2]['Frame']

    x_pred = previous_frames[2]['X'] + vx2 * delta_t + 0.5 * ax * delta_t ** 2
    y_pred = previous_frames[2]['Y'] + vy2 * delta_t + 0.5 * ay * delta_t ** 2
    z_pred = previous_frames[2]['Z'] + vz2 * delta_t + 0.5 * az * delta_t ** 2
    return x_pred, y_pred, z_pred

def process_data(data, start_frame, end_frame, median, z, prediction):
    """處理數據，計算有效攝影機的平均值"""
    processed_data = []

    for frame_number in tqdm(range(start_frame, end_frame + 1), desc='處理數據中'):
        row = data[data['Frame'] == frame_number]

        if not row.empty:
            row = row.iloc[0]
            valid_x, valid_y, valid_z = filter_outliers(row, data, median, z)

            if len(valid_x) >= 2 or len(previous_frames) < 3:  # 超過兩組有效攝影機
                processed_data.append({
                    'Frame': row['Frame'],
                    'X_avg': np.mean(valid_x),
                    'Y_avg': np.mean(valid_y),
                    'Z_avg': np.mean(valid_z),
                })
                add_to_previous_frames(np.mean(valid_x), np.mean(valid_y), np.mean(valid_z), row['Frame'])
            elif len(valid_x) == 1 and prediction:
                # 如果只有一組相機則與預測值比較
                x_pred, y_pred, z_pred = predict_position(row['Frame'])
                if abs(x_pred - valid_x[0]) <= DEVIATION_THRESHOLD and \
                    abs(y_pred - valid_y[0]) <= DEVIATION_THRESHOLD and \
                    abs(z_pred - valid_z[0]) <= DEVIATION_THRESHOLD:
                    processed_data.append({
                        'Frame': row['Frame'],
                        'X_avg': valid_x[0],
                        'Y_avg': valid_y[0],
                        'Z_avg': valid_z[0],
                    })
                    add_to_previous_frames(valid_x[0], valid_y[0], valid_z[0], row['Frame'])
                elif frame_number - previous_frames[0]['Frame'] <= 15:
                    if z_pred < 0: z_pred = 1
                    processed_data.append({
                        'Frame': row['Frame'],
                        'X_avg': x_pred,
                        'Y_avg': y_pred,
                        'Z_avg': z_pred,
                    })
            elif frame_number - previous_frames[0]['Frame'] <= 15 and prediction:
                # 若該幀的數據無效，則進行預測
                x_pred, y_pred, z_pred = predict_position(row['Frame'])
                if z_pred < 0: z_pred = 1
                processed_data.append({
                    'Frame': row['Frame'],
                    'X_avg': x_pred,
                    'Y_avg': y_pred,
                    'Z_avg': z_pred,
                })
        elif len(previous_frames) >= 3 and frame_number - previous_frames[0]['Frame'] <= 15 and prediction:
            # 如果該幀數據不存在，則進行預測，超過太久不預測
            x_pred, y_pred, z_pred = predict_position(frame_number)
            if z_pred < 0: z_pred = 1
            processed_data.append({
                'Frame': frame_number,
                'X_avg': x_pred,
                'Y_avg': y_pred,
                'Z_avg': z_pred,
            })
    return pd.DataFrame(processed_data)

def load_coordinates_from_file(filename):
    """
    Load and convert coordinate data from a text file into a NumPy array in clockwise order.

    Args:
        filename (str): The path to the text file containing the coordinates.

    Returns:
        numpy.ndarray: A 2D NumPy array where each row is a pair of integers representing a coordinate in clockwise order.
    """
    # Read the entire content of the file
    with open(filename, 'r') as file:
        data_string = file.read().strip()

    # Split the data by new lines to separate each coordinate
    lines = data_string.splitlines()

    # Split each coordinate by ';' and convert to integers
    coordinates = [tuple(map(int, line.split(';'))) for line in lines]

    # Convert the list of tuples into a 2D NumPy array
    numpy_array = np.array(coordinates)

    # Reorder coordinates to be in clockwise order: left-top, right-top, right-bottom, left-bottom
    # Assuming the input order is: left-top, left-bottom, right-bottom, right-top
    clockwise_order = [0, 3, 2, 1]
    numpy_array = numpy_array[clockwise_order]

    return numpy_array

def get_half_court_coordinates(filename):
    # Load court corner coordinates
    court_corners = load_coordinates_from_file(filename)

    court_corners = court_corners.astype(np.float32)

    # Ideal court coordinates (assuming width is 610 and height is 1340)
    ideal_court = np.array([[0, 0], [610, 0], [610, 1340], [0, 1340]], dtype=np.float32)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(court_corners, ideal_court)

    # Ideal half-court coordinates
    upper_half = np.array([[46, 0], [564, 0], [564, 670], [46, 670]], dtype=np.float32)
    lower_half = np.array([[46, 670], [564, 670], [564, 1340], [46, 1340]], dtype=np.float32)

    # Transform the half-court coordinates back to the original image coordinates
    upper_half_transformed = cv2.perspectiveTransform(upper_half[np.newaxis], np.linalg.inv(matrix))[0]
    lower_half_transformed = cv2.perspectiveTransform(lower_half[np.newaxis], np.linalg.inv(matrix))[0]

    #WARNING: the coords must be int32
    upper_half_transformed = upper_half_transformed.astype(np.int32)
    lower_half_transformed = lower_half_transformed.astype(np.int32)

    # Return the coordinates in clockwise order
    return upper_half_transformed, lower_half_transformed

def scoring_sequential(video_path, processed_df, output_path, start_frame, end_frame, court_txt, net):
    """基於 avgZ 進行計分，逐幀讀取影片並同步處理數據"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cooldown_frames = fps * SCORE_COOLDOWN_SECONDS
    last_score_frame = -cooldown_frames  # 確保第一幀可以正常檢查
    upper_half, lower_half = get_half_court_coordinates(court_txt)
    score_flag = -1
    draw_counter = 0
    low_hight_counter = 0
    lowframe_start = 0
    bump_flag = 0

    # 初始化分數
    score_1 = 0
    score_2 = 0

    lowest_x = 0
    lowest_y = 0
    lowest_z = 0

    #儲存通過網子前後的yz
    y_prev = 0
    y_current = 0
    z_prev = 0
    z_current = 0
    court = -1
    net_frame = 0

    processed_dict = processed_df.set_index('Frame').to_dict('index')

    # 跳到起始幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame  # 當前處理的影片幀數

    # 即時寫入影片
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    pbar = tqdm(total=end_frame - start_frame, desc='處理影片中', unit='幀')
    while frame_id < end_frame:  # 限制處理範圍
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break  # 到達影片末尾

        # 獲取當前幀對應的數據，若無則使用 NaN 填充
        data = processed_dict.get(frame_id, {'X_avg': np.nan, 'Y_avg': np.nan, 'Z_avg': np.nan})

        if not np.isnan(data['Y_avg']) and abs(data['Y_avg'] - 6700) < 1000 and frame_id - net_frame > 100:
            y_prev, z_prev = y_current, z_current
            y_current, z_current = data['Y_avg'], data['Z_avg']

            if y_prev and (y_prev > 6700 and y_current < 6700) or (y_prev < 6700 and y_current > 6700):
                z_interp = z_prev + ((6700 - y_prev) / (y_current - y_prev)) * (z_current - z_prev)

                if z_interp <= 1580:  # 1550 代表網子高度
                    net_frame = frame_id  # 記錄掛網發生的幀數
                    court = 1 if y_prev > Y_DIVIDER else 0 # 超過6700在上半場
                    # print("*", court, frame_id, y_prev, y_current)
                    y_prev = 0
                    y_current = 0

        # 判斷是否符合得分條件
        if not np.isnan(data['Z_avg']) and Z_RANGE[0] <= data['Z_avg'] <= Z_RANGE[1]:
            if low_hight_counter == 0:
                lowest_x = data['X_avg']
                lowest_y = data['Y_avg']
                lowframe_start = frame_id
            elif lowest_z > data['Z_avg'] and bump_flag == 0:
                lowest_x = data['X_avg']
                lowest_y = data['Y_avg']
            else:
                bump_flag = 1

            low_hight_counter += 1

            if frame_id - last_score_frame > cooldown_frames and low_hight_counter >= 10:
                last_score_frame = frame_id  # 更新最後得分的幀號

                # 判斷得分
                in_field = (FIELD_LIMITS['x_min'] <= lowest_x <= FIELD_LIMITS['x_max'] and
                            FIELD_LIMITS['y_min'] <= lowest_y <= FIELD_LIMITS['y_max'])

                # print(lowest_x, lowest_y, frame_id)
                # print(court, net_frame, frame_id - net_frame)

                if frame_id - net_frame < 100 and net: #判斷掛網
                    if court == 0:
                        score_1 += 1
                        score_flag = 5
                    elif court == 1:
                        score_2 += 1
                        score_flag = 4
                elif lowest_y > Y_DIVIDER:  # 下半場
                    if in_field:
                        score_2 += 1  # Player 2 得分
                        score_flag = 0
                    else:
                        score_1 += 1  # Player 1 得分
                        score_flag = 1
                else:  # 上半場
                    if in_field:
                        score_1 += 1  # Player 1 得分
                        score_flag = 2
                    else:
                        score_2 += 1  # Player 2 得分
                        score_flag = 3
        elif frame_id - lowframe_start > 30:
            low_hight_counter = 0
            bump_flag = 0

        if score_flag != -1:
            if draw_counter < 10:
                draw_counter += 1
                if score_flag == 0:
                    cv2.polylines(frame, [upper_half], isClosed=True, color=(0, 255, 0), thickness=7)  # Green polygon
                elif score_flag == 1:
                    cv2.polylines(frame, [upper_half], isClosed=True, color=(0, 0, 255), thickness=7)  # Red polygon
                elif score_flag == 2:
                    cv2.polylines(frame, [lower_half], isClosed=True, color=(0, 255, 0), thickness=7)  # Green polygon
                elif score_flag == 3:
                    cv2.polylines(frame, [lower_half], isClosed=True, color=(0, 0, 255), thickness=7)  # Red polygon
                elif score_flag == 4:
                    cv2.polylines(frame, [upper_half], isClosed=True, color=(255, 0, 0), thickness=7)  # Blue polygon
                elif score_flag == 5:
                    cv2.polylines(frame, [lower_half], isClosed=True, color=(255, 0, 0), thickness=7)  # Blue polygon
            else:
                draw_counter = 0
                score_flag = -1

        # 在當前畫面上繪製分數和座標數據
        cv2.putText(frame, f'Player1: {score_1}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Player2: {score_2}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Frame: {frame_id}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        cv2.putText(frame, f'X_avg: {data["X_avg"]:.2f}' if not np.isnan(data["X_avg"]) else 'X_avg: N/A',
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        cv2.putText(frame, f'Y_avg: {data["Y_avg"]:.2f}' if not np.isnan(data["Y_avg"]) else 'Y_avg: N/A',
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        cv2.putText(frame, f'Z_avg: {data["Z_avg"]:.2f}' if not np.isnan(data["Z_avg"]) else 'Z_avg: N/A',
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    return score_1, score_2


def scoring_sequential_novid(processed_df, start_frame, end_frame, net):
    """基於 avgZ 進行計分，逐幀讀取影片並同步處理數據"""
    cooldown_frames = fps * SCORE_COOLDOWN_SECONDS
    last_score_frame = -cooldown_frames  # 確保第一幀可以正常檢查
    upper_half, lower_half = get_half_court_coordinates("court.txt")
    score_flag = -1
    draw_counter = 0
    low_hight_counter = 0
    lowframe_start = 0
    bump_flag = 0

    scored_list = []

    # 初始化分數
    score_1 = 0
    score_2 = 0

    lowest_x = 0
    lowest_y = 0
    lowest_z = 0

    #儲存通過網子前後的yz
    y_prev = 0
    y_current = 0
    z_prev = 0
    z_current = 0
    court = -1
    net_frame = 0

    processed_dict = processed_df.set_index('Frame').to_dict('index')

    # 跳到起始幀
    frame_id = start_frame  # 當前處理的影片幀數

    # 即時寫入影片

    pbar = tqdm(total=end_frame - start_frame, desc='計分處理中', unit='幀')
    while frame_id < end_frame:  # 限制處理範圍
        scorer = None
        pbar.update(1)

        # 獲取當前幀對應的數據，若無則使用 NaN 填充
        data = processed_dict.get(frame_id, {'X_avg': np.nan, 'Y_avg': np.nan, 'Z_avg': np.nan})

        if not np.isnan(data['Y_avg']) and abs(data['Y_avg'] - 6700) < 1000 and frame_id - net_frame > 100:
            y_prev, z_prev = y_current, z_current
            y_current, z_current = data['Y_avg'], data['Z_avg']

            if y_prev and (y_prev > 6700 and y_current < 6700) or (y_prev < 6700 and y_current > 6700):
                z_interp = z_prev + ((6700 - y_prev) / (y_current - y_prev)) * (z_current - z_prev)

                if z_interp <= 1580:  # 1550 代表網子高度
                    net_frame = frame_id  # 記錄掛網發生的幀數
                    court = 1 if y_prev > Y_DIVIDER else 0 # 超過6700在上半場
                    # print("*", court, frame_id, y_prev, y_current)
                    y_prev = 0
                    y_current = 0

        # 判斷是否符合得分條件
        if not np.isnan(data['Z_avg']) and Z_RANGE[0] <= data['Z_avg'] <= Z_RANGE[1]:
            if low_hight_counter == 0:
                lowest_x = data['X_avg']
                lowest_y = data['Y_avg']
                lowframe_start = frame_id
            elif lowest_z > data['Z_avg'] and bump_flag == 0:
                lowest_x = data['X_avg']
                lowest_y = data['Y_avg']
            else:
                bump_flag = 1

            low_hight_counter += 1

            if frame_id - last_score_frame > cooldown_frames and low_hight_counter >= 10:
                last_score_frame = frame_id  # 更新最後得分的幀號

                # 判斷得分
                in_field = (FIELD_LIMITS['x_min'] <= lowest_x <= FIELD_LIMITS['x_max'] and
                            FIELD_LIMITS['y_min'] <= lowest_y <= FIELD_LIMITS['y_max'])

                # print(lowest_x, lowest_y, frame_id)
                # print(court, net_frame, frame_id - net_frame)

                if frame_id - net_frame < 100 and net: #判斷掛網
                    if court == 0:
                        score_1 += 1
                        score_flag = 5
                        scorer = 1
                    elif court == 1:
                        score_2 += 1
                        score_flag = 4
                        scorer = 2
                elif lowest_y > Y_DIVIDER:  # 下半場
                    if in_field:
                        score_2 += 1  # Player 2 得分
                        score_flag = 0
                        scorer = 2
                    else:
                        score_1 += 1  # Player 1 得分
                        score_flag = 1
                        scorer = 1
                else:  # 上半場
                    if in_field:
                        score_1 += 1  # Player 1 得分
                        score_flag = 2
                        scorer = 1
                    else:
                        score_2 += 1  # Player 2 得分
                        score_flag = 3
                        scorer = 2
        elif frame_id - lowframe_start > 30:
            low_hight_counter = 0
            bump_flag = 0

        # 在當前畫面上繪製分數和座標數據
        frame_id += 1
        if scorer is not None:
            time = frame_id / fps
            time = pd.to_datetime(time, unit='s').strftime('%H:%M:%S')
            scored_list.append([time, frame_id, score_1, score_2, scorer, score_flag])

    scored_df = pd.DataFrame(scored_list, columns=['Time', 'Frame', 'Score_1', 'Score_2', 'Scorer', 'Score_Flag'])
    return scored_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='3d_result_all.csv', help='file path of the TrackNet result')
    parser.add_argument('--inputvid', type=str, default='video.mp4', help='file path of the input video')
    parser.add_argument('--court', type=str, default='court.txt', help='file path of the court')
    parser.add_argument('--outputvid', type=str, default='scoring_result.mp4', help='file path of the output video')
    parser.add_argument('--output', type=str, default='scored.csv', help='output scored csv')
    parser.add_argument('--no-video', action='store_true', default=False, help='whether to output video')
    parser.add_argument('--start', type=int, default=0, help='starting frame')
    parser.add_argument('--end', type=int, default=112616, help='ending frame')
    parser.add_argument('--plot', action='store_true', default=False, help='Whether to plot Y_avg and Z_avg')

    parser.add_argument('--nomedian', action='store_true', default=False, help='whether to use median filter')
    parser.add_argument('--noz', action='store_true', default=False, help='whether to use Z filter')
    parser.add_argument('--noprediction', action='store_true', default=False, help='whether to use prediction')
    parser.add_argument('--nonet', action='store_true', default=False, help='whether to use net fault detection')
    args = parser.parse_args()

    file_path = args.input
    if not os.path.exists(file_path):
       get_all_3d_coordinates(df_filename=file_path)

    previous_frames = []

    data = pd.read_csv(file_path)

    print(not args.nomedian, not args.noz, not args.noprediction, not args.nonet)
    #if not os.path.exists('processed.csv'):
    processed_df = process_data(data, args.start, args.end, not args.nomedian, not args.noz, not args.noprediction)
    #processed_df.to_csv('processed.csv', index=False)
    #else:
    #    processed_df = pd.read_csv('processed.csv')

    if args.no_video:
        scored_df = scoring_sequential_novid(processed_df, args.start, args.end, not args.nonet)
    else:
        scored_df = scoring_sequential(args.inputvid, processed_df, args.outputvid, args.start, args.end, args.court, not args.nonet)
        print(f"Video saved to: {args.outputvid}")

    scored_df.to_csv(args.output, index=False)
    print(f"Scoring results saved to: {args.output}")

    # scoring_sequential(video_path, processed_df, output_path, start_frame, end_frame)
    # print('影片處理完成, 輸出路徑:', output_path)
    if args.plot:
        plot_data(processed_df, args.start, args.end)
