import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定義參數
DEVIATION_THRESHOLD = 50  # 偏差篩選閾值
Z_RANGE = (0, 50)  # Z_avg 的得分範圍
FIELD_LIMITS = {'x_min': 0, 'x_max': 5100, 'y_min': 0, 'y_max': 13400}
Y_DIVIDER = 6700  # 場地上下半場分界
SCORE_COOLDOWN_SECONDS = 8  # 計分冷卻時間

def plot_data(processed_df):
    """繪製 avgZ 折線圖"""
    plt.figure(figsize=(10, 6))
    plt.plot(processed_df['Frame'], processed_df['Z_avg'], marker='o')
    plt.title('Filtered Z_avg vs Frame (Outlier Removed)')
    plt.xlabel('Frame')
    plt.ylabel('Z_avg')
    plt.grid()
    output_file = 'z_avg_filtered_outliers_slice.png'
    plt.savefig(output_file, dpi=300)
    plt.show()

def filter_outliers(row, data):
    """過濾每個 Frame 中偏差過大的攝影機數據"""
    x_vals, y_vals, z_vals = [], [], []
    for col in data.columns:
        if '_X' in col:
            cam_name = col.split('_X')[0]
            x, y, z = row[col], row.get(f"{cam_name}_Y"), row.get(f"{cam_name}_Z")
            if pd.notna(x) and pd.notna(y) and pd.notna(z):
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)
    
    # 計算中位數作為基準
    median_x, median_y, median_z = np.median(x_vals), np.median(y_vals), np.median(z_vals)
    
    # 根據偏差篩選出有效數據
    valid_x, valid_y, valid_z = [], [], []
    for x, y, z in zip(x_vals, y_vals, z_vals):
        if abs(x - median_x) <= DEVIATION_THRESHOLD and \
           abs(y - median_y) <= DEVIATION_THRESHOLD and \
           abs(z - median_z) <= DEVIATION_THRESHOLD:
            valid_x.append(x)
            valid_y.append(y)
            valid_z.append(z)
    return valid_x, valid_y, valid_z

def process_data(data):
    """處理數據，計算有效攝影機的平均值"""
    processed_data = []
    for _, row in data.iterrows():
        valid_x, valid_y, valid_z = filter_outliers(row, data)
        if len(valid_x) >= 2:  # 超過兩台有效攝影機
            processed_data.append({
                'Frame': row['Frame'],
                'X_avg': np.mean(valid_x),
                'Y_avg': np.mean(valid_y),
                'Z_avg': np.mean(valid_z),
            })
    return pd.DataFrame(processed_data)

def scoring_with_avgZ(processed_df, video_path):
    """基於 avgZ 進行計分，並在影片中繪製結果，確保每幀顯示數據"""
    cap = cv2.VideoCapture(video_path)
    scoring_frames = []
    score_1, score_2 = 0, 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cooldown_frames = fps * SCORE_COOLDOWN_SECONDS
    last_score_frame = -cooldown_frames  # 確保第一幀可以正常檢查

    # 將處理數據轉換為字典，方便查找
    processed_dict = processed_df.set_index('Frame').to_dict('index')

    frame_id = 0  # 當前處理的影片幀數
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 獲取當前幀對應的數據，若無則使用 NaN 填充
        data = processed_dict.get(frame_id, {'X_avg': np.nan, 'Y_avg': np.nan, 'Z_avg': np.nan})

        # 判斷是否符合得分條件
        if not np.isnan(data['Z_avg']) and Z_RANGE[0] <= data['Z_avg'] <= Z_RANGE[1]:
            if frame_id - last_score_frame > cooldown_frames:
                last_score_frame = frame_id  # 更新最後得分的幀號

                # 判斷得分
                in_field = (FIELD_LIMITS['x_min'] <= data['X_avg'] <= FIELD_LIMITS['x_max'] and
                            FIELD_LIMITS['y_min'] <= data['Y_avg'] <= FIELD_LIMITS['y_max'])

                if data['Y_avg'] > Y_DIVIDER:  # 下半場
                    if in_field:
                        score_2 += 1  # Player 2 得分
                    else:
                        score_1 += 1  # Player 1 得分
                else:  # 上半場
                    if in_field:
                        score_1 += 1  # Player 1 得分
                    else:
                        score_2 += 1  # Player 2 得分

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

        scoring_frames.append(frame)
        frame_id += 1

    cap.release()
    return scoring_frames, score_1, score_2



# 主程序
if __name__ == "__main__":
    file_path = '3d_result.csv'
    video_path = 'video/multiview9/3_Defisheye_0.35.mp4'

    data = pd.read_csv(file_path)
    processed_df = process_data(data)
    plot_data(processed_df)
    scoring_frames, score_1, score_2 = scoring_with_avgZ(processed_df, video_path)

    print(f'Player1 Score: {score_1}, Player2 Score: {score_2}')

    # 儲存處理過的影片
    out = cv2.VideoWriter('scoring_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for frame in scoring_frames:
        out.write(frame)
    out.release()
