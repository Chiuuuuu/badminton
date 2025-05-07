import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

tracknet_results_dir = 'tracknet_results'
calibration_results_dir = 'calibration_results'
three_d_dir = '3d_predictions' # output directory

def compute_projection_matrix(world_points, camera_points):
    A = []
    for i in range(world_points.shape[0]):
        X, Y, Z = world_points[i]
        x, y = camera_points[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
    A = np.array(A)

    # 使用 SVD 解出投影矩陣
    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape(3, 4)
    return P

def triangulate_points(Ps, points):
    """
    傳入多個投影矩陣 Ps 和對應的影像點 points，計算三維點的座標。

    Args:
        Ps (list of np.ndarray): 投影矩陣列表，每個矩陣為 3x4。
        points (list of np.ndarray): 影像點列表，每個點為 (u, v)。
    Returns:
        np.ndarray: 計算得到的三維點的笛卡爾坐標 (X, Y, Z)。
    """
    A = []
    for P, point in zip(Ps, points):
        u, v = point
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)

    # 使用 SVD 求解
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    return X[:3] / X[3]  # 轉換為笛卡爾坐標系

def get_3d_coordinate(main_camera_filename, sub_camera_filename):
    main_camera = main_camera_filename.split('.')[0]
    sub_camera = sub_camera_filename.split('.')[0]

    main_camera_csv = os.path.join(tracknet_results_dir, main_camera_filename)
    sub_camera_csv = os.path.join(tracknet_results_dir, sub_camera_filename)

    main_camera_df = pd.read_csv(main_camera_csv)
    sub_camera_df = pd.read_csv(sub_camera_csv)

    main_camera_calibration_df = pd.read_csv(os.path.join(calibration_results_dir, main_camera_filename))
    sub_camera_calibration_df = pd.read_csv(os.path.join(calibration_results_dir, sub_camera_filename))

    main_camera_screen_points = main_camera_calibration_df[['screen_x', 'screen_y']].values
    sub_camera_screen_points = sub_camera_calibration_df[['screen_x', 'screen_y']].values

    main_camera_world_points = main_camera_calibration_df[['world_x', 'world_y', 'world_z']].values
    sub_camera_world_points = sub_camera_calibration_df[['world_x', 'world_y', 'world_z']].values

    main_camera_P = compute_projection_matrix(main_camera_world_points, main_camera_screen_points)
    sub_camera_P = compute_projection_matrix(sub_camera_world_points, sub_camera_screen_points)

    world_points = []

    # 將 Frame 設為索引
    main_camera_df.set_index('Frame', inplace=True)
    sub_camera_df.set_index('Frame', inplace=True)

    # 篩選 Visibility 為 1 的資料，確保兩個 DataFrame 都有值
    valid_frames = main_camera_df[(main_camera_df['Visibility'] == 1)].index.intersection(
        sub_camera_df[(sub_camera_df['Visibility'] == 1)].index
    )

    # 遍歷篩選後的 Frame
    for frame in tqdm(valid_frames, desc=f'由 {main_camera_filename} 和 {sub_camera_filename} 計算 3D 座標', leave=False):
        main_camera_points = main_camera_df.loc[frame, ['X', 'Y']].values
        sub_camera_points = sub_camera_df.loc[frame, ['X', 'Y']].values

        # 計算 3D 點
        world_point = triangulate_points(
            [main_camera_P, sub_camera_P],
            [main_camera_points, sub_camera_points]
        )
        world_points.append([frame, *world_point])

    # 如果需要，將 world_points 轉為 DataFrame
    world_points_df = pd.DataFrame(world_points, columns=['Frame', f'main_{main_camera}_sub_{sub_camera}_X', f'main_{main_camera}_sub_{sub_camera}_Y', f'main_{main_camera}_sub_{sub_camera}_Z'])

    # 如果需要，將 world_points_df 寫入 CSV 檔案
    # world_points_df.to_csv(f'{three_d_dir}/main_camera_{main_camera}_sub_camera_{sub_camera}.csv', index=False)
    return world_points_df

def merge_3d_coordinate(world_points_dfs):
    merged_df = world_points_dfs[0]
    for i in range(1, len(world_points_dfs)):
        merged_df = pd.merge(merged_df, world_points_dfs[i], on='Frame', how='outer')

    return merged_df


def get_all_3d_coordinates(df_filename='3d_results_all.csv'):
    # os.makedirs(three_d_dir, exist_ok=True)
    tracknet_folder = Path(tracknet_results_dir)
    cameras_filename = [f.name for f in tracknet_folder.iterdir() if f.is_file() and f.suffix == '.csv']
    world_points_dfs = []

    for i in tqdm(range(len(cameras_filename)), desc='計算所有 3D 座標'):
        for j in range(i + 1, len(cameras_filename)):
            world_points_df = get_3d_coordinate(cameras_filename[i], cameras_filename[j])
            world_points_dfs.append(world_points_df)

    merged_3d_coordinate_df = merge_3d_coordinate(world_points_dfs)
    merged_3d_coordinate_df.to_csv(df_filename, index=False)
    print(f'3D 座標計算完成, 輸出至 {df_filename}')

if __name__ == '__main__':
    get_all_3d_coordinates()