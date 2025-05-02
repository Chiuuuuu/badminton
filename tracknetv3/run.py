import subprocess
import glob
import os
import sys

# 定義參數
conda_env = "tn"  # Conda 環境名稱
python_script = "predict.py"  # Python 腳本名稱
video_dir = "./video/multiview9/"  # 影片所在目錄
tracknet_file = "ckpts/TrackNet_best.pt"  # TrackNet 模型檔案
save_dir = "./prediction/multiview9"  # 儲存結果的目錄
additional_args = "--output_video --large_video"  # 額外參數

# 確保工作目錄為 Python 腳本所在的資料夾
script_dir = os.path.abspath(os.path.dirname(__file__))  # 獲取腳本的目錄
os.chdir(script_dir)

# 列舉目錄中的所有 .mp4 檔案
video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

if not video_files:
    print(f"目錄 {video_dir} 中沒有找到任何 .mp4 檔案！")
else:
    print(f"找到以下 .mp4 檔案：")
    for video in video_files:
        print(video)

    # 對每個 .mp4 檔案執行指令
    for video_file in video_files:
        # 定義完整的 shell 指令
        command = f"""
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate {conda_env} && \
        python {python_script} \
        --video_file {video_file} \
        --tracknet_file {tracknet_file} \
        --save_dir {save_dir} {additional_args}
        """

        print(f"正在處理檔案：{video_file}")
        try:
            # 使用 Popen 即時輸出
            process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, text=True, shell=True, executable="/bin/bash")
            process.communicate()  # 等待完成
            if process.returncode == 0:
                print(f"檔案 {video_file} 執行成功！")
            else:
                print(f"檔案 {video_file} 執行失敗，返回碼：{process.returncode}")
        except Exception as e:
            print(f"執行檔案 {video_file} 時發生錯誤：{e}")
