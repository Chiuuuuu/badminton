# 羽毛球自動計分

court.txt放的是可以照到找個場地攝影機的球場座標

## 使用說明

### Step 1. 放入 TrackNet 預測結果

對每個不同視角的攝影機執行：
`python predict.py --video_file ./video/new/3.mp4 --tracknet_file ckpts/TrackNet_best.pt --save_dir prediction --output_video --defisheye`
將 TrackNet 的預測結果（csv 檔案）放入 `tracknet_results` 資料夾中
```
tracknet_results/
├── 1.csv
├── 2.csv
└── ...
```


### Step 2. 放入攝影機校正結果


![image](https://hackmd.io/_uploads/SkPnnXckel.png)

校正csv檔中要有多個螢幕坐標系和世界坐標系的點對(至少要6個點以及球網高的那兩個點)

注意這裡的csv檔名要和step 1.的csv檔名一一對應

將攝影機螢幕座標對應世界座標的點對資料（csv 檔案）放入 `calibration_results` 資料夾中。
```
calibration_results/
├── 1.csv
├── 2.csv
└── ...
```


### Step 3. 執行自動計分

執行以下指令，自動判定得分方並可選擇是否輸出影片：

```
python scoring_height_net.py
```

參數選項（scoring_height_net.py）

可透過以下參數客製化輸出內容：

| 參數名稱           | 說明                                      | 預設值                  |
|--------------------|-------------------------------------------|--------------------------|
| `--input`          | 3D 預測資料 CSV 路徑                      | `3d_result_all.csv`      |
| `--inputvid`       | 原始影片路徑                              | `video.mp4`              |
| `--court`          | 球場座標檔路徑                            | `court.txt`              |
| `--outputvid`      | 輸出影片路徑                              | `scoring_result.mp4`     |
| `--output`         | 輸出得分 CSV 路徑                         | `scored.csv`             |
| `--no-video`       | 不輸出影片（加入此 flag 即不輸出）        | `False`                  |
| `--start`          | 設定開始處理的影格                        | `0`        |
| `--end`            | 設定結束處理的影格                        | `112616`                 |
| `--plot`           | 顯示 Y/Z 均值走勢圖                        | `False`                  |
| `--nomedian`       | 不使用中位數濾波                           | `False`      |
| `--noz`            | 不使用 Z 軸濾波                            | `False`      |
| `--noprediction`   | 不使用落點預測功能                         | `False`      |
| `--nonet`          | 不進行網碰偵測（跳過過網高度判定）         | `False`      |
如果要輸出影片可以把能拍到整個場地的攝影機作為inputvid

如果有需要輸出影片，**可以先跑** `get_court.py` **手動標記球場四個角落(左上左下右下右上)**，如此一來計分時會有顏色框顯示，執行方式：

```
python get_court.py
```

參數選項（get_court.py）

| 參數名稱  | 說明                        | 預設值       |
|---------------|---------------------------|---------------|
| `--video`     | 用於操作標記的影片路徑       | `video.mp4`  |
| `--output`    | 儲存球場角落座標的路徑       | `court.txt`  |



### Step 4. 執行評估

使用預測結果與實際標籤比對，執行：

```
python evaluate.py
```

參數選項（evaluate.py）

| 參數名稱        | 說明                           | 預設值                              |
|-------------------|--------------------------------|-----------------------------------|
| `--predicted`     | 預測得分的 CSV 檔案路徑              | `scored.csv`                     |
| `--label`         | 真實標籤的 CSV 檔案路徑               | `label.csv`                      |
| `--output_csv`    | 儲存詳細比對結果的 CSV 檔案路徑        | `evaluate_results/evaluation.csv` |
| `--output_txt`    | 儲存評估指標的文字檔路徑                | `evaluation_metrics/metrics.txt`  |

## 專案結構

```
.
├── tracknet_results/        # TrackNet 預測結果放置位置
├── calibration_results/     # 攝影機校正結果放置位置
├── three_d_coordinate.py    # 3D 座標重建程式
├── scoring_height_net.py    # 自動計分程式
```


