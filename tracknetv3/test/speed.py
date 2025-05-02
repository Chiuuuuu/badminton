import matplotlib.pyplot as plt
import pandas as pd

# 讀取使用者上傳的 speed.csv 檔案
file_path = './speed.csv'
speed_data = pd.read_csv(file_path)

# 顯示資料的前幾列來確認內容
speed_data.head()
# 假設索引即為 frame number，將資料整理為 dataframe
speed_data.columns = ['speed']
speed_data['frame number'] = speed_data.index

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.plot(speed_data['frame number'], speed_data['speed'], marker='o')
plt.title('Speed vs Frame Number')
plt.xlabel('Frame Number')
plt.ylabel('Speed')
plt.grid(True)
plt.show()