import os

# 定義資料夾路徑
labels_dir = r"C:\Users\TMP214\Downloads\牛奶盒+塑膠+手搖+其他+鋁箔+更多牛奶 還沒5變2\牛奶盒\labels\test"  # 替換為你的資料夾名稱

# 遍歷資料夾內的所有文件
for file_name in os.listdir(labels_dir):
    if file_name.endswith(".txt"):  # 僅處理 .txt 文件
        file_path = os.path.join(labels_dir, file_name)
        
        # 讀取文件內容
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # 修改內容
        with open(file_path, "w") as f:  # 覆蓋寫入
            for line in lines:
                parts = line.strip().split()  # 分割每一行
                if parts[0] == '5':  # 如果第一部分是 '5'
                    parts[0] = '2'  # 替換為 '2'
                f.write(" ".join(parts) + "\n")  # 寫回文件

        print(f"已更新文件: {file_name}")
