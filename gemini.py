import os
import shutil
import yaml

def process_dataset(dataset_path, output_path, yaml_file, target_mapping):
    """
    處理資料集，提取指定 labels 的圖片，並重新標記類別號碼，過濾不要的標註。

    Args:
        dataset_path: 資料集根目錄路徑。
        output_path: 輸出目錄路徑。
        yaml_file: 定義類別的 YAML 檔案路徑 (僅用於顯示類別名稱)。
        target_mapping: 類別號碼映射字典，例如 {原始ID: 新ID}。
    """

    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            labels_data = yaml.safe_load(f)
            label_names = labels_data.get('names', [])
    except FileNotFoundError:
        print(f"警告：找不到 YAML 檔案：{yaml_file}，將不會顯示類別名稱。")
        label_names = []
    except yaml.YAMLError as exc:
        print(f"警告：解析 YAML 檔案 {yaml_file} 時發生錯誤：{exc}，將不會顯示類別名稱。")
        label_names = []

    for split_name in ["train", "test", "val"]:
        split_path = os.path.join(dataset_path, split_name)
        if not os.path.exists(split_path):
            continue

        labels_dir = os.path.join(split_path, "labels")
        images_dir = os.path.join(split_path, "images")

        if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
            print(f"警告：{split_name} 資料夾缺少 labels 或 images 子資料夾。")
            continue

        for label_file in os.listdir(labels_dir):
            if label_file.endswith(".txt"):
                label_path = os.path.join(labels_dir, label_file)
                annotations = read_annotations(label_path, target_mapping)

                if annotations is None:
                    continue

                if annotations: # 只有當有提取到標註時才複製檔案
                    image_name = label_file[:-4] + ".jpg"
                    image_path = os.path.join(images_dir, image_name)

                    if os.path.exists(image_path):
                        output_split_dir_images = os.path.join(output_path, split_name, "images")
                        output_split_dir_labels = os.path.join(output_path, split_name, "labels")

                        os.makedirs(output_split_dir_images, exist_ok=True)
                        os.makedirs(output_split_dir_labels, exist_ok=True)

                        shutil.copy(image_path, os.path.join(output_split_dir_images, image_name))

                        # 寫入修改後的標註檔案
                        output_label_path = os.path.join(output_split_dir_labels, label_file)
                        with open(output_label_path, 'w') as outfile:
                            for annotation in annotations:
                                bbox_str = " ".join(map(str, annotation['bbox']))
                                outfile.write(f"{annotation['class_id']} {bbox_str}\n")
                                class_name = label_names[annotation['class_id']] if annotation['class_id'] < len(label_names) else f"未知類別({annotation['class_id']})"
                                print(f"已複製 {image_name} (類別：{class_name}) 到 {output_split_dir_images}")
                    else:
                        print(f"找不到對應的圖片：{image_path}")
                else:
                    print(f"檔案 {label_file} 沒有符合條件的標註，不複製。")


def read_annotations(label_file, target_mapping):
    """讀取 TXT 格式的標註檔案，並根據映射修改類別號碼，過濾不要的標註"""
    annotations = []
    try:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = line.split()
                if len(values) < 5:
                    print(f"警告：檔案 {label_file} 中有格式不正確的行：{line}")
                    continue
                try:
                    original_class_id = int(values[0])
                    if original_class_id in target_mapping: # 只保留有在映射表中的類別
                        class_id = target_mapping[original_class_id]
                        bbox = [float(v) for v in values[1:]]
                        annotations.append({'class_id': class_id, 'bbox': bbox})
                except ValueError:
                    print(f"警告：檔案 {label_file} 中有無法轉換為數值的資料：{line}")
                    continue
        return annotations
    except FileNotFoundError:
        print(f"錯誤：找不到檔案：{label_file}")
        return None
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        return None

# 範例使用方式
dataset_path = r"D:\牛奶盒+塑膠+手搖+其他+鋁箔+更多牛奶 還沒5變2"
output_path = "./All"
yaml_file = r"D:\trash\project-all+newmilk\dataset.yaml"
target_mapping = {5:2}  # 將原始類別 0 映射到 10，原始類別 2 映射到 11

process_dataset(dataset_path, output_path, yaml_file, target_mapping)

print("處理完成")