import os
from PIL import Image
import pillow_heif
from tqdm import tqdm
import shutil
from pathlib import Path

def setup_folders(新分類: str, 超新分類: str):
    """設置輸入和輸出資料夾"""
    # 創建輸出資料夾(如果不存在)
    os.makedirs(新分類, exist_ok=True)
    
    # 確認輸入資料夾存在
    if not os.path.exists(新分類):
        raise FileNotFoundError(f"輸入資料夾 '{新分類:}' 不存在")

def convert_heic_to_jpeg(input_file: str, output_file: str, quality: int = 95):
    """將單個HEIC檔案轉換為JPEG"""
    try:
        # 讀取HEIC圖片
        image = Image.open(input_file)
        
        # 轉換為RGB(以防是RGBA格式)
        rgb_image = image.convert('RGB')
        
        # 儲存為JPEG
        rgb_image.save(output_file, 'JPEG', quality=quality)
        return True
    except Exception as e:
        print(f"轉換檔案 '{input_file}' 時發生錯誤: {str(e)}")
        return False

def batch_convert(input_folder: str, output_folder: str, quality: int = 95):
    """批次轉換資料夾中的所有HEIC檔案"""
    # 註冊HEIF檔案支援
    pillow_heif.register_heif_opener()
    
    try:
        # 設置資料夾
        setup_folders(input_folder, output_folder)
        
        # 取得所有HEIC檔案
        heic_files = [f for f in os.listdir(input_folder) 
                     if f.lower().endswith(('.heic', '.heif'))]
        
        if not heic_files:
            print(f"在 '{input_folder}' 中沒有找到HEIC檔案")
            return
        
        # 顯示進度條
        print(f"開始轉換 {len(heic_files)} 個檔案...")
        successful = 0
        failed = 0
        
        for filename in tqdm(heic_files, desc="轉換進度"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(
                output_folder, 
                Path(filename).stem + '.jpg'
            )
            
            if convert_heic_to_jpeg(input_path, output_path, quality):
                successful += 1
            else:
                failed += 1
        
        # 顯示結果
        print("\n轉換完成!")
        print(f"成功: {successful} 個檔案")
        print(f"失敗: {failed} 個檔案")
        
    except Exception as e:
        print(f"程式執行時發生錯誤: {str(e)}")

if __name__ == "__main__":
    # 設定參數
    INPUT_FOLDER = "heic_files"  # 輸入資料夾
    OUTPUT_FOLDER = "jpeg_files" # 輸出資料夾
    JPEG_QUALITY = 95           # JPEG品質(1-100)
    
    # 執行轉換
    batch_convert(INPUT_FOLDER, OUTPUT_FOLDER, JPEG_QUALITY)