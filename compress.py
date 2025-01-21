import os
from PIL import Image
import logging
from datetime import datetime

def setup_logging():
    """設置日誌記錄"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'image_compression_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def compress_image(input_path, output_path, max_size_mb=1.0, quality=85):
    """壓縮單一圖片"""
    try:
        # 打開圖片
        with Image.open(input_path) as img:
            # 如果是PNG且有透明通道，轉換為RGB
            if img.format == 'PNG' and img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # 先用原始品質儲存
            img.save(output_path, 'JPEG', quality=quality)
            
            # 檢查檔案大小並進行額外壓縮如果需要的話
            while os.path.getsize(output_path) > (max_size_mb * 1024 * 1024) and quality > 5:
                quality -= 5
                img.save(output_path, 'JPEG', quality=quality)
            
            compression_ratio = os.path.getsize(input_path) / os.path.getsize(output_path)
            logging.info(f'已壓縮 {input_path} (壓縮比: {compression_ratio:.2f}x)')
            
    except Exception as e:
        logging.error(f'壓縮 {input_path} 時發生錯誤: {str(e)}')
        return False
    return True

def process_directory(input_dir, output_dir, max_size_mb=1.0):
    """遞迴處理目錄中的所有圖片"""
    # 建立輸出目錄
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 支援的圖片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 遍歷目錄
    for root, dirs, files in os.walk(input_dir):
        # 計算相對路徑
        rel_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, rel_path)
        
        # 確保輸出目錄存在
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        
        # 處理檔案
        for file in files:
            # 檢查是否為圖片
            if os.path.splitext(file)[1].lower() in image_extensions:
                input_path = os.path.join(root, file)
                output_path = os.path.join(current_output_dir, 
                                         os.path.splitext(file)[0] + '.jpg')
                compress_image(input_path, output_path, max_size_mb)

def main():
    """主程式"""
    setup_logging()
    
    # 設定輸入和輸出目錄
    input_directory = input("請輸入要處理的資料夾路徑: ").strip()
    output_directory = input_directory + "_compressed"
    
    # 設定最大檔案大小（MB）
    max_size = float(input("請輸入壓縮後的最大檔案大小（MB）[預設 1.0]: ") or 1.0)
    
    logging.info(f'開始處理資料夾: {input_directory}')
    logging.info(f'壓縮後檔案將儲存至: {output_directory}')
    
    # 開始處理
    process_directory(input_directory, output_directory, max_size)
    logging.info('處理完成')

if __name__ == "__main__":
    main()