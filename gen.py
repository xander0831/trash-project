import os
import random
import re
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import argparse

def get_device():
    # 檢查 GPU 是否可用，使用 PyTorch
    if torch.cuda.is_available():
        print("Using GPU for computation.")
        return "cuda"  # PyTorch 的 GPU 裝置
    else:
        print("No GPU found, falling back to CPU.")
        return "cpu"  # PyTorch 的 CPU 裝置

# 動態參數設定
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate images of common trash in Taiwan using Stable Diffusion.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for dataset and output.")
    parser.add_argument("--huggingface_token", type=str, required=True, help="Hugging Face access token.")
    return parser.parse_args()

def get_dataset_image_size(dataset_dir):
    sizes = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    sizes.append(img.size)  # (width, height)
    if sizes:
        # 統計圖片尺寸有哪些
        most_common_size = max(set(sizes), key=sizes.count)
        print(f"Most common image size in dataset: {most_common_size}")
        return most_common_size
    else:
        raise ValueError("No valid image files found in dataset!")


# 獲得資料夾中現有圖片的最大編號
def get_start_index(output_dir, category):
    files = os.listdir(output_dir)
    indices = []
    for f in files:
        if f.startswith(category) and f.endswith(".jpg"):
            match = re.search(r"_(\\d+)", f)  # 檢測尋找數字部分
            if match:
                indices.append(int(match.group(1)))
    return max(indices, default=0) + 1  # 如果沒有檔案則從 1 開始


# 生成圖片並且調整大小
def generate_images(pipe, category, num_images, prompts, output_dir, inspection_dir, target_size, device):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(inspection_dir, exist_ok=True)
    start_index = get_start_index(inspection_dir, category)  # 獲取起始編號
    print(f"Generating {category} images starting from index {start_index}...")
    for i in range(num_images):
        # 隨機選擇一個 prompt
        prompt = random.choice(prompts)
        with torch.no_grad():  # 禁用梯度計算，提升推理效能
            pipe.to(device)  # 將管線模型移到裝置
            image = pipe(prompt).images[0]
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        # 保存圖片到 generated_inspection 目錄
        current_index = start_index + i
        inspection_path = os.path.join(inspection_dir, f"{category}_{current_index:04d}.jpg")
        image.save(inspection_path)
        print(f"Saved inspection image: {inspection_path}, size: {image.size}")

# 主程式
def main():
    args = parse_arguments()
    base_dir = args.base_dir
    dataset_dir = os.path.join(base_dir, "data/taiwan-trash")
    output_base_dir = os.path.join(base_dir, "data/output")
    inspection_base_dir = os.path.join(base_dir, "data/generated_inspection")

    categories = {
        "衛生紙": 100,
        "鋁箔包":0,
        "手搖杯": 0,
        "metal": 0,
        "general": 0,
        
        
    }

    prompts = {
        "衛生紙":[
            "Common and used a tissue, it should be white ,wrinkled or crumpled to indicate that it has been used. The appearance should be natural and realistic, some stains may be present."
            # "A wrinkled tissue paper wad on a carpeted living room floor, next to a sofa, with warm lamplight creating soft shadows"
            # "A wadded-up white tissue paper on a park bench, surrounded by fallen autumn leaves, natural sunlight highlighting its wrinkled texture"
            # "A balled-up tissue paper on a student desk, with textbooks and pencil cases in the background, fluorescent lighting from above"
            # "A transparent plastic bottle with a simple design, floating on a clean reflective surface that mirrors the bottle. The background is an abstract gradient of soft blue tones, giving a futuristic feel."

        ]
        ,
         "鋁箔包":[
          "A single red milk tea carton with Chinese characters and a blue straw, centered on a clean wooden desk surface",
          "An abandoned red milk tea box resting on weathered concrete, surrounded by subtle ground textures",
          "A red beverage carton photographed straight-on against a pure white background, featuring clear Asian typography and branding",
          "A milk tea box with a blue straw positioned in the center of a smooth marble countertop",
          "A standalone red drink carton captured from a slight downward angle on a minimalist black surface"
        ]
        ,
        "plastic": [
            "a single discarded plastic bottle on a wooden surface, realistic, 4k, high quality",
            "a solitary plastic bag lying on the ground, realistic, 4k, high quality",
            "a clear plastic container placed on a flat surface, realistic, 4k, high quality",
            "a lone plastic straw on a desk, realistic, 4k, high quality"
        ],
        "手搖杯": [
            # "a crumpled sheet of paper on a wooden table, realistic, 4k, high quality",
            # "a single newspaper page lying on a concrete floor, realistic, 4k, high quality",
            # "a solitary used tissue paper on a desk, realistic, 4k, high quality"
            "Generate an image of a commonly seen used paper cup from a Taiwanese beverage shop. The cup should be slightly crumpled or wrinkled to indicate it has been used, with a straw hole on the lid or without a lid. The design can include a simple printed logo or abstract pattern, often found in milk tea or juice shops. The cup should have a natural and realistic appearance."],
        "metal": [
            "a rusty metal can lying on a dirt ground, realistic, 4k, high quality",
            "a single soda can on a concrete surface, realistic, 4k, high quality",
            "a small metal bolt placed on a wooden table, realistic, 4k, high quality",
            "a solitary aluminum foil crumpled on the ground, realistic, 4k, high quality"
        ],

         "general": [
            "a blue disposable surgical mask , realistic, 4k, high quality",
            "a crumpled snack wrapper on a table, realistic, 4k, high quality",
            "an empty candy wrapper , realistic, 4k, high quality",
            "an empty and used chocolate bar wrapper discarded on the ground, realistic, 4k, high quality",
            "an empty crushed aluminum foil snack pouch lying on the floor, realistic, 4k, high quality"
        ]
    }

    device = get_device()  # 獲取裝置（改用 PyTorch 的 GPU 或 CPU）
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(device)  # 將模型移到裝置，並使用 torch.float16

    target_size = (512, 512)  # 設定固定圖片大小

    for category, num_images in categories.items():
        output_dir = os.path.join(output_base_dir, category)
        inspection_dir = os.path.join(inspection_base_dir, category)
        generate_images(pipe, category, num_images, prompts[category], output_dir, inspection_dir, target_size, device)

    print("Image generation completed!")

if __name__ == "__main__":
    main()
