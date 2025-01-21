# import torch
# from google.cloud import vision
# from PIL import Image
# from torchvision import transforms
# import os

# from google.cloud import vision

# YOUR_SERVICE = 'GCPultimate-task-445707-kye.json'

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = YOUR_SERVICE
# client = vision.ImageAnnotatorClient()
# # 加載您的 PyTorch 模型
# model = torch.load(r"D:\trash\runs\detect\train\weights\best.pt", map_location=torch.device("cpu"))
# model.eval()  # 設定模型為評估模式

# def gcp_classify(image_path):
#     """
#     使用 GCP Vision API 分類圖片。
    
#     :param image_path: 圖片的本地路徑
#     :return: GCP 的分類結果
#     """
#     # 建立 Vision API 客戶端
#     client = vision.ImageAnnotatorClient()

#     # 讀取圖片
#     with open(image_path, "rb") as image_file:
#         content = image_file.read()
#         image = vision.Image(content=content)

#     # 使用 Label Detection 或其他偵測方法
#     response = client.label_detection(image=image)
#     labels = response.label_annotations

#     # 取出最可能的分類結果
#     if labels:
#         return labels[0].description  # 返回可能性最高的類別名稱
#     else:
#         return "Unknown"

# def yolo_classify(image_path):
#     """
#     假設這裡是 YOLO 的推理函式，返回分類結果。
    
#     :param image_path: 圖片的本地路徑
#     :return: YOLO 的分類結果
#     """
#     # 示例：直接返回 YOLO 偵測的類別名稱（實際需要加入 YOLO 推理邏輯）
#     return "Class_A"

# def custom_model_classify(image_path):
#     """
#     使用自訓練的 PyTorch 模型 (best.pt) 進行分類。
    
#     :param image_path: 圖片的本地路徑
#     :return: 自定義模型的分類結果
#     """
#     # 圖片預處理（根據模型的需求進行調整）
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # 假設模型需要 224x224 輸入大小
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # 打開圖片並進行預處理
#     image = Image.open(image_path).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0)  # 增加 batch 維度

#     # 模型推理
#     with torch.no_grad():
#         output = model(input_tensor)
#         _, predicted_class = torch.max(output, 1)

#     # 根據訓練時的類別名稱進行對應
#     class_names = ["boba", "lunchbox", "pouch","other","lastic"]  # 替換為實際類別名稱
#     return class_names[predicted_class.item()]

# def classify_result(yolo_result, gcp_result, custom_result):
#     """
#     判斷 YOLO、GCP 和自定義模型的分類結果，並返回最終分類結果。
    
#     :param yolo_result: YOLO 的分類結果
#     :param gcp_result: GCP 的分類結果
#     :param custom_result: 自定義模型的分類結果
#     :return: 最終分類結果
#     """
#     # 優先級範例：如果 YOLO 和自定義模型一致，返回該結果，否則返回 GCP 結果
#     if yolo_result == custom_result:
#         return yolo_result
#     else:
#         return gcp_result

# if __name__ == "__main__":
#     # 圖片路徑
#     image_path = r"C:\Users\TMP214\Pictures\紙杯.jpg"

#     # 獲取 YOLO 的分類結果
#     yolo_result = yolo_classify(image_path)

#     # 獲取 GCP 的分類結果
#     gcp_result = gcp_classify(image_path)

#     # 獲取自定義模型的分類結果
#     custom_result = custom_model_classify(image_path)

#     # 獲取最終分類結果
#     final_result = classify_result(yolo_result, gcp_result, custom_result)

#     print(f"YOLO result: {yolo_result}")
#     print(f"GCP result: {gcp_result}")
#     print(f"Custom model result: {custom_result}")
#     print(f"Final classification result: {final_result}")

# ___________________

# from ultralytics import YOLO
# import torch
# from google.cloud import vision
# from PIL import Image
# import os

# YOUR_SERVICE = 'GCPultimate-task-445707-kye.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = YOUR_SERVICE

# # 正確載入 YOLO 模型
# model = YOLO(r"D:\trash\runs\detect\train\weights\best.pt")

# def yolo_classify(image_path):
#     """
#     使用 YOLO 模型進行圖片分類。
#     """
#     try:
#         # 使用 YOLO 的 predict 方法
#         results = model.predict(source=image_path, conf=0.25)
        
#         # 檢查是否有檢測結果
#         if len(results) > 0 and len(results[0].boxes) > 0:
#             # 獲取最高置信度的預測
#             boxes = results[0].boxes
#             class_id = int(boxes.cls[boxes.conf.argmax()])
#             confidence = float(boxes.conf.max())
            
#             # 類別名稱映射
#             class_names = ["boba", "lunchbox", "pouch", "other", "plastic"]
#             if class_id < len(class_names):
#                 print(f"Detected {class_names[class_id]} with confidence: {confidence:.2f}")
#                 return class_names[class_id]
        
#         print("No detection or low confidence detection")
#         return "unknown"
        
#     except Exception as e:
#         print(f"Error in YOLO inference: {str(e)}")
#         return "error"

# def gcp_classify(image_path):
#     """
#     使用 GCP Vision API 分類圖片。
#     """
#     client = vision.ImageAnnotatorClient()
#     with open(image_path, "rb") as image_file:
#         content = image_file.read()
#         image = vision.Image(content=content)

#     response = client.label_detection(image=image)
#     labels = response.label_annotations

#     if labels:
#         return labels[0].description
#     return "Unknown"
# def gemini(YOUR_JPG):
#     import os
#     from time import time

#     from vertexai.generative_models import GenerativeModel, Part



#     # gcloud auth application-default login
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GCPultimate-task-445707-kye.json'

#     model = GenerativeModel('gemini-2.0-flash-exp')  # may vary model name

#     generation_config = {
#         'max_output_tokens': 100,
#         'temperature': 0,
#         'response_mime_type': 'application/json'
#     }

#     prompt = '''請你辨識圖片並分成以下分類
#         names:
#         - boba
#         - lunchbox
#         - pouch
#         - other
#         - plastic
#         回傳信心大於0.5以上的
#         並以類似下列格式回傳
#         {
#         "other": "0.5",
#         "plastic": "0.8"
#         }'''

#     with open(YOUR_JPG, 'rb') as f:
#         data = Part.from_data(data=f.read(), mime_type='image/jpeg')

#     start = time()

#     r = model.generate_content(
#         [prompt, data],
#         generation_config=generation_config
        
#     )
#     return r
# def classify_result(yolo_result, gcp_result):
#     """
#     結合 YOLO 和 GCP 的分類結果。
#     """
#     if yolo_result not in ["unknown", "error"]:
#         return yolo_result
#     return gcp_result

# if __name__ == "__main__":
#     try:
#         image_path = r"D:\trash\project-all+newmilk\images\train\lunchbox0166.jpg"

#         # 執行分類
#         yolo_result = yolo_classify(image_path)
#         gcp_result = gcp_classify(image_path)
#         final_result = classify_result(yolo_result, gcp_result)
#         g_result = gemini(image_path)
#         r = g_result.candidates[0].content.parts[0].text
#         # print(r)
#         # # text = g_result['candidates'][0]['content']['parts'][0]['text']
#         import json
#         data = json.loads(r)
#         items_and_probabilities = [(item, float(prob)) for item, prob in data.items()]
#         # # # 3. 提取 'names' 字段
#         # names = [item['names'] for item in data]
#         # print(names)
#         # 打印結果
#         print(f"\nResults:")
#         print(f"YOLO result: {yolo_result}")
#         print(f"GCP result: {gcp_result}")
#         print(f"Final classification result: {final_result}")
#         print(f"Gemini result: {items_and_probabilities}")
#     except Exception as e:
#         print(f"Main error: {str(e)}")
#         import traceback
#         print(traceback.format_exc())


# _____________________________________________________-

# from ultralytics import YOLO
# import torch
# from google.cloud import vision
# from PIL import Image
# import os
# import json
# from vertexai.generative_models import GenerativeModel, Part

# YOUR_SERVICE = 'GCPultimate-task-445707-kye.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = YOUR_SERVICE

# # 載入 YOLO 模型
# model = YOLO(r"C:\Users\TMP214\Downloads\best.pt")

# def yolo_classify(image_path):
#     """
#     使用 YOLO 模型進行圖片分類，返回類別和置信度。
#     """
#     try:
#         results = model.predict(source=image_path, conf=0.25)
        
#         if len(results) > 0 and len(results[0].boxes) > 0:
#             boxes = results[0].boxes
#             class_id = int(boxes.cls[boxes.conf.argmax()])
#             confidence = float(boxes.conf.max())
            
#             class_names = ["boba", "lunchbox", "pouch", "other", "plastic"]
#             if class_id < len(class_names):
#                 return {class_names[class_id]: confidence}
        
#         return {"unknown": 0.0}
        
#     except Exception as e:
#         print(f"Error in YOLO inference: {str(e)}")
#         return {"error": 0.0}

# def gemini(image_path):
#     """
#     使用 Gemini 進行圖片分類。
#     """
#     model = GenerativeModel('gemini-2.0-flash-exp')
    
#     generation_config = {
#         'max_output_tokens': 100,
#         'temperature': 0,
#         'response_mime_type': 'application/json'
#     }

#     prompt = '''請你辨識圖片並分成以下分類
#         names:
#         - boba
#         - lunchbox
#         - pouch
#         - other
#         - plastic
#         鋁箔包分類到pouch，紙杯分類到boba，寶特瓶分類到plastic
#         回傳信心大於0.5以上的
#         並以類似下列格式回傳
#         {
#         "other": "0.5",
#         "plastic": "0.8"
#         }'''

#     with open(image_path, 'rb') as f:
#         data = Part.from_data(data=f.read(), mime_type='image/jpeg')

#     response = model.generate_content([prompt, data], generation_config=generation_config)
#     result_text = response.candidates[0].content.parts[0].text
#     return json.loads(result_text)

# def compare_results(yolo_result, gemini_result):
#     """
#     比較 YOLO 和 Gemini 的結果並決定最終輸出。
#     """
#     yolo_class = list(yolo_result.keys())[0]
#     yolo_conf = list(yolo_result.values())[0]
    
#     # 找出 Gemini 結果中置信度最高的類別
#     if gemini_result:
#         gemini_class, gemini_conf = max(gemini_result.items(), key=lambda x: float(x[1]))
#         gemini_conf = float(gemini_conf)
#     else:
#         return {"result": yolo_result, "source": "YOLO (Gemini無結果)"}
    
#     # 分析結果差異
#     results_match = yolo_class in gemini_result
    
#     # 詳細比較信息
#     comparison = {
#         "yolo_prediction": {yolo_class: yolo_conf},
#         "gemini_prediction": {gemini_class: gemini_conf},
#         "results_match": results_match,
#     }
    
#     # 決定最終結果
#     if results_match:
#         final_result = {"result": yolo_result, "source": "YOLO（結果一致）"}
#     else:
#         final_result = {"result": {gemini_class: gemini_conf}, "source": "Gemini（結果不一致）"}
    
#     return {**comparison, **final_result}

# if __name__ == "__main__":
#     try:
#         image_path = r"D:\trash\project-all+newmilk\images\train\lunchbox0101.jpg"

#         # 執行分類
#         yolo_result = yolo_classify(image_path)
#         gemini_result = gemini(image_path)
        
#         # 比較結果
#         final_analysis = compare_results(yolo_result, gemini_result)
        
#         # 打印詳細結果
#         print("\n分類結果分析:")
#         print(f"YOLO 預測: {final_analysis['yolo_prediction']}")
#         print(f"Gemini 預測: {final_analysis['gemini_prediction']}")
#         print(f"結果是否一致: {'是' if final_analysis['results_match'] else '否'}")
#         print(f"\n最終結果: {final_analysis['result']}")
#         print(f"來源: {final_analysis['source']}")
        
#     except Exception as e:
#         print(f"錯誤: {str(e)}")
#         import traceback
#         print(traceback.format_exc())

#________________________________________
#攝像頭
from ultralytics import YOLO
import torch
from google.cloud import vision
import cv2
import os
import json
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import numpy as np
from datetime import datetime
import threading
import queue
import time

YOUR_SERVICE = 'GCPultimate-task-445707-kye.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = YOUR_SERVICE

# 載入 YOLO 模型
model = YOLO(r"D:\trash\runs\detect\train2\weights\best.pt")

# 創建共享隊列，增加緩衝區大小
frame_queue = queue.Queue(maxsize=2)  # 增加到2以減少阻塞
result_queue = queue.Queue(maxsize=2)

def draw_detection(frame, detection):
    """
    在圖像上繪製最佳檢測結果。
    """
    if detection is None:
        return
        
    x1, y1, x2, y2 = detection["bbox"]
    cls = detection["class"]
    conf = detection["confidence"]
    
    # 使用綠色框
    color = (0, 255, 0)
    
    # 繪製邊界框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 準備標籤文字
    label = f"{cls} {conf:.2f}"
    
    # 獲取文字大小
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # 繪製文字背景
    cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
    
    # 繪製文字
    cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def yolo_classify(frame):
    """
    使用 YOLO 模型進行圖片分類，返回類別、置信度和邊界框。
    """
    try:
        # 使用半精度浮點數加速推理
        with torch.cuda.amp.autocast():
            results = model.predict(source=frame, conf=0.25, verbose=False)  # 關閉詳細輸出
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            class_names = ["boba", "lunchbox", "pouch", "other", "plastic"]
            
            best_idx = boxes.conf.argmax()
            class_id = int(boxes.cls[best_idx])
            confidence = float(boxes.conf[best_idx])
            
            if class_id < len(class_names):
                x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
                best_detection = {
                    "class": class_names[class_id],
                    "confidence": confidence,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                }
                return {best_detection["class"]: best_detection["confidence"]}, best_detection
        
        return {"unknown": 0.0}, None
        
    except Exception as e:
        print(f"Error in YOLO inference: {str(e)}")
        return {"error": 0.0}, None

def gemini(image):
    """
    使用 Gemini 進行圖片分類，增加快取機制。
    """
    try:
        # 降低圖片解析度以加速處理
        height, width = image.shape[:2]
        scale_factor = 0.5  # 縮小一半
        resized_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
        
        model = GenerativeModel('gemini-2.0-flash-exp')
        
        generation_config = {
            'max_output_tokens': 100,
            'temperature': 0,
            'response_mime_type': 'application/json'
        }

        prompt = '''請你辨識圖片並分成以下分類
            names:
            - boba
            - lunchbox
            - pouch
            - other
            - plastic
            回傳信心大於0.5以上的
            並以類似下列格式回傳
            {
            "other": "0.5",
            "plastic": "0.8"
            }'''

        is_success, buffer = cv2.imencode(".jpg", resized_image, [cv2.IMWRITE_JPEG_QUALITY, 80])  # 降低JPEG品質
        if not is_success:
            return {}
        
        image_bytes = buffer.tobytes()
        data = Part.from_data(data=image_bytes, mime_type='image/jpeg')

        response = model.generate_content([prompt, data], generation_config=generation_config)
        result_text = response.candidates[0].content.parts[0].text
        return json.loads(result_text)
    except Exception as e:
        print(f"Error in Gemini inference: {str(e)}")
        return {}

def compare_results(yolo_result, gemini_result):
    """
    比較 YOLO 和 Gemini 的結果並決定最終輸出。
    """
    yolo_class = list(yolo_result.keys())[0]
    yolo_conf = list(yolo_result.values())[0]
    
    if gemini_result:
        gemini_class, gemini_conf = max(gemini_result.items(), key=lambda x: float(x[1]))
        gemini_conf = float(gemini_conf)
        results_match = yolo_class in gemini_result
        
        if results_match:
            return {"class": yolo_class, "confidence": yolo_conf, "source": "YOLO"}
        else:
            return {"class": gemini_class, "confidence": gemini_conf, "source": "Gemini"}
    
    return {"class": yolo_class, "confidence": yolo_conf, "source": "YOLO"}

def detection_thread():
    """
    在單獨的執行緒中運行檢測
    """
    last_gemini_check = datetime.now()
    gemini_interval = 5  # 增加間隔到5秒以減少API調用頻率
    last_gemini_result = {}
    skip_frames = 0  # 用於幀跳過
    
    while True:
        try:
            frame = frame_queue.get()
            if frame is None:
                break
            
            skip_frames += 1
            if skip_frames % 2 != 0:  # 每隔一幀處理一次
                continue
                
            # YOLO 檢測
            yolo_result, detection = yolo_classify(frame)
            
            # Gemini 檢測
            current_time = datetime.now()
            if (current_time - last_gemini_check).total_seconds() > gemini_interval:
                last_gemini_result = gemini(frame)
                last_gemini_check = current_time
            
            # 比較結果
            final_result = compare_results(yolo_result, last_gemini_result)
            
            # 將結果和檢測框資訊放入隊列
            if not result_queue.full():
                result_queue.put((final_result, detection))
                
        except Exception as e:
            print(f"Detection thread error: {str(e)}")

def run_camera_detection():
    """
    使用攝像頭進行即時辨識。
    """
    # 初始化攝像頭
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 設定FPS
    
    # 啟動檢測執行緒
    detector = threading.Thread(target=detection_thread)
    detector.daemon = True  # 設為守護執行緒
    detector.start()
    
    print("按 'q' 退出程式")
    last_result = {"class": "unknown", "confidence": 0.0, "source": "None"}
    last_detection = None
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 更新檢測幀
            if not frame_queue.full():
                frame_queue.put(frame)
            
            # 獲取結果（非阻塞）
            try:
                last_result, last_detection = result_queue.get_nowait()
            except queue.Empty:
                pass
            
            # 繪製檢測框
            draw_detection(frame, last_detection)
            
            # 顯示結果
            result_text = f"Class: {last_result['class']} ({last_result['confidence']:.2f})"
            source_text = f"Source: {last_result['source']}"
            
            cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, source_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Object Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        frame_queue.put(None)
        detector.join(timeout=1.0)  # 設定超時
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_camera_detection()
    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        print(traceback.format_exc())
# _____________________________________

