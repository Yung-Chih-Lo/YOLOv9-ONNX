# YOLOv9-ONNX-INFERENCE

> 歷史專案，保留原始推論程式與環境紀錄供參考。本次文件整理未重新安裝環境或執行模型推論。

使用 Python、ONNX Runtime 與 OpenCV 進行 YOLOv8／YOLOv9 ONNX 物件偵測。以下原始範例與環境版本屬歷史紀錄，不代表對所有匯出模型或目前套件版本的相容性保證。

## 重現前需準備

- Git 中未包含 `resources/`；請自行準備 `resources/images/test.jpg` 與 `resources/weights/yolov9m-converted.onnx`，或在 `main.py` 調整路徑。
- 原始權重分享連結保留於下方，未重新確認可用性、檔案版本或雜湊。
- `main.py` 預設 `device="cuda"`，並使用 `cv2.imshow`，需要相容的 GPU 環境與圖形介面。
- `device="cpu"` 會選擇 CPU provider；但 session 模組仍直接 import Torch，因此 CPU 路徑亦依賴 Torch。
- `requirements.txt` 原樣保留，包含兩個 ONNX Runtime 套件條目與 `torch==2.3.0+cu121`。原文件沒有完整記錄 Torch wheel 的安裝來源、cuDNN 與各套件的實際安裝順序；它不是已驗證可直接重建的環境鎖檔。下方 CUDA 安裝指令僅供歷史參考。

## 模型介面與輸出

現存程式使用第一個模型輸入及第一個輸出；輸入影像轉 RGB、縮放後除以 255，轉為 float32 NCHW 張量。預設輸入尺寸為 `(640, 640)`，由 `imgsz` 指定，未自動套用讀取到的模型輸入尺寸。

後處理預期第一個輸出經 `squeeze(...).T` 後，每列為 `[cx, cy, width, height, class_scores...]`。其他輸出布局、已內建 NMS 的模型或其他任務模型，不能直接假設相容。前處理直接縮放影像，沒有 letterbox。

`model(image)` 回傳偵測字典列表，欄位為 `class_index`、`confidence`、`box`；box 為原圖座標的 `[x1, y1, x2, y2]`。繪圖類別名稱由模型的 `names` metadata 讀取，現有解析預期其內容為字典。信心門檻實際預設為 0.7，IoU 門檻為 0.3。

## 保存方式

保留原始程式、依賴版本、示例輸出及 [GPL-3.0 授權](LICENSE.md)。已追蹤的 Python 快取仍保留；新增的忽略規則僅防止未追蹤的快取與本機環境繼續加入版本控制。後續若重啟使用，應在獨立分支重建環境並驗證模型輸出與後處理。

## 原始示例與環境紀錄

![output image](output.png)
using yolov9m.onnx

# Important:
- support yolov8
- support yolov9

# My environment:
- python: 3.10.14
- cuda: 12.5
- onnx: 1.16.2
- onnxruntime: 1.18.1
- onnxruntime-gpu: 1.18.1

# Historical installation notes:
```bash
git clone https://github.com/Yung-Chih-Lo/YOLOv9-ONNX-INFERENCE.git
cd YOLOv9-ONNX-INFERENCE
pip install -r requirements.txt
```
## Visual C++ 2019 runtime
- official link: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

## ONNX-GPU (historical commands, not revalidated): 
- official link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
- CUDA 12.x: pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
- CUDA 11.X: pip install onnxruntime-gpu

## Model links:
- https://drive.google.com/drive/folders/1CZQX0LE_boLYKw5wjG3qO_1Ed59tgERm?usp=sharing


# Examples:

```python
import cv2
from yolo import YOLO
import time

if __name__ == "__main__":
    image_path = 'resources/images/test.jpg'
    onnx_model_path = 'resources/weights/yolov9m-converted.onnx'
    
    image = cv2.imread(image_path)
    model = YOLO(model_path=onnx_model_path, warmup=True, device="cuda")
    start = time.perf_counter()
    results = model(image)
    print(f"inference time: {(time.perf_counter() - start)*1000:.2f} ms")
    annotated_image = model.plot()
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

# Historical next step:
- YOLOv10 was a planned extension; this is not a claim of implemented support.

# References:
- YOLOv9 official repo: https://github.com/WongKinYiu/yolov9
- YOLOv9 onnx: https://github.com/danielsyahputra/yolov9-onnx
- YOLOv8 onnx: https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection