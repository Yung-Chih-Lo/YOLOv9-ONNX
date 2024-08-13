# 推理引擎的核心邏輯
import time
import cv2
import numpy as np
import onnxruntime
import torch
from yolo.utils import xywh2xyxy, multiclass_nms, get_onnx_session, get_input_details, get_output_details
from yolo.utils import Annotator

class YOLOv9:
    def __init__(self,
                 model_path: str,
                 class_mapping_path: str,
                 original_size: tuple[int, int] = (1280, 720),
                 score_thres: float = 0.1,
                 conf_thres: float = 0.4,
                 iou_thres: float = 0.4,) -> None:
        
        self.conf_threshold = conf_thres  # 設定信心閾值
        self.iou_threshold = iou_thres  # 設定IoU（交集並集比）閾值
        self.score_threshold = score_thres # 設定分數閾值, 預設0.1
        self.boxes =  None
        self.scores = None
        self.class_ids = None
        self.img_height = None
        self.img_width = None
        self.input_names = None
        self.input_shape = None
        self.input_height = None
        self.input_width = None
        self.output_names = None
        self.annotator = Annotator(model_path)
        self.model_path = model_path

        
        self.class_mapping_path = class_mapping_path
        
        self.image_width, self.image_height = original_size
        self.initialize_model()

    def initialize_model(self, path):
        self.session = get_onnx_session(path)
        # 獲取模型輸入輸出資訊
        self.input_names, self.input_shape, self.input_height, self.input_width = self.get_input_details(self.session)
        self.output_names, self.output_shape = self.get_output_details(self.session)
    
    def create_session(self) -> None:
        # TODO 待處理，class_mapping 的問題
        if self.class_mapping_path is not None:
            with open(self.class_mapping_path, 'r') as file:
                yaml_file = yaml.safe_load(file)
                self.classes = yaml_file['names']
                self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
                

    def preprocess(self) -> np.ndarray:
        """將輸入的影像進行預處理，包括轉換色彩空間、調整大小、縮放像素值和調整張量維度。

        Args:
            img (np.ndarray): 輸入的影像

        Returns:
            np.ndarray: 預處理後的影像張量
        """
        # TODO 考慮要不要把這個寫在utils裡面
        self.img_height, self.img_width = self.img.shape[:2]  # 獲取圖片的高度和寬度
        input_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # 將圖片轉換為RGB格式
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))  # 調整圖片大小
        input_img = input_img / 255.0  # 將像素值縮放到0到1之間
        input_img = input_img.transpose(2, 0, 1)  # 調整圖片張量的維度
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)  # 增加一個維度並轉換為float32型別
        return input_tensor

    
    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Rescale box
        boxes = predictions[:, :4]
        
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.int32)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.score_threshold, nms_threshold=self.iou_threshold)
        detections = []
        for bbox, score, label in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
                "class_name": self.get_label_name(label)
            })
        return detections
    
    def get_label_name(self, class_id: int) -> str:
        return self.classes[class_id]
        
    def detect(self, img: np.ndarray) -> list:
        self.img = img  
        input_tensor = self.preprocess()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        return self.postprocess(outputs)
    
    def draw_detections(self, img, detections: list):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            detections: List of detection result which consists box, score, and class_ids
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        for detection in detections:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = detection['box'].astype(int)
            class_id = detection['class_index']
            confidence = detection['confidence']

            # Retrieve the color for the class ID
            color = self.color_palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create the label text with class name and score
            label = f"{self.classes[class_id]}: {confidence:.2f}"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__=="__main__":

    weight_path = "weights/yolov9-c.onnx"
    image = cv2.imread("assets/sample_image.jpeg")
    h, w = image.shape[:2]
    detector = YOLOv9(model_path=f"{weight_path}",
                      class_mapping_path="weights/metadata.yaml",
                      original_size=(w, h))
    detections = detector.detect(image)
    detector.draw_detections(image, detections=detections)
    
    cv2.imshow("Tambang Preview", image)
    cv2.waitKey(0) 