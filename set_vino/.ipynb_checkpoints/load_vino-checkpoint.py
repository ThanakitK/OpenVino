import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Tuple, Dict
import cv2
from ultralytics.utils.plotting import colors
from ultralytics.utils import ops
import torch
from typing import Tuple
import openvino as ov
import json


class LoadVino :
    def __init__(self, model_path, device, label_map):
        self.core = ov.Core()
        self.det_ov_model = self.core.read_model(model_path)
        self.det_ov_model.reshape({0: [1, 3, 640, 640]})
        self.model = self.core.compile_model(self.det_ov_model, device)
        self.label_map = label_map
    
    def plot_one_box(self, box: np.ndarray, img: np.ndarray,
                     color: Tuple[int, int, int] = None,
                     label: str = None, line_thickness: int = 5):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    
    def draw_results(self,results:Dict, source_image:np.ndarray, label_map:Dict):
        first_result = results[0]  # Access the first dictionary in the list
        boxes = first_result["det"]  # Access the "det" key within the first dictionary
        for idx, (*xyxy, conf, lbl) in enumerate(boxes):
            label = f'{label_map[int(lbl)]} {conf:.2f}'
            source_image = self.plot_one_box(xyxy, source_image, label=label, color=colors(int(lbl)), line_thickness=1)
        return source_image 

    
    def letterbox(self,img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
    
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)
    
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    
        dw /= 2  # divide padding into 2 sides
        dh /= 2
    
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

        
    def preprocess_image(self, img0: np.ndarray):
        # resize
        img = self.letterbox(img0)[0]
        
        # Convert HWC to CHW
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img

        
    def image_to_tensor(self, image:np.ndarray):
        input_tensor = image.astype(np.float32)  # uint8 to fp32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # add batch dimension
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor

        
    def postprocess(
        self,
        pred_boxes: np.ndarray, 
        input_hw: Tuple[int, int], 
        orig_img: np.ndarray, 
        min_conf_threshold: float = 0.25, 
        nms_iou_threshold: float = 0.7, 
        agnosting_nms: bool = False, 
        max_detections: int = 300,
    ):
        nms_kwargs = {"agnostic": agnosting_nms, "max_det": max_detections}
        preds = ops.non_max_suppression(
            torch.from_numpy(pred_boxes),
            min_conf_threshold,
            nms_iou_threshold,
            nc=len(self.label_map),
            **nms_kwargs
        )
    
        results = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append({"det": [], "segment": []})
                continue
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
    
    
        return results

        
    def detect(self, image: np.ndarray):
        preprocessed_image = self.preprocess_image(image)
        input_tensor = self.image_to_tensor(preprocessed_image)
        result = self.model(input_tensor)
        boxes = result[self.model.output(0)]
        input_hw = input_tensor.shape[2:]
        detections = self.postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image)
        return detections


    def format_detection(self, detections, label_map):
        self.detections = detections
        self.label_map = label_map
        self.output_data = {
            "status": True,
            "code": 200,
            "message": "success",
            "data": []
        }
        for detection in self.detections:
            det_data = detection['det']
            for row in det_data:
                formatted_detection = {
                    "confident": float(row[4]),
                    "label": self.label_map[int(row[5])],
                    "description": "Object-Detection",
                    "xmin": int(row[0]),
                    "ymin": int(row[1]),
                    "xmax": int(row[2]),
                    "ymax": int(row[3])
                }
                self.output_data['data'].append(formatted_detection)

    def to_json(self):
        return json.dumps(self.output_data, indent=2)