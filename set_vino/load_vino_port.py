import numpy as np
from typing import Tuple, Dict
import openvino as ov

from set_vino.load_vino import loadVino

class LoadVino(loadVino) :
    def __Init__(model_path, device, label_map, model_imgsz):
        pass

    def Process_output(self,pred_boxes: np.ndarray, img_width, img_height):
        return self.process_output(pred_boxes= np.ndarray, img_width = img_width, img_height = img_height)

    def Draw_results(self,results:Dict, source_image:np.ndarray, label_map:Dict):
        return self.draw_results(results=results, source_image=sorce_image, label_map = label_map)