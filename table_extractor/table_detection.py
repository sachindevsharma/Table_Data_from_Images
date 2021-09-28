from dataclasses import dataclass
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from .detectron_config import cfg

predictor = DefaultPredictor(cfg)


@dataclass
class TableDetection:

    def find_tables_in_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("image can either be path to file i.e. string or an instance of np.ndarray.")
            
        outputs = predictor(image)
        boxes = outputs["instances"].pred_boxes
        table_data = self.extract_table_coordinates(boxes)
        if table_data:
            x1, y1, x2, y2 = table_data
            return image[y1:y2, x1:x2]
        else:
            return None
        
    def extract_table_coordinates(self, boxes):

        if len(boxes) == 0:
            return None

        elif len(boxes) == 1:
            for i in boxes:
                x1, y1 = int(i[0]), int(i[1])
                x2, y2 = int(i[2]), int(i[3])
        else:
            x1, x2 = [], []
            y1, y2 = [], []
            for i in boxes:
                x1.append(i[0])
                y1.append(i[1])
                x2.append(i[2])
                y2.append(i[3])

            x1, y1 = int(min(x1)), int(min(y1))
            x2, y2 = int(max(x2)), int(max(y2))        

        return x1, y1, x2, y2
    

    