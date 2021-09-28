from dataclasses import dataclass
import numpy as np
import pandas as pd

import cv2
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

@dataclass
class BorderedTableExtractor:
    
    blur_kernel_size: tuple = (17, 17)
    min_table_area: int = 1e5
    min_cell_width: int = 40
    min_cell_height: int = 10
    
    def get_tables(self, image):
        if isinstance(image, str):
            gray_image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            gray_image = image.copy()
        else:
            raise ValueError("image can either be path to file i.e. string or an instance of np.ndarray.")
        threshold_image = self._preprocess_image(gray_image)
        contours = self._get_contours(threshold_image)
        bounding_rects = self._find_tables(contours)
        tables = [gray_image[y:y+h, x:x+w] for x, y, w, h in bounding_rects]
        
        if tables:
            return True, tables
        else:
            return False, None
        
    def get_text_from_table(self, table_image):
        ''''''
        threshold_image = self._preprocess_image(table_image)
        contours = self._get_contours(threshold_image)
        cells = self._find_table_cells(contours)
        rows = self._group_cells_by_row(cells)
        text_data = self._extract_text_from_image(table_image, rows)
        return text_data
                
    
    def _preprocess_image(self, image):
        ''' Perform Image Thresholding'''
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, self.blur_kernel_size, cv2.BORDER_CONSTANT)
        threshold_image = cv2.adaptiveThreshold(~blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 15, -2)
        return threshold_image
    
    def _get_contours(self, image):
        SCALE = 5

        image_width, image_height = image.shape
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
        hor_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, hor_kernel)
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_height / SCALE)))
        ver_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, ver_kernel)

        hor_dilated = cv2.dilate(hor_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
        ver_dilated = cv2.dilate(ver_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))

        grids = hor_dilated + ver_dilated
        contours, heirarchy = cv2.findContours(grids, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours
    
    def draw_contours(self, image):
        gray_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        threshold_image = self._preprocess_image(gray_image)
        contours = self._get_contours(threshold_image)
        
        contour_image = np.zeros(gray_image.shape)
        contour_boxes = []

        for contour in contours:
            box = cv2.boundingRect(contour)
            x,y,w,h = box
            contour_image = cv2.drawContours(contour_image, [contour], -1, (0, 0, 255), 1)

        return contour_image

    def _find_tables(self, contours):
        
        contours = [c for c in contours if cv2.contourArea(c) > self.min_table_area]
        contour_area = [cv2.contourArea(c) for c in contours]
        if contour_area:
            contours = [contours[np.argmax(contour_area)]]
        arcs = [0.1 * cv2.arcLength(c, True) for c in contours]
        polygons = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, arcs)]

        # approx_rects = [p for p in approx_polys if len(p) == 4]
        bounding_rects = [cv2.boundingRect(a) for a in polygons]
        return bounding_rects
    
    def _find_table_cells(self, contours):
        arcs = [0.05 * cv2.arcLength(c, True) for c in contours]
        polygons = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, arcs)]
        
        # Filter out contours that aren't rectangular. 
        approx_rects = [p for p in polygons if len(p) == 4]
        bounding_rects = [cv2.boundingRect(a) for a in polygons]
        cells = [r for r in bounding_rects 
                 if self.min_cell_width < r[2] and self.min_cell_height < r[3]
                ]

        largest_cell = max(cells, key=lambda r: r[2] * r[3])
        cells = [b for b in bounding_rects if b is not largest_cell]
        
        return cells
    
    def _group_cells_by_row(self, cells: list):
        cell_rows = []
        while cells:
            first = cells[0]
            rest = cells[1:]
            row = [c for c in rest if self._cells_in_same_row(c, first)]
            row = sorted([first] + row, key=lambda c: c[0])
            cell_rows.append(row)
            cells = list(set(rest) - set(row))
        
        cell_rows = sorted(cell_rows, key=lambda x: x[0][1])
        return cell_rows
    
    def _cells_in_same_row(self, c1, c2):
        c1_center = c1[1] + c1[3] - c1[3] / 2
        c2_top = c2[1]
        c2_bottom = c2[1] + c2[3]
        return c2_top < c1_center < c2_bottom

    def _extract_text_from_image(self, image, rows):
        config = '--oem 1 --psm 12 --tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
        details = pytesseract.image_to_data(image, 
                                            output_type=Output.DICT, 
                                            lang="eng",
                                            config=config)
        df = pd.DataFrame(details)
        df['right'] = df.left + df.width
        df['bottom'] = df.top + df.height
        
        text_data = {}
        for ind, row in enumerate(rows):
            cols = {}
            for i, (x, y, w, h) in enumerate(row):
                tmp_df = df[(df.left>=x) & (df.right<=x+w) & (df.top>=y) & (df.bottom<=y+h)]
                text = ' '.join([i.strip() for i in tmp_df.text.values if i.strip() != ''])
                cols[i] = text
            text_data[ind] = cols
        return text_data