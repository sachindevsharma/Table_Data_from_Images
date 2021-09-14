from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import statistics

import cv2
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

@dataclass
class BorderlessTableExtractor:
    
    min_text_height_limit: int = 10
    
    def draw_grids_on_borderless_table(self, image):
        gray_image = cv2.imread(image)
        image = self._preprocess_borderless_table(gray_image)
        threshold_image = self._preprocess_image(image)
        contours = self.find_contours(~threshold_image)
        contour_image = self.get_contour_image(contours, gray_image.shape)
        contours = self.find_contour_on_contours(contour_image)
        contours = self.remove_noise(contours)
        contour_boxes = self.get_bounding_boxes(contours)
        cell_rows = self._group_cells_by_row(contour_boxes)
        columns = self._group_cells_by_col(contour_boxes)
        hor_lines, ver_lines = self._get_lines_to_draw_grid(image, cell_rows, columns)
        final_image = self.draw_lines(image, hor_lines, ver_lines)
        return final_image
        
            
            
    def find_contours(self, threshold_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 5))
        dilated_image = cv2.dilate(threshold_image, kernel, anchor=(-1, -1), iterations=1)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def get_contour_image(self, contours, img_shape):
        contour_image = np.zeros(img_shape, dtype=np.uint8)
        contour_boxes = []

        for contour in contours:
            box = cv2.boundingRect(contour)
            x,y,w,h = box
            contour_image = cv2.rectangle(contour_image, (x,y), (x+w, y+h), (255), -1)

        return contour_image
    
    def find_contour_on_contours(self, contour_image):
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(contour_image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(8, 8))
        open_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(open_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def remove_noise(self, contours, minContourArea=100):
        ''' 
        Removing NOISE by removing contours with area less than minContourArea 
        '''
        contours = [c for c in contours if cv2.contourArea(c) > minContourArea]
        return contours
    
    def get_bounding_boxes(self, contours):
        contour_boxes = []
        for contour in contours:
            box = cv2.boundingRect(contour)
            h = box[3]

            if self.min_text_height_limit < h:
                contour_boxes.append(box)
        return contour_boxes
    
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
    
    def _group_cells_by_col(self, cells: list):
        columns = []
        cells = sorted(cells, key=lambda x:x[1])
        cells = sorted(cells, key=lambda x:x[0])
        while cells:
            first = cells[0]
            rest = cells[1:]
            cols = [c for c in rest if self._cells_in_same_col(first, c)]
            cols = sorted([first] + cols, key=lambda c: c[1])
            columns.append(cols)
            cells = list(set(rest) - set(cols))

        columns = sorted(columns, key=lambda x: x[0][0])
        return columns

    def _cells_in_same_col(self, c1, c2):
        c1_left = c1[0]
        c1_right = c1[0] + c1[2]
        c2_left = c2[0] 
        c2_right = c2[0] + c2[2]
        return (c1_left <= c2_left <= c1_right) or (c2_left <= c1_left <= c2_right)
                
    def _get_lines_to_draw_grid(self, image, rows, columns):
        bottom = [max([y+h for (x,y,w,h) in row]) for row in rows]
        top = [min([y for (x,y,w,h) in row]) for row in rows][1:]
        
        right = [max([x+w for (x,y,w,h) in col]) for col in columns]
        left = [min([x for (x,y,w,h) in col]) for col in columns][1:]

        x_first, y_first = [5], [5]
        x_last, y_last = [image.shape[0] - 5], [image.shape[1] - 5]
        
        # Finding the center horizontally to draw the lines
        hor_lines = x_first + [b + int((a-b)/2) for a,b in zip(top, bottom)] + x_last
        ver_lines = y_first + [b + int((a-b)/2) for a,b in zip(left, right)] + y_last
        hor_lines, ver_lines = self._remove_overlaping_lines(hor_lines, ver_lines, rows, columns)
        
        return hor_lines, ver_lines
        
        
    def _remove_overlaping_lines(self, hor_lines, ver_lines, rows, columns):
        rows_x = []
        cols_y = []

        for row in rows:
            for (x,y,w,h) in row:
                rows_x.extend(list(range(x, x+w)))

        for col in columns:
            for (x,y,w,h) in col:
                cols_y.extend(list(range(y, y+h)))

        new_ver_lines = list(set(ver_lines) - set(rows_x))
        new_hor_lines = list(set(hor_lines) - set(cols_y))

        return new_hor_lines, new_ver_lines
    
    def draw_lines(self, image, hor_lines, ver_lines):
    
        final_image = image.copy()
        x1 = 0
        x2 = image.shape[1]

        y1 = 0
        y2 = image.shape[0]

        for i in hor_lines:
            y = int(i)
            final_image = cv2.line(final_image, (x1,y), (x2, y), (255,0,0), 1)

        for j in ver_lines:
            x = int(j)
            final_image = cv2.line(final_image, (x,y1), (x,y2), (255,0,0), 1)

        return final_image
    
    def _preprocess_image(self, image):
        ''' Perform Image Thresholding'''
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return threshold_image
    
    def _preprocess_borderless_table(self, image):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
        detected_lines = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            contour_image = cv2.drawContours(image, [c], -1, (255,255,255), 2)
        try:
            return contour_image
        except:
            return image