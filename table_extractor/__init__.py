from dataclasses import dataclass
import pandas as pd
import numpy as np
from .bordered_table import BorderedTableExtractor
from .borderless_table import BorderlessTableExtractor
from .table_detection import TableDetection


@dataclass
class TableExtractor:
    
    def find_tables(self, image):
        
        table_data = []
        bordered_tables_image = []
        tables_from_image = TableDetection().find_tables_in_image(image)

        if tables_from_image:
            for table in tables_from_image:
                table_image, data = self.extract_table_data(table)
                if table_image.any():
                    bordered_tables_image.append(table_image)
                    table_data.append(data)
        return tables_from_image, bordered_tables_image, table_data
        
    def extract_table_data(self, image):
        ex_grid = BorderedTableExtractor()
        found, table = ex_grid.get_tables(image)
        
        if found:
            data = ex_grid.get_text_from_table(table)
            df = pd.DataFrame.from_dict(data).T
            return table, df
            
        else:
            ex = BorderlessTableExtractor()
            bordered_table = ex.draw_grids_on_borderless_table(image)
            found, table = ex_grid.get_tables(bordered_table)
            
            if found:
                data = ex_grid.get_text_from_table(table)
                df = pd.DataFrame.from_dict(data).T
                return bordered_table, df
            else: 
                return np.zeros((1,1)), []
            
        