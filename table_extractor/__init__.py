from dataclasses import dataclass
import pandas as pd
from .bordered_table import BorderedTableExtractor
from .borderless_table import BorderlessTableExtractor
from .table_detection import TableDetection


@dataclass
class TableExtractor:
    
    def find_tables(self, image):
        tables = TableDetection().find_tables_in_image(image)
        all_data = []
        bordered_table = []
        
        if tables is not None:
            bordered_table, table_data = self.extract_table_data(tables)
            all_data.append(table_data)
        return tables, bordered_table, all_data
        
    def extract_table_data(self, image):
        ex_grid = BorderedTableExtractor()
        found, tables = ex_grid.get_tables(image)
        
        if found:
            print('No of Tables :', len(tables))
            for table in tables:
                data = ex_grid.get_text_from_table(table)
                df = pd.DataFrame.from_dict(data).T
                return table, df
            
        else:
            ex = BorderlessTableExtractor()
            bordered_table = ex.draw_grids_on_borderless_table(image)
            found, tables = ex_grid.get_tables(bordered_table)
            
            if found:
                print('No of Tables :', len(tables))
                for table in tables:
                    data = ex_grid.get_text_from_table(table)
                    df = pd.DataFrame.from_dict(data).T
                    return bordered_table, df
            else: 
                print('No Tables detected')
                return []
            
        