import pandas as pd
from .bordered_table import BorderedTableExtractor
from .borderless_table import BorderlessTableExtractor


class TableExtractor:
    
    def extract_tables(self, image):
        ex_grid = BorderedTableExtractor()
        found, tables = ex_grid.get_tables(image)
        
        if found:
            print('No of Tables :', len(tables))
            for table in tables:
                data = ex_grid.get_text_from_table(table)
                df = pd.DataFrame.from_dict(data).T
                return df
            
        else:
            ex = BorderlessTableExtractor()
            bordered_table = ex.draw_grids_on_borderless_table(image)
            found, tables = ex_grid.get_tables(bordered_table)
            
            if found:
                print('No of Tables :', len(tables))
                for table in tables:
                    data = ex_grid.get_text_from_table(table)
                    df = pd.DataFrame.from_dict(data).T
                    return df
            else: 
                print('No Tables detected')
                return []
            
        