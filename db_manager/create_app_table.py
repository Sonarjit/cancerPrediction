from db_manager.table_columns import TABLE_COLUMNS
from db_manager.db_utils import create_table

def create_history_table():
    db_name = "db_manager/database/history"
    table_name = "history_table"
    columns = TABLE_COLUMNS[table_name]
    create_table(db_name, table_name, columns)

def create_single_inference_table():
    db_name = "db_manager/database/single_inference"
    table_name = "single_inference_table"
    columns = TABLE_COLUMNS[table_name]
    create_table(db_name, table_name, columns)