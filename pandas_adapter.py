import pandas as pd
from enum import IntEnum


class File_types(IntEnum):
    CSV = 0,
    EXCEL = 1


pandas_readers = {
    File_types.CSV: pd.read_csv,
    File_types.EXCEL: pd.read_excel
}


pandas_writers = {
    File_types.CSV: pd.DataFrame.to_csv,
    File_types.EXCEL: pd.DataFrame.to_excel,
}


# file_extensions = {
#     '.xlsx': File_types.EXCEL
# }


def make_DataFrame_from_tensor(tensor):
    return pd.DataFrame(tensor.numpy())
