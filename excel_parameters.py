from enum import IntEnum
#excel file parameters

first_row = 6

# dna_feature_column_number = 2
dna_feature_column_number = 0
begin_feature_column = 10
end_feature_column = 22


class Parameters(IntEnum):
    dH = 0,
    dG = 1,
    dS = 2,
    Tm = 3,



# column_numbers = {
#     Parameters.dH: 4,
#     Parameters.dS: 5,
#     Parameters.dG: 6,
#     Parameters.Tm: 9,
# }

column_numbers = {
    Parameters.dH: 1,
    Parameters.dS: 2,
    Parameters.dG: 3,
    Parameters.Tm: 5,
}

# class input_dna_file():
#     def __init__(self, file_name, param_column_numbers_dict, first_row = None, end_row = None):
#