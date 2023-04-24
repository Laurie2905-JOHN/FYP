import pandas as pd

def txt_to_excel(input_file, output_file):
    # Read the input file as a CSV using pandas
    data_frame = pd.read_csv(input_file, sep=',', lineterminator='\n')

    # Save the data frame as an Excel file using the openpyxl engine
    data_frame.to_excel(output_file, index=False, engine='openpyxl')

if __name__ == '__main__':
    input_file = "C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/zero_txt_data.txt"
    output_file = 'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/test.xlsx'


    txt_to_excel(input_file, output_file)
    print(f'Successfully converted {input_file} to {output_file}')