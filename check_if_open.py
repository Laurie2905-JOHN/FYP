import os
import psutil

def get_open_files_in_directory(directory_path):
    open_files = []

    for process in psutil.process_iter(['open_files']):
        try:
            for open_file in process.info['open_files']:
                if open_file.path.startswith(directory_path):
                    open_files.append(open_file.path)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    return open_files

directory_path = r"C:\Users\lauri\OneDrive\Documents (1)"

open_files_in_directory = get_open_files_in_directory(directory_path)
if open_files_in_directory:
    print("The following files are open in the directory:")
    for file_path in open_files_in_directory:
        print(file_path)
else:
    print("No files are open in the directory.")