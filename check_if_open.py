# Function which gets a list of open files in a specific directory.
# Made if there were issues with open files causing errors in the dash app
# Import the required libraries
import psutil

# Define a function to get the list of open files in a specified directory
def get_open_files_in_directory(directory_path):
    # Initialize an empty list to store the open files' paths
    open_files = []

    # Iterate through all the running processes and get their open files information
    for process in psutil.process_iter(['open_files']):
        try:
            # Iterate through the open files of the current process
            for open_file in process.info['open_files']:
                # Check if the open file's path starts with the specified directory path
                if open_file.path.startswith(directory_path):
                    # Add the open file's path to the list of open files
                    open_files.append(open_file.path)
        # Handle exceptions caused by access denial or non-existent process
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    # Return the list of open files in the specified directory
    return open_files

# Specify the directory path to check for open files
directory_path = r""

# Call the function to get the list of open files in the specified directory
open_files_in_directory = get_open_files_in_directory(directory_path)

# Check if there are any open files in the directory
if open_files_in_directory:
    # Print the list of open files in the directory
    print("The following files are open in the directory:")
    for file_path in open_files_in_directory:
        print(file_path)
else:
    # Print a message if no files are open in the directory
    print("No files are open in the directory.")