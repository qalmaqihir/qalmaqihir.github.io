import os
import sys

def process_md_files(directory):
    # Define the string to search for
    target_string = "Notes [book] Data Science Handbook"

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            print(f"Reading file: {filename}")

            # Read the file content
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Check if the first line contains the target string
            # if lines and lines[0].strip() == target_string:
            print(f"Found target string in {filename}. Removing the first line.")
            # Remove the first line
            lines = lines[1:]

            # Write the modified content back to the file
            with open(file_path, "w") as file:
                file.writelines(lines)
            print(f"File saved after removing the target string: {filename}")
            # else:
            #     print(f"Target string not found in {filename}. No changes made.")
            print(f"Finished processing file: {filename}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    process_md_files(directory)
