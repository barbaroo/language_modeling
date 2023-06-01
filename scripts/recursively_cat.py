import sys
import os

def recursive_cat(path):
    """Recursively concatenate the contents of all files in a given path.

    Args:
        path: The path to the directory to start with.

    Returns:
        A string containing the file names and contents of all the files in the directory tree.
    """

    contents = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_contents = f.read()
                    contents.append(f"{file_path}:\n{file_contents}")
            except Exception as e:
                print(f"Error reading file: {file_path}")
                print(e)
        elif os.path.isdir(file_path):
            contents.append(recursive_cat(file_path))

    return '\n\n'.join(contents)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
    path = sys.argv[1]
    contents = recursive_cat(path)
    print(contents)
