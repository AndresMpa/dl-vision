import os
import re


def extract_file_name_data(file_name, prefix, suffix):
    """
    Extracts the model name from a file name based on a given prefix and suffix

    Args:
        file_name (str): The input file name containing the model name
        prefix (str): The prefix before the model name in the file name
        suffix (str): The suffix after the model name in the file name

    Returns:
        str or None: The extracted model name or None if not found
    """
    pattern = rf'{re.escape(prefix)}([a-zA-Z0-9]+){re.escape(suffix)}'
    match = re.search(pattern, file_name)

    if match:
        return match.group(1)
    else:
        return None


def get_current_path(path):
    """
    Gives the current absolute path

    Args:
        path (str): Path to add into root path

    Returns:
        dir_path (str): Absolute directory path
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    dir_path = os.path.join(root_dir, path)
    return dir_path


def file_exists(file_path):
    """
    Check if a file exists

    Args:
        - file_path (str): The path to the file

    Returns:
        - bool: True if the file exists, False otherwise
    """
    return os.path.exists(file_path)


def list_files(path, extention):
    """
    List files with a specific extension in a directory

    Args:
        - path (str): The path to the directory containing the files
        - extension (str): The file extension to filter files. Only files with
        this extension will be included in the list

    Returns:
        - files (list): A list of file names in the specified directory
        (`path`) that have the specified file extension (`extension`)
    """
    files = [f for f in os.listdir(path) if f.endswith(extention)]
    return files


def check_path(path):
    """
    Checks if a path exist

    Args:
        path (str): Path to check

    Returns:
        True is exist, False if not
    """
    return (os.path.isdir(path))


def create_path(path, filename):
    """
    Creates an absolute path for a file

    Args:
        path (str): A directory path to host the file
        filename (str): A full file name extension included

    Returns:
        file_path (str): A file path to host a file
    """
    absolute_path = get_current_path(path)
    file_path = os.path.join(absolute_path, filename)

    return file_path


def create_dir(path):
    """
    Creates a given path checking is exists of not

    Args:
        path (str): Path to create
    """
    try:
        if (check_path(path)):
            return
        else:
            dir_path = get_current_path(path)
            os.mkdir(dir_path)

            print(f"Directory {path} created")
    except OSError:
        print(f"[ERROR]: Creating {path}")


def clear_dir(path):
    """
    Delete all files inside a given directory

    Args:
        path (str): Path to directory to clear
    """
    try:
        path = get_current_path(path)
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        print("Files deleted successfully.")
    except OSError:
        print(f"[ERROR]: Deleting files at {path}")
