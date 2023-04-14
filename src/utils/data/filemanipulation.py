"""
    Module with auxiliary functions to manipulate files and directories.
"""

import os
import shutil

def create_dir(dir_name: str, subdirs: list = None):
    """
    Create a directory if it does not exist.
    Optionally, create subdirectories.

    Args:
        dir_name (str): Name of the directory to be created.
        subdirs (list): List of subdirectories to be added to the directory.

    Returns:
        None
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        if subdirs is not None:
            for subdir in subdirs:
                os.makedirs(os.path.join(dir_name, subdir))

def copy_files(file_names: list, source_dir: str, dest_dir: str):
    """
    Copy files from a source directory to a destination directory.

    Args:
        file_names (list): List of file names to be copied.
        source_dir (str): Source directory.
        dest_dir (str): Destination directory.

    Returns:
        None
    """
    for file_name in file_names:
        shutil.copyfile(
            os.path.join(source_dir, file_name),
            os.path.join(dest_dir, file_name)
        )