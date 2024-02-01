"""
    Module with auxiliary smaller functions for miscellaneous purposes.
"""
import os
import shutil
import sys
import io

import pandas as pd


def list_split(lst: list, ratio: list):
    """
    Split a list into multiple lists according to a given ratio.

    If the size of a split is not an integer, the last split will contain
    the remaining elements.

    Args:
        lst (list): List to be split.
        ratio (list): List of ratios for each split.

    Returns:
        list: List of lists.
    """
    split = []
    last_index = 0
    n_ratios = len(ratio)
    for i in range(n_ratios):
        r = ratio[i]
        if i == n_ratios - 1:
            split.append(lst[last_index:])
        else:
            split.append(lst[last_index:last_index + int(len(lst) * r)])
            last_index += int(len(lst) * r)
    return split


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


def catch_stdout(func):
    """
    Decorator to catch the stdout of a function.

    Args:
        func (function): Function to be decorated.

    Returns:
        function: Decorated function.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function.
        """
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        func(*args, **kwargs)
        results = new_stdout.getvalue()
        sys.stdout = old_stdout
        return results

    return wrapper


def list_subdirectories(directory):
    """
    Lists only subdirectories in a given directory and its subdirectories,
    but does not list files.

    Args:
        directory (str): The directory to list subdirectories from.

    Returns:
        list: A list of subdirectories.
    """
    subdirectories = []
    for dirpath, dirnames, _ in os.walk(directory):
        # Exclude files and only append directories to the subdirectories list
        subdirectories.extend([os.path.join(dirpath, dirname)
                              for dirname in dirnames])
    return subdirectories


def list_files(directory):
    """
    Lists only files in a given directory and its subdirectories,
    but does not list subdirectories.

    Args:
        directory (str): The directory to list files from.

    Returns:
        list: A list of files.
    """
    files = []
    for dirpath, _, filenames in os.walk(directory):
        # Exclude directories and only append files to the files list
        files.extend([os.path.join(dirpath, filename)
                      for filename in filenames])
    return files


def transform_multilevel_header(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Function to transform a dataframe with a multilevel header into a dataframe
    with a single level header.

    Args:
        dataframe (pd.DataFrame): Dataframe with a multilevel header.

    Returns:
        pd.DataFrame: Dataframe with a single level header.
    """
    new_columns = []
    dataframe_copy = dataframe.copy()
    for col in dataframe_copy.columns:
        valid_levels = [level for level in col if 'Unnamed' not in level]
        valid_levels = [level for level in valid_levels if level != '']
        if len(valid_levels) == 0:
            new_columns.append(col[-1])
        else:
            new_columns.append('_'.join(valid_levels))
    dataframe_copy.columns = new_columns

    return dataframe_copy
