"""
    Module with auxiliary smaller functions for miscellaneous purposes.
"""


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
