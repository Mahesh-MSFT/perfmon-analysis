def remove_first_word_after_backslashes(col):
    """
    Remove the first word following the first pair of backslashes and the backslash immediately after that.
    """
    first_backslash = col.find('\\\\')
    second_backslash = col.find('\\', first_backslash + 2)
    if second_backslash != -1:
        return col[second_backslash + 1:]
    return col