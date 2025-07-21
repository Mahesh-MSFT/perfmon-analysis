def extract_header_from_column_name(column_name):
    """
    Extract the word between the first two backslashes and the next backslash from the column name.
    GPU pipeline optimized (CPU-optimized for string operations).
    """
    # Processing strategy: CPU-based (string processing)
    # String operations are inherently CPU-optimized and don't benefit from GPU acceleration
    
    try:
        start = column_name.find('\\\\') + 2
        end = column_name.find('\\', start)
        header = column_name[start:end]
    except IndexError:
        header = column_name.split()[0]
    return header
