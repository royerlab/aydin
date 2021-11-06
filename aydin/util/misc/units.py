def human_readable_byte_size(nbytes: int) -> str:
    """

    Parameters
    ----------
    nbytes : int

    Returns
    -------
    Human readable string : str

    """
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return f"{f} {suffixes[i]}"
