"""Human-readable formatting utilities for byte sizes."""


def human_readable_byte_size(nbytes: int) -> str:
    """Convert a byte count to a human-readable string with unit suffix.

    Automatically selects the most appropriate unit (B, KB, MB, GB, TB, PB)
    and formats the value with up to two decimal places, removing trailing
    zeros.

    Parameters
    ----------
    nbytes : int
        Number of bytes.

    Returns
    -------
    str
        Human-readable string, e.g. '1.5 GB' or '256 KB'.
    """
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return f"{f} {suffixes[i]}"
