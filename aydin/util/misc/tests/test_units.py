"""Tests for human_readable_byte_size utility."""

from aydin.util.misc.units import human_readable_byte_size


def test_zero_bytes():
    """Test that zero bytes renders as '0 B'."""
    assert human_readable_byte_size(0) == '0 B'


def test_sub_kb():
    """Test that sub-kilobyte values render in bytes."""
    assert human_readable_byte_size(512) == '512 B'


def test_exact_kb():
    """Test that exactly 1024 bytes renders as '1 KB'."""
    assert human_readable_byte_size(1024) == '1 KB'


def test_exact_mb():
    """Test that exactly 1 MB renders as '1 MB'."""
    assert human_readable_byte_size(1024**2) == '1 MB'


def test_exact_gb():
    """Test that exactly 1 GB renders as '1 GB'."""
    assert human_readable_byte_size(1024**3) == '1 GB'


def test_exact_tb():
    """Test that exactly 1 TB renders as '1 TB'."""
    assert human_readable_byte_size(1024**4) == '1 TB'


def test_exact_pb():
    """Test that exactly 1 PB renders as '1 PB'."""
    assert human_readable_byte_size(1024**5) == '1 PB'


def test_decimal_formatting():
    """Test that fractional sizes render with decimal places."""
    # 1.5 KB = 1536 bytes
    assert human_readable_byte_size(1536) == '1.5 KB'


def test_trailing_zeros_stripped():
    """Test that trailing zeros are stripped from formatted output."""
    # 1.00 MB should render as "1 MB", not "1.00 MB"
    result = human_readable_byte_size(1024**2)
    assert '.00' not in result
    assert result == '1 MB'
