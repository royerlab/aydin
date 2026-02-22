"""Memoization utilities supporting NumPy arrays.

This subpackage provides a single-entry cache decorator that works with
unhashable types like NumPy arrays by using object identity for cache
key comparison.
"""
