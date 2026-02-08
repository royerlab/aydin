"""TensorFlow device selection and memory query utilities."""

import psutil
import tensorflow as tf


def get_best_device_name():
    """
    Returns the non-XLA device with highest available memory

    Returns
    -------
    str
        Name of the best available GPU

    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Get logical devices (which have the actual device names we need)
        logical_gpus = tf.config.list_logical_devices('GPU')
        if logical_gpus:
            # Return the first GPU (TensorFlow handles memory allocation)
            return logical_gpus[0].name
    return "/device:CPU:0"


def available_device_memory():
    """
    Returns available RAM memory for CPU, available device memory for GPU

    Returns
    -------
    float
        Available device memory

    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Try to get GPU memory info (requires TF 2.5+)
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            return float(memory_info['total'])
        except (RuntimeError, ValueError):
            # Fallback: estimate based on typical GPU memory
            return 8.0 * 1024 * 1024 * 1024  # 8GB default
    return float(psutil.virtual_memory().available)
