import psutil

# from tensorflow_core.python.client import device_lib
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_best_device_name():
    """
    Returns the non-XLA device with highest available memory

    Returns
    -------
    str
        Name of the best available GPU

    """

    if tf.test.gpu_device_name() != "":
        available_gpu_devices = []
        for device in device_lib.list_local_devices():
            if "GPU" in device.name and "XLA" not in device.name:
                available_gpu_devices.append(device)
        available_memory_amounts = [
            device.memory_limit for device in available_gpu_devices
        ]
        best_gpu_index = available_memory_amounts.index(max(available_memory_amounts))
        return available_gpu_devices[best_gpu_index].name
    else:
        return "/device:CPU:0"


def available_device_memory():
    """
    Returns available RAM memory for CPU, available device memory for GPU

    Returns
    -------
    float
        Available device memory

    """
    for device in device_lib.list_local_devices():
        if tf.test.gpu_device_name() != "" and device.name == tf.test.gpu_device_name():
            return float(device.memory_limit)
    else:
        return float(psutil.virtual_memory().available)
