# from tensorflow.python.client import device_lib
# import tensorflow as tf
from aydin.util.tf.device import available_device_memory

if __name__ == '__main__':
    # print(device_lib.list_local_devices())
    # print(tf.test.gpu_device_name())
    print(available_device_memory())
