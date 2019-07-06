from pitl.opencl.opencl_provider import OpenCLProvider


def test_opencl_provider():

    opencl = OpenCLProvider()

    assert len(opencl.get_all_devices())!=0

    for device in opencl.get_all_devices():
        print(device)