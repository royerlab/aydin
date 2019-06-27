import os

import pyopencl as cl
from pyopencl._cl import get_platforms


class OpenCLProvider:

    def __init__(self, includes=[], excludes=['CPU']):

        os.environ['PYOPENCL_NO_CACHE'] = '1'

        self.devices = self.get_filtered_device_list(includes, excludes)
        print(f"filtered devices: {self.devices}")

        selected_device = self.devices[1]
        print(f"selected device: {selected_device}")

        self.context = cl.Context([selected_device])
        self.queue = cl.CommandQueue(self.context)

        self.program_cache = {}

    def get_filtered_device_list(self, includes=[], excludes=[], sort_by_mem_size=True):
        valid_devices = []
        platforms = get_platforms()
        # print(platforms)
        for platform in platforms:
            devices = platform.get_devices()
            # print(devices)

            for exclude in excludes:
                devices = [device for device in devices if not exclude in device.name]

            for include in includes:
                devices = [device for device in devices if include in device.name]

            valid_devices += devices

        if sort_by_mem_size:
            devices = sorted(devices, key=lambda x: x.global_mem_size, reverse=True)

        print(devices)
        print([device.global_mem_size for device in devices])

        return list(devices)

    def build(self, program_code):

        if program_code in self.program_cache:
            return self.program_cache[program_code]
        else:
            program = cl.Program(self.context, program_code).build()
            self.program_cache[program_code] = program
            return program



