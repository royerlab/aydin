import itertools
import os

import pyopencl as cl
from pyopencl._cl import get_platforms


class OpenCLProvider:

    def __init__(self, includes=[], excludes=['CPU']):

        os.environ['PYOPENCL_NO_CACHE'] = '1'

        for device in self.get_all_devices():
            print(f'device: {device} with mem: {device.global_mem_size}')

        self.devices = self.get_filtered_device_list(includes, excludes)
        print(f"filtered devices: {self.devices}")

        self.device = self.devices[0]
        print(f"selected device: {self.device}")

        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        self.program_cache = {}


    def get_all_devices(self):
        return list(itertools.chain.from_iterable([platform.get_devices() for platform in get_platforms()]))

    def get_filtered_device_list(self, includes=[], excludes=[], sort_by_mem_size=True):
        valid_devices = []

        devices = self.get_all_devices()
        # print(platforms)

        for exclude in excludes:
            devices = [device for device in devices if not exclude in device.name]

        for include in includes:
            devices = [device for device in devices if include in device.name]

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



