import re

import numpy
from m2cgen import interpreters
from pyopencl.array import Array

from aydin.providers.opencl.opencl_provider import OpenCLProvider
from aydin.regression.gbm_utils.light_gbm_assembler import LightGBMModelAssembler
from aydin.util.array.nd import nd_split_slices
from aydin.util.log.log import lsection, lprint
from aydin.util.misc.recursion import RecursionLimit


class GBMOpenCLPrediction:
    """
    Class that handles GBM OpenCL prediction.

    """

    def __init__(self, num_trees_per_kernel_call=100, max_gpu_length_array=6 * 1e6):

        """
        Constructs a GBM OpenCL inference instance
        """

        super().__init__()

        self.num_trees_per_kernel_call = num_trees_per_kernel_call
        self.max_gpu_length_array = max_gpu_length_array

        self.opencl_provider: OpenCLProvider = None
        self.x_gpu = None
        self.y_gpu = None

    def _ensure_opencl_prodider_initialised(self):
        if not hasattr(self, 'opencl_provider') or self.opencl_provider is None:
            self.opencl_provider = OpenCLProvider()

    def export_to_opencl_c(
        self, model, function_name='score', dtype='float', indent=4, tree_slice=None
    ):

        code = self.export_to_c(model, indent=indent, tree_slice=tree_slice)

        code = code.replace('double score(', 'inline double score(__global ')

        code = code.replace('score(', function_name + '(')

        code = code.replace('double', dtype)

        if dtype == 'float':
            code = re.sub(r'([\d]+\.[\d]+[-+e\d]*)', r'\1f', code)

        return code

    def export_to_c(self, model, indent=4, tree_slice=None):

        with RecursionLimit(3000):
            interpreter = interpreters.CInterpreter(indent=indent)
            model_ast = LightGBMModelAssembler(model, tree_slice=tree_slice).assemble()
            code = interpreter.interpret(model_ast)
            return code

    def predict(self, model, x, num_iteration=None, dtype='float', indent=4):

        with lsection("GBM OpenCL prediction:"):
            self._ensure_opencl_prodider_initialised()

            num_features = x.shape[-1]
            num_datapoints = x.shape[0]

            def get_program(tree_slice=slice(None)):
                forrest_code = self.export_to_opencl_c(
                    model, dtype=dtype, indent=indent, tree_slice=tree_slice
                )

                return f"""
                 {forrest_code}
                 __kernel void gbm_kernel(__global float *x, __global float *y)
                 {{
                   const int i = get_global_id(0);
                   const int x_offset = {num_features}*i ;
                   const int y_offset = i ;
                   y[y_offset] = score(x+x_offset);
                 }}
                 """

            x = numpy.ascontiguousarray(x, dtype=numpy.float32)

            gpu_array_length = min(int(self.max_gpu_length_array), num_datapoints)
            lprint(f"Proposed GPU array length: {gpu_array_length}")

            nb_array_slices = max(1, num_datapoints // gpu_array_length)
            array_slices = list(
                s[0] for s in nd_split_slices((num_datapoints,), (nb_array_slices,))
            )
            gpu_array_length = array_slices[0].stop - array_slices[0].start
            lprint(f"Effective GPU array length: {gpu_array_length}")

            if self.x_gpu is None or (
                self.x_gpu.shape != (gpu_array_length, num_features)
                and self.x_gpu.dtype != numpy.float32
            ):
                self.x_gpu = Array(
                    self.opencl_provider.queue,
                    (gpu_array_length, num_features),
                    numpy.float32,
                )

            if self.y_gpu is None or (
                self.y_gpu.shape != (gpu_array_length,)
                and self.y_gpu.dtype != numpy.float32
            ):
                self.y_gpu = Array(
                    self.opencl_provider.queue, (gpu_array_length,), numpy.float32
                )
                self.y_gpu.fill(0)

            num_trees = num_iteration
            nb_kernel_slices = max(1, num_trees // self.num_trees_per_kernel_call)
            kernel_slices = list(
                s[0] for s in nd_split_slices((num_iteration,), (nb_kernel_slices,))
            )

            lprint(f"Number of trees                : {num_trees}")
            lprint(f"Number of trees per kernel call: {self.num_trees_per_kernel_call}")
            lprint(f"Number of kernel slices        : {nb_kernel_slices}")

            y = numpy.zeros(num_datapoints)

            with lsection("Running kernels:"):
                for index, one_slice in enumerate(kernel_slices):
                    with lsection(
                        f"Kernel slice {index+1}/{len(kernel_slices)}: {one_slice}"
                    ):
                        with lsection("Generating kernel program code:"):
                            program_code = get_program(tree_slice=one_slice)
                            lprint(f"Kernel program length: {len(program_code)}")

                        program = self.opencl_provider.build(program_code)
                        gbm_kernel = program.gbm_kernel

                        with lsection(
                            f"Running kernel for {len(array_slices)} array slices"
                        ):
                            for one_array_slice in array_slices:
                                x_slice = x[one_array_slice]
                                x_slice = numpy.pad(
                                    x_slice,
                                    pad_width=(
                                        (0, self.x_gpu.shape[0] - x_slice.shape[0]),
                                        (0, 0),
                                    ),
                                    mode='constant',
                                )
                                self.x_gpu.set(x_slice, self.opencl_provider.queue)
                                # self.y_gpu.fill(0)

                                with lsection(
                                    f"Running kernel for array slice : {one_array_slice}"
                                ):
                                    gbm_kernel(
                                        self.opencl_provider.queue,
                                        (gpu_array_length,),
                                        None,
                                        self.x_gpu.data,
                                        self.y_gpu.data,
                                    )

                                    subarray_length = (
                                        one_array_slice.stop - one_array_slice.start
                                    )
                                    y[one_array_slice] += self.y_gpu.get(
                                        self.opencl_provider.queue
                                    )[0:subarray_length]

            del self.x_gpu
            del self.y_gpu

            return y
