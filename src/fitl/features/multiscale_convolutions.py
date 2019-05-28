from __future__ import absolute_import, print_function

import numpy as np
import pyopencl as cl
from pyopencl.array import to_device, Array

from fitl.opencl.opencl_provider import OpenCLProvider


class MultiscaleConvolutionalFeatures:
    """
    Multiscale convolutional feature generator.
    Uses OpenCL to acheive very fast feature generation.

    TODO: How to handle the situation in which we want the features array to be stored on disk (dask, zarr, mem_map).
    TODO: add 1D, and 4D (for channels)
    """

    def __init__(self,
                 opencl_provider=OpenCLProvider(),
                 kernel_widths=[5, 3, 3, 3],
                 kernel_scales=[1, 3, 5, 7],
                 exclude_center=False
                 ):
        """
        Constructs a multiscale convolutional feature generator that uses OpenCL.

        :param opencl_provider: 
        :type opencl_provider: 
        :param kernel_widths: 
        :type kernel_widths: 
        :param kernel_scales: 
        :type kernel_scales: 
        :param exclude_center: 
        :type exclude_center: 
        """
        self.check_nans = False
        self.debug_log = False

        self.opencl_provider = opencl_provider
        self.queue = opencl_provider.queue
        self.context = opencl_provider.context

        self.kernel_widths = kernel_widths
        self.kernel_scales = kernel_scales
        self.exclude_center = exclude_center

    def compute(self, image, features=None):
        """
        Computes the features given an image. If the input image is of shape (d,h,w),
        resulting features are of shape (d,h,w,n) where n is the number of features.
        :param image: 
        :type image: 
        :return: 
        :rtype: 
        """
        image = image.astype(np.float32)

        # Checking NaNs just in case:
        if self.check_nans and np.isnan(np.sum(image)):
            raise Exception(f'NaN values occur in image!')

        image_dimension = len(image.shape)

        # We move the image to the GPU. Needs to fit entirely, could be a problem for very very large images.
        image_gpu = to_device(self.queue, image)

        # This array on the GPU will host a single feature.
        # We will use that as temp destination for each feature generated on the GPU.
        feature_gpu = Array(self.queue, image_gpu.shape, np.float32)

        # Switching for different number of dimensions,
        # then we first compute the nb of features, allocate the corrected sized numpy array,
        # and then we compute the features...
        if image_dimension == 2:
            nb_features = self.collect_features_2D(image_gpu)
            features = np.zeros((nb_features,) + image.shape, dtype=np.float32)
            features = self.collect_features_2D(image_gpu, feature_gpu, features)

        elif image_dimension == 3:
            nb_features = self.collect_features_3D(image_gpu)
            features = np.zeros((nb_features,) + image.shape, dtype=np.float32)
            features = self.collect_features_3D(image_gpu, feature_gpu, features)

        # We currently only support 2D and 3D. TODO: add 1D and 4D (for channels)
        elif image_dimension > 3:
            raise Exception(f'dimension above {image_dimension} not yet implemented!')

        # Creates a view of the array in which the features are indexed by the last dimension:
        features = np.moveaxis(features, 0, -1)

        return features

    def collect_features_2D(self, image_gpu, feature_gpu=None, features=None):
        """
        Computes 2D features, one by one.

        :param image_gpu:
        :type image_gpu:
        :param feature_gpu:
        :type feature_gpu:
        :param features:
        :type features:
        :return:
        :rtype:
        """
        if self.debug_log:
            if features is None:
                print(f"Counting the number of features...")
            else:
                print(f"Computing features...")

        feature_index = 0
        for width, scale in zip(self.kernel_widths, self.kernel_scales):
            radius = width // 2
            for i in range(-radius, +radius + 1):
                for j in range(-radius, +radius + 1):

                    if self.exclude_center and scale == 1 and i == 0 and j == 0:
                        continue

                    if features is not None:
                        if self.debug_log:
                            print(f"(width={width}, scale={scale}, i={i}, j={j})")
                        self.compute_feature_2d_avg(image_gpu, feature_gpu, i * scale, j * scale, scale, scale, self.exclude_center)
                        # features[feature_index] = feature_gpu.get()
                        cl.enqueue_copy(self.queue, features[feature_index], feature_gpu.data)

                        if self.check_nans and np.isnan(np.sum(features[feature_index])):
                            print(features[feature_index])
                            raise Exception(f'NaN values occur in features!')

                    feature_index += 1

        if features is not None:
            return features
        else:
            return feature_index

    def collect_features_3D(self, image_gpu, feature_gpu=None, features=None):
        """
        Computes 3D features, one by one.
        :param image_gpu:
        :type image_gpu:
        :param feature_gpu:
        :type feature_gpu:
        :param features:
        :type features:
        :return:
        :rtype:
        """
        if self.debug_log:
            if features is None:
                print(f"Counting the number of features...")
            else:
                print(f"Computing features...")

        feature_index = 0

        for width, scale in zip(self.kernel_widths, self.kernel_scales):
            radius = width // 2
            for i in range(-radius, +radius + 1):
                for j in range(-radius, +radius + 1):
                    for k in range(-radius, +radius + 1):
                        if self.exclude_center and scale == 1 and i == 0 and j == 0 and k == 0:
                            continue

                        if features is not None:
                            if self.debug_log:
                                print(f"(width={width}, scale={scale}, i={i}, j={j}, k={k})")
                            self.compute_feature_3d_avg(image_gpu, feature_gpu, i * scale, j * scale, k * scale, scale, scale, scale, self.exclude_center)
                            # features[feature_index] = feature_gpu.get()
                            cl.enqueue_copy(self.queue, features[feature_index], feature_gpu.data)

                            if self.check_nans and np.isnan(np.sum(features[feature_index])):
                                raise Exception(f'NaN values occur in features!')

                        feature_index += 1

        if features is not None:
            return features
        else:
            return feature_index

    def compute_feature_2d_avg(self, image_gpu, feature_gpu, dx, dy, w, h, exclude_center=True):
        """
        Compute a given feature for a displacement (dx,dy) relative to teh center pixel, and a patch size (w,h)

        :param image_gpu:
        :type image_gpu:
        :param feature_gpu:
        :type feature_gpu:
        :param dx:
        :type dx:
        :param dy:
        :type dy:
        :param w:
        :type w:
        :param h:
        :type h:
        :param exclude_center:
        :type exclude_center:
        """
        image_width = image_gpu.shape[0]
        image_height = image_gpu.shape[1]

        rx = w // 2
        ry = h // 2

        program_code = f"""
        __kernel void feature_kernel(__global float *image, __global float *feature)
        {{
            int fx = get_global_id(0);
            int fy = get_global_id(1);

            float sum  =0.0f;
            int   count=0;

            for(int j={dy-ry}; j<={dy+ry}; j++)
            {{
                int y = fy+j;
                y = y<0 ? 0 : y;
                y = y>={image_height} ? {image_height-1} : y;
                
                for(int i={dx-rx}; i<={dx+rx}; i++)
                {{
                    int x = fx+i;
                    x = x<0 ? 0 : x;
                    x = x>={image_width} ? {image_width-1} : x;
                    
                    //printf("(%d,%d)\\n", x, y);
                    
                    int image_index = x + y * {image_width};
                    float value = image[image_index];
                    if ({'true' if exclude_center else 'false'}  && fx==x && fy==y)
                        continue;
                        
                    //printf("%d\\n", image_index);
                    //printf("%f\\n", value);
                    
                    sum+=value; 
                    count++;
                }}
            }}
            
            int feature_index = fx + fy * {image_width};
            float value = sum/count;
            feature[feature_index] = isnan(value) ? 0 : value;
        }}
        """
        # print(program_code)

        program = cl.Program(self.context, program_code).build()

        feature_kernel = program.feature_kernel

        feature_kernel(self.queue, image_gpu.shape, None, image_gpu.data, feature_gpu.data)

    def compute_feature_3d_avg(self, image_gpu, feature_gpu, dx, dy, dz, w, h, d, exclude_center=True):

        image_width = image_gpu.shape[0]
        image_height = image_gpu.shape[1]
        image_depth = image_gpu.shape[2]

        rx = w // 2
        ry = h // 2
        rz = d // 2

        program_code = f"""
        __kernel void feature_kernel(__global float *image, __global float *feature)
        {{
            int fx = get_global_id(0);
            int fy = get_global_id(1);
            int fz = get_global_id(2);

            float sum  =0.0f;
            int   count=0;

            for(int k={dz-rz}; k<={dz+rz}; k++)
            {{
                int z = fz+k;
                z = z<0 ? 0 : z;
                z = z>={image_depth}  ? {image_depth -1} : z;
                
                for(int j={dy-ry}; j<={dy+ry}; j++)
                {{
                    int y = fy+j;
                    y = y<0 ? 0 : y;
                    y = y>={image_height} ? {image_height-1} : y;
                    
                    for(int i={dx-rx}; i<={dx+rx}; i++)
                    {{
                        int x = fx+i;
                        x = x<0 ? 0 : x;
                        x = x>={image_width}  ? {image_width -1} : x;
                        
                        //printf("(%d, %d, %d)\\n", x, y, z);
                        
                        int image_index = x + y * {image_width} + z * {image_width*image_height};
                        float value = image[image_index];
                        if ({'true' if exclude_center else 'false'}  && fx==x && fy==y && fz==z)
                            continue;
    
                        //printf("%d\\n", image_index);
                        //printf("%f\\n", value);
    
                        sum+=value; 
                        count++;
                    }}
                }}
            }}

            int feature_index = fx + fy * {image_width} + fz * {image_width*image_height};
            float value = sum/count;
            feature[feature_index] = isnan(value) ? 0 : value;
        }}
        """
        # print(program_code)

        program = cl.Program(self.context, program_code).build()

        feature_kernel = program.feature_kernel

        feature_kernel(self.queue, image_gpu.shape, None, image_gpu.data, feature_gpu.data)
