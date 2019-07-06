
def compute_feature_1d(generator, image_gpu, feature_gpu, dx, lx, exclude_center=True, reduction='sum'):
    """
    Compute a given feature for a displacement (dx) relative to the center pixel, and a patch size (lx)

    """
    image_x = image_gpu.shape[0]

    rx = lx // 2

    program_code = f"""

      inline float sum_reduction(float acc, float value)
      {{
          return acc + value;
      }}

      inline float max_reduction(float acc, float value)
      {{
          return max(acc , value);
      }}

      inline float min_reduction(float acc, float value)
      {{
          return min(acc , value);
      }}

      __kernel void feature_kernel(__global float *image, __global float *feature)
      {{
          int fx = get_global_id(0);

          float acc  =0.0f;
          int   count=0;

          for(int i={dx-rx}; i<={dx+rx}; i++)
          {{
              int x = fx+i;
              x = x<0 ? 0 : x;
              x = x>={image_x} ? {image_x-1} : x;

              //printf("(%d,%d)\\n", x, y);

              int image_index = x ;
              float value = image[image_index];
              
              {'if(fx==x) continue;' if exclude_center else ''}
                  
              //printf("%d\\n", image_index);
              //printf("%f\\n", value);

              acc = {reduction}_reduction(acc,value);  
              count++;
          }}
        

          int feature_index = fx;
          float value = acc/count;
          feature[feature_index] = isnan(value) ? 0 : value;
      }}
      """
    # print(program_code)

    program = generator.opencl_provider.build(program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(generator.opencl_provider.queue, image_gpu.shape, None, image_gpu.data, feature_gpu.data)

