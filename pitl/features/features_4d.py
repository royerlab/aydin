
def compute_feature_4d(generator, image_gpu, feature_gpu, dx, dy, dz, dw, lx, ly, lz, lw, exclude_center=True, reduction='sum'):
    """
    Compute a given feature for a displacement (dx,dy,dz,dw) relative to the center pixel, and a patch size (lx,ly,lz,lw)

    """
    image_x  = image_gpu.shape[3]
    image_y  = image_gpu.shape[2]
    image_z  = image_gpu.shape[1]
    image_w  = image_gpu.shape[0]

    rx = lx // 2
    ry = ly // 2
    rz = lz // 2
    rw = lw // 2

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
          int fx = get_global_id(2);
          int fy = get_global_id(1);
          int fz = get_global_id(0);
          
          for(int fw = 0; fw<{image_w}; fw++)
          {{
    
              float acc  =0.0f;
              int   count=0;
              
              for(int l={dw-rw}; l<={dw+rw}; l++)
              {{
                  int w = fw+l;
                  w = w<0 ? 0 : w;
                  w = w>={image_w}  ? {image_w -1} : w;
    
                  for(int k={dz-rz}; k<={dz+rz}; k++)
                  {{
                      int z = fz+k;
                      z = z<0 ? 0 : z;
                      z = z>={image_z}  ? {image_z -1} : z;
        
                      for(int j={dy-ry}; j<={dy+ry}; j++)
                      {{
                          int y = fy+j;
                          y = y<0 ? 0 : y;
                          y = y>={image_y} ? {image_y-1} : y;
        
                          for(int i={dx-rx}; i<={dx+rx}; i++)
                          {{
                              int x = fx+i;
                              x = x<0 ? 0 : x;
                              x = x>={image_x}  ? {image_x -1} : x;
        
                              //printf("(%d, %d, %d, %d)\\n", x, y, z, w);
        
                              int image_index = x + y * {image_x} + z * {image_x*image_y} + w * {image_x*image_y*image_z};
                              float value = image[image_index];
        
                              {'if (fx==x && fy==y && fz==z && fw==w) continue;' if exclude_center else ''}
        
                              //printf("%d\\n", image_index);
                              //printf("%f\\n", value);
        
                              acc = {reduction}_reduction(acc,value); 
                              count++;
                          }}
                      }}
                  }}
              }}
    
              int feature_index = fx + fy * {image_x} + fz * {image_x*image_y} + fw * {image_x*image_y*image_z};
              float value = acc/count;
              feature[feature_index] = isnan(value) ? 0 : value;
          }}
      }}
      """
    # print(program_code)

    program = generator.opencl_provider.build(program_code)

    feature_kernel = program.feature_kernel

    feature_kernel(generator.opencl_provider.queue, image_gpu.shape[1:], None, image_gpu.data, feature_gpu.data)
