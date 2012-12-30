__kernel void add_numbers(__global float4* data, 
      __local float* local_result, __global float* group_result) {

   float sum;
   float4 input1, input2, sum_vector;
   uint global_addr, local_addr;

   global_addr = get_global_id(0) * 2;
   input1 = data[global_addr];
   input2 = data[global_addr+1];
   sum_vector = input1 + input2;

   local_addr = get_local_id(0);
   local_result[local_addr] = sum_vector.s0 + sum_vector.s1 + 
                              sum_vector.s2 + sum_vector.s3; 
   barrier(CLK_LOCAL_MEM_FENCE);

   if(get_local_id(0) == 0) {
      sum = 0.0f;
      for(int i=0; i<get_local_size(0); i++) {
         sum += local_result[i];
      }
      group_result[get_group_id(0)] = sum;
   }
}

__kernel void darken(__global float4* input, __global float4* output)
{
   uint global_addr = get_global_id(0);
   output[global_addr] = input[global_addr] / 2;
}

__kernel void darkenImage(__read_only image2d_t input_image, __write_only image2d_t output_image)
{
    // try mirrored?
    const sampler_t sampler =
        CLK_FILTER_NEAREST|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 coord = (int2)( get_global_id(0), get_global_id(1) );

    float4 val = read_imagef (input_image, sampler, coord) / 2;

    write_imagef (output_image, coord, val);
}

// sampling kernel for laplacian/gaussian pyramids
const float sampling_kernel[5] = {
    01.f/16.f, 04.f/16.f, 06.f/16.f, 04.f/16.f, 01.f/16.f
};

__kernel void downsample_row(__read_only image2d_t input_image, __write_only image2d_t output_image)
{
    // try mirrored?
    const sampler_t sampler =
        CLK_FILTER_NEAREST|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 coord = (int2)( get_global_id(0), get_global_id(1) );

    float4 sample = 0.0f;
    for (int i = -2; i < 3; ++i)
    {
        sample += read_imagef (input_image, sampler, coord+(int2)(i, 0)) * sampling_kernel[2+i];
    }

    write_imagef (output_image, coord, sample);
}

__kernel void downsample_col(__read_only image2d_t input_image, __write_only image2d_t output_image)
{

    // try mirrored?
    const sampler_t sampler =
        CLK_FILTER_NEAREST|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 coord = (int2)( get_global_id(0), get_global_id(1) );

    float4 sample = 0.0f;
    for (int i = -2; i < 3; ++i)
    {
        sample += read_imagef (input_image, sampler, coord+(int2)(0, i)) * sampling_kernel[2+i];
    }

    write_imagef (output_image, coord, sample);
}
