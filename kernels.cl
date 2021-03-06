// try mirrored?
__constant const sampler_t g_sampler =
        CLK_FILTER_NEAREST|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;

/***************************************************************************
 *                            Gaussian Kernels                             *
 ***************************************************************************/

// sampling kernel for laplacian/gaussian pyramids
__constant const float sampling_kernel[5] = {
    01.f/16.f, 04.f/16.f, 06.f/16.f, 04.f/16.f, 01.f/16.f
};

/*__attribute__(( vec_type_hint (float4)))*/
__kernel void downsample_row(__read_only image2d_t input_image, __write_only image2d_t output_image)
{
    int2 out_coord = (int2)( get_global_id(0), get_global_id(1) );
    int2 in_coord = (int2)( out_coord.x * 2, out_coord.y );

    float4 sample = 0.0f;
    for (int i = -2; i < 3; ++i)
    {
        sample += read_imagef (input_image, g_sampler, in_coord+(int2)(i, 0)) * sampling_kernel[2+i];
    }

    write_imagef (output_image, out_coord, sample);
}

/*__attribute__(( vec_type_hint (float4)))*/
__kernel void downsample_col(__read_only image2d_t input_image, __write_only image2d_t output_image)
{

    int2 out_coord = (int2)( get_global_id(0), get_global_id(1) );
    int2 in_coord = (int2)( out_coord.x, out_coord.y * 2 );

    float4 sample = 0.0f;
    for (int i = -2; i < 3; ++i)
    {
        sample += read_imagef (input_image, g_sampler, in_coord+(int2)(0, i)) * sampling_kernel[2+i];
    }

    write_imagef (output_image, out_coord, sample);
}

/**
 * Create two pixels at once, while upsamling.
 * Dimensions of problem are same as input image.
 * Corner case arises when output has odd num of pixels.
 */
__kernel void upsample_col(__read_only image2d_t input_image, __write_only image2d_t output_image)
{
    int2 out_dim = get_image_dim(output_image);

    int2 in_coord = (int2)( get_global_id(0), get_global_id(1) );
    int2 out_coord = (int2)( in_coord.x, in_coord.y * 2 );

    // the 3 input pixels that contribute to the two pixels in output
    float4 in0 = read_imagef (input_image, g_sampler, in_coord+(int2)(0, -1));
    float4 in1 = read_imagef (input_image, g_sampler, in_coord+(int2)(0, 0));
    float4 in2 = read_imagef (input_image, g_sampler, in_coord+(int2)(0, 1));

    // coefficients are derived from how much the output pixel
    // contributed before in the downsampling step
    float4 out0 = (   in0 * sampling_kernel[0]
                    + in1 * sampling_kernel[2]
                    + in2 * sampling_kernel[0])
                  * 2;

    float4 out1 = (   in1 * sampling_kernel[1]
                    + in2 * sampling_kernel[1])
                  * 2;

    write_imagef (output_image, out_coord, out0);

    // corner case when output has odd number of pixels.
    // do not write out the second output pixel
    // TODO investigate impact of divergence
    if ((out_dim.y % 2 != 0)
        && out_coord.y == out_dim.y - 1)
    {
        return;
    }

    write_imagef (output_image, out_coord+(int2)(0,1), out1);
}

/***************************************************************************
 *                         Pyramid Related Kernels                         *
 ***************************************************************************/

/**
 * Given the original image and the gaussian blurred image (upsampled from a
 * lower level of a pyramid) create the laplacian by subtracting the two.
 *
 * Importantly, the alpha channel (used for gaussian pyramid) is preserved as is.
 */
__kernel void create_laplacian(__read_only image2d_t original,
                               __read_only image2d_t blurred,
                               __write_only image2d_t laplacian)
{
    int2 coord = (int2)( get_global_id(0), get_global_id(1) );

    float4 o = read_imagef (original, g_sampler, coord);
    float4 b = read_imagef (blurred, g_sampler, coord);

    float4 l = o - b;
    // TODO remove experiment
    /*l += 0.5f;*/
    /*l.s3 = 1.0f;*/
    l.s3 = o.s3; // preserve original alpha;

    write_imagef (laplacian, coord, l);
}

__kernel void collapse_level( __read_only image2d_t blurred,
                              __read_only image2d_t laplacian,
                              __write_only  image2d_t collapsed)
{
    int2 coord = (int2)( get_global_id(0), get_global_id(1) );

    float4 b = read_imagef (blurred, g_sampler, coord);
    float4 l = read_imagef (laplacian, g_sampler, coord);

    float4 c = b + l;
    // TODO Shouldn't need this?
    c = clamp(c, 0.0f, 1.0f);

    write_imagef (collapsed, coord, c);
}

__kernel void fuse_level( __read_only  image2d_array_t array,
                          __write_only image2d_t fused)
{
    int2 coord = (int2)( get_global_id(0), get_global_id(1) );
    int depth = get_image_array_size(array);

    float4 acc = 0.0f;
    float weight_sum = 0.0f; // sum of all weights in alpha channel

    for (int i = 0; i < depth; ++i)
    {
        int4 array_coord = (int4)(coord.x, coord.y, i, 0);
        float4 pix = read_imagef (array, g_sampler, array_coord);

        // TODO what if all weights are zero for pixel?

        // accumulate weight
        weight_sum += pix.s3;

        // multiply by own weight
        pix  *= pix.s3; 

        acc += pix;
    }

    // divide by weight sum to normalize
    acc /= weight_sum;

    write_imagef (fused, coord, acc);
}

/***************************************************************************
 *                          HDR Quality Measures                           *
 ***************************************************************************/

__kernel void upsample_row(__read_only image2d_t input_image, __write_only image2d_t output_image)
{
    int2 out_dim = get_image_dim(output_image);

    int2 in_coord = (int2)( get_global_id(0), get_global_id(1) );
    int2 out_coord = (int2)( in_coord.x * 2, in_coord.y);

    // the 3 input pixels that contribute to the two pixels in output
    float4 in0 = read_imagef (input_image, g_sampler, in_coord+(int2)(-1, 0));
    float4 in1 = read_imagef (input_image, g_sampler, in_coord+(int2)(0, 0));
    float4 in2 = read_imagef (input_image, g_sampler, in_coord+(int2)(1, 0));

    // coefficients are derived from how much the output pixel
    // contributed before in the downsampling step
    float4 out0 = (   in0 * sampling_kernel[0]
                    + in1 * sampling_kernel[2]
                    + in2 * sampling_kernel[0])
                  * 2;

    float4 out1 = (   in1 * sampling_kernel[1]
                    + in2 * sampling_kernel[1])
                  * 2;

    write_imagef (output_image, out_coord, out0);

    // corner case when output has odd number of pixels.
    // do not write out the second output pixel
    // TODO investigate impact of divergence
    if ((out_dim.x % 2 != 0)
        && out_coord.x == out_dim.x - 1)
    {
        return;
    }

    write_imagef (output_image, out_coord+(int2)(1,0), out1);
}

/***************************************************************************
 *                          HDR Quality Measures                           *
 ***************************************************************************/

/**
 * Return the standard deviation squared, of R, G, B component values in a pixel
 */
inline float sigma_squared_rgb(float4 pixel)
{
    float4 squared = pown(pixel, 2);
    float mean = ((float)pixel.s0 + pixel.s1 + pixel.s2) / 3.0f;
    float mean_squared = pown(mean, 2);
    float mean_of_squared = ((float)squared.s0 + squared.s1 + squared.s2) / 3.0f;

    // TODO due to floating point instability, answer can become negative
    // if all values are equal. Find more stable algo.

    return sqrt(fabs(mean_of_squared - mean_squared));
}

inline float well_exposedness(float4 pixel)
{
    // TODO looks like shit

    // 1961 paper "Acosine approximation to the normal distribution"
    // by D. H. Raab and E. H. Green, Psychometrika, Volume 26, pages 447-450
    float4 component_wise = 0.5f + cospi( 1.75f * (pixel - 0.5f) );

    // create single measure for pixel
    return component_wise.s0 + component_wise.s1 + component_wise.s2;
}

// uses gaussian distribution
inline float well_exposedness_naive(float4 pixel)
{
    float const denominator = 0.08f; // sigma^2 * 2, where sigma = 0.2
    float4 component_wise = exp( (float4) - (pown( pixel - 0.5f, 2 ) / denominator) );

    // create single measure for pixel
    return component_wise.s0 + component_wise.s1 + component_wise.s2;
}

// part of quality measure.
__constant const float discreet_laplacian[3][3] = {
    0.5f/6.f, 1.f/6.f, 0.5f/6.f,
     1.f/6.f,   -1.f,   1.f/6.f,
    0.5f/6.f, 1.f/6.f, 0.5f/6.f
};

__kernel void compute_quality(__read_only image2d_t input_image, __write_only image2d_t output_image)
{
    int2 coord = (int2)( get_global_id(0), get_global_id(1) );

    // find laplacian at pixel per component
    float4 laplacian = 0.0f;
    for (int i = -1; i < 2; ++i)
    {
        for (int j = -1; j < 2; ++j)
        {
            laplacian += read_imagef (input_image, g_sampler, coord+(int2)(i, j) )
                             * discreet_laplacian[1+i][1+j];
        }
    }

    // laplacian of alpha channel will be zero.
    // to average across channels, just get length of vector
    // TODO benefits to fast_length?
    float laplacian_measure = fast_length(fabs(laplacian));

    float4 pixel = read_imagef (input_image, g_sampler, coord);

    float sigma = sigma_squared_rgb(pixel);
    float exposedness = well_exposedness_naive(pixel);

    // assign quality measure to alpha channel
    // TODO multiples are ad hoc. do better
    pixel.s3 = ( laplacian_measure * 3.0f )
             + ( sigma * 1.5f )
             + ( exposedness * 0.2f );

    write_imagef (output_image, coord, pixel);
}

__kernel void compute_quality_bal(__read_only image2d_t input_image, __write_only image2d_t output_image)
{
    int2 coord = (int2)( get_global_id(0), get_global_id(1) );
    float4 pixel = read_imagef (input_image, g_sampler, coord);

    //float sigma = sigma_squared_rgb(pixel);
    float exposedness = well_exposedness(pixel);

    // assign quality measure to alpha channel
    // TODO multiples are ad hoc. do better
    pixel.s3 = well_exposedness(pixel) * 0.2f;
    //         + ( sigma * 1.5f )
    //         + ( exposedness * 0.2f );

    write_imagef (output_image, coord, pixel);
}

__kernel void compute_quality_sigma(__read_only image2d_t input_image, __write_only image2d_t output_image)
{
    int2 coord = (int2)( get_global_id(0), get_global_id(1) );
    float4 pixel = read_imagef (input_image, g_sampler, coord);

    float sigma = sigma_squared_rgb(pixel);

    // assign quality measure to alpha channel
    // TODO multiples are ad hoc. do better
    pixel.s3 +=( sigma * 1.5f );
    //         + ( exposedness * 0.2f );

    write_imagef (output_image, coord, pixel);
}
