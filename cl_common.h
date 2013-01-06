#ifndef CL_COMMON_H_8IA0EQYE
#define CL_COMMON_H_8IA0EQYE

#define __CL_ENABLE_EXCEPTIONS
#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <array>
#include <type_traits>

/**
 * Some missing traits from OpenCL C++ headers
 */
namespace cl
{

    namespace detail
    {

        struct cl_image_info;
        template<>
        struct param_traits<detail::cl_image_info, CL_IMAGE_ARRAY_SIZE>
        {
            enum { value = CL_IMAGE_ARRAY_SIZE };
            typedef ::size_t param_type;
        };

    }

}

namespace DynamiCL
{
    char const* clErrorToStr(cl_int err);

    /**
     * Initializes the necessary handles to run OpenCL computations
     */
    struct ComputeContext
    {
        cl::Device const device;
        cl::Context const context;
        cl::CommandQueue const queue;

        ComputeContext();
    };

    /**
     * Gathers some useful information on the
     * capabilities of a device.
     */
    struct DeviceCapabilities
    {
        cl::detail::param_traits<cl::detail::cl_device_info, CL_DEVICE_GLOBAL_MEM_SIZE>::param_type
            const memSize;
        cl::detail::param_traits<cl::detail::cl_device_info, CL_DEVICE_MAX_MEM_ALLOC_SIZE>::param_type
            const maxAllocSize;

        DeviceCapabilities(cl::Device device);
    };

    /* Create program from a file and compile it */
    cl::Program buildProgram(cl::Context const& ctx, cl::Device dev, char const* filename);

    /***************************************************************************
     *                           cl::Vector helpers                            *
     ***************************************************************************/

    namespace detail
    {

        template<typename Vec>
        void vector_constructor_impl(Vec& vec)
        { }

        // only works for non-array arguments
        template<typename Vec, typename U, typename... Us>
        void vector_constructor_impl(Vec& vec, U&& first, Us&&... rest)
        {
            vec.push_back(std::forward<U>(first));
            vector_constructor_impl(vec, std::forward<Us>(rest)...);
        }

    }

    template <typename T>
    struct VectorConstructor
    {

        template <typename... Ts>
        static cl::vector<T, sizeof...(Ts)>
        construct(Ts&&... args)
        {
            cl::vector<T, sizeof...(Ts)> vec;
            detail::vector_constructor_impl(vec, std::forward<Ts>(args)...);
            return vec;
        }

    };

    template <>
    struct VectorConstructor<typename ::size_t>
    {

        template <typename... Ts>
        static cl::size_t<sizeof...(Ts)>
        construct(Ts&&... args)
        {
            cl::size_t<sizeof...(Ts)> vec;
            detail::vector_constructor_impl(vec, std::forward<Ts>(args)...);
            return vec;
        }

    };

    /***************************************************************************
     *                           cl::Image helpers                             *
     ***************************************************************************/

    namespace detail
    {
        /**
         * Traits to determine types/operations
         * that are possible for an OpenCL image
         */
        template <typename CLImage> struct image_traits;

        template <>
        struct image_traits<cl::Image1D>
        {
            static const size_t N = 1;
            static const bool is_array = false;
            static const cl_mem_object_type mem_type = CL_MEM_OBJECT_IMAGE1D;
            typedef cl::Image1D climage_type;

            static constexpr cl_int dim_info[N] =
                { CL_IMAGE_WIDTH };
        };

        template <>
        struct image_traits<cl::Image1DArray>
        {
            static const size_t N = 2;
            static const bool is_array = true;
            static const cl_mem_object_type mem_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
            typedef cl::Image1DArray climage_type;

            static constexpr cl_int dim_info[N] =
                { CL_IMAGE_WIDTH, CL_IMAGE_ARRAY_SIZE };
        };

        template <>
        struct image_traits<cl::Image2D>
        {
            static const size_t N = 2;
            static const bool is_array = false;
            static const cl_mem_object_type mem_type = CL_MEM_OBJECT_IMAGE2D;
            typedef cl::Image2D climage_type;

            static constexpr cl_int dim_info[N] =
                { CL_IMAGE_WIDTH, CL_IMAGE_HEIGHT };
        };

        template <>
        struct image_traits<cl::Image2DArray>
        {
            static const size_t N = 3;
            static const bool is_array = true;
            static const cl_mem_object_type mem_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            typedef cl::Image2DArray climage_type;

            static constexpr cl_int dim_info[N] =
                { CL_IMAGE_WIDTH, CL_IMAGE_HEIGHT, CL_IMAGE_ARRAY_SIZE };
        };

        template <>
        struct image_traits<cl::Image3D>
        {
            static const size_t N = 3;
            static const bool is_array = false;
            static const cl_mem_object_type mem_type = CL_MEM_OBJECT_IMAGE3D;
            typedef cl::Image3D climage_type;

            static constexpr cl_int dim_info[N] =
                { CL_IMAGE_WIDTH, CL_IMAGE_HEIGHT, CL_IMAGE_DEPTH };
        };


        template <typename CLImage>
        CLImage
        construct_image(cl::Context const& context,
                        std::array<size_t, image_traits<CLImage>::N> const& dims,
                        cl_mem_flags flags,
                        void* host_ptr)
        {
            static constexpr size_t N = image_traits<CLImage>::N;
            static_assert( (N >= 1) && (N <= 3), "Image dimensions must be between 1 and 3." );

            cl_image_desc desc;

            desc.image_type = image_traits<CLImage>::mem_type;
            desc.image_width = dims[0];

            desc.image_height = 0;
            if ( N > 1 )
            {
                if (image_traits<CLImage>::is_array
                    && N == 2)
                {
                    desc.image_array_size = dims[1];
                }
                else 
                {
                    desc.image_height = dims[1];
                }
            }

            desc.image_depth = 0;
            if ( N > 2 )
            {
                if (image_traits<CLImage>::is_array
                    && N == 3)
                {
                    desc.image_array_size = dims[2];
                }
                else 
                {
                    desc.image_depth = dims[2];
                }
            }

            desc.image_row_pitch = 0;
            desc.image_slice_pitch = 0;
            desc.num_mip_levels = 0;
            desc.num_samples = 0;
            desc.buffer = 0;

            cl::ImageFormat format = cl::ImageFormat(CL_RGBA, CL_FLOAT);

            cl_int error;
            cl_mem mem = ::clCreateImage(
                context(), 
                flags, 
                &format, 
                &desc, 
                host_ptr, 
                &error);

            ::cl::detail::errHandler(error, "clCreateImage");
            return CLImage(mem);
        }

    }

    template <typename CLImage>
    typename detail::image_traits<CLImage>::climage_type
    createCLImage(ComputeContext const& c,
                  std::array<size_t, detail::image_traits<CLImage>::N> const& dims,
                  void* hostPtr = nullptr)
    {
        cl_mem_flags flags = CL_MEM_READ_WRITE;
        if (!hostPtr)
        {
            flags |= CL_MEM_HOST_READ_ONLY;
        }
        else
        {
            flags |= CL_MEM_COPY_HOST_PTR;
        }

        return detail::construct_image<CLImage>(c.context, dims, flags, hostPtr);
    }

    namespace detail
    {
        template <typename CLImage, size_t END>
        void get_dims_impl(CLImage const& image,
                std::array<size_t, END>& dims,
                std::integral_constant<size_t, END>,
                std::integral_constant<size_t, END>)
        {
            static const size_t N = detail::image_traits<CLImage>::N;
            typedef CLImage climage_type;

            static_assert( END == N, "Iteration length has to be equal to image dimension.");
        }

        /**
         * Compile time iteration to initialize std::array
         * has to be done at compile-time since image could have any dimension,
         * and compilation fails if we ask for an inexising dimension from it.
         * E.g. Depth from an Image1D
         */
        template <typename CLImage, size_t N, size_t M, size_t END>
        void get_dims_impl(CLImage const& image,
                std::array<size_t, N>& dims,
                std::integral_constant<size_t, M>,
                std::integral_constant<size_t, END>)
        {
            typedef CLImage climage_type;

            static_assert( detail::image_traits<CLImage>::N == N,
                           "Array size must match dimensionality of image.");
            static_assert( END == N, "Iteration length has to be equal to image dimension.");
            static_assert( M < END, "Iteration index must not exceed image dimension.");

            // perform actual assignment
            dims[M] = image.template getImageInfo<
                        detail::image_traits<climage_type>::dim_info[M]
                    >();

            // continue iteration
            get_dims_impl(image, dims,
                    std::integral_constant<size_t, M+1>(),
                    std::integral_constant<size_t, END>());
        }

    }

    /**
     * Return OpenCL image dimensions in a std::array
     */
    template <typename CLImage>
    std::array<size_t, detail::image_traits<CLImage>::N>
    getDims(CLImage const& image)
    {
        static const size_t N = detail::image_traits<CLImage>::N;
        typedef CLImage climage_type;

        std::array<size_t, N> dims;

        detail::get_dims_impl(image, dims,
                    std::integral_constant<size_t, 0>(),
                    std::integral_constant<size_t, N>());

        //for (size_t i = 0; i < N; ++i)
        //{
            //dims[i] = image.template getImageInfo<
                        //detail::image_traits<climage_type>::dim_info[i]
                    //>();
        //}

        //if ( N > 1 )
        //{
            //dims[1] = image.template getImageInfo<
                        //detail::image_traits<climage_type>::height_info
                    //>();
        //}

        //if ( N > 2 )
        //{
            //dims[2] = image.template getImageInfo<
                        //detail::image_traits<climage_type>::depth_info
                    //>();
        //}

        return dims;
    }

    inline cl::NDRange toNDRange(std::array<size_t, 1> dims)
    {
        return cl::NDRange(dims[0]);
    }

    inline cl::NDRange toNDRange(std::array<size_t, 2> dims)
    {
        return cl::NDRange(dims[0], dims[1]);
    }

    inline cl::NDRange toNDRange(std::array<size_t, 3> dims)
    {
        return cl::NDRange(dims[0], dims[1], dims[2]);
    }

    /**
     * Take an array and fills a cl::size_t<3> vector with its values, and any
     * remaining spots are filled with @a fillValue.
     *
     * Useful for enqueueReads in OpenCL.
     */
    template <size_t N>
    inline cl::size_t<3> toSizeVector(std::array<size_t, N> const& dims, size_t fillValue = 0)
    {
        static_assert( N <= 3, "Array dimensions must not exceed 3." );

        cl::size_t<3> vec;
        // TODO: OpenCL cpp headers don't have the right typedefs
        // to be able to use stl algorithms, so have to do things manually...
        size_t i = 0;
        for (; i < N; ++i)
        {
            vec.push_back(dims[i]);
        }
        for (; i < 3; ++i)
        {
            vec.push_back(fillValue);
        }

        return vec;
    }

}

#endif /* end of include guard: CL_COMMON_H_8IA0EQYE */
