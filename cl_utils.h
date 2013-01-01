#ifndef CL_ERRORS_H_GTNVZIRP
#define CL_ERRORS_H_GTNVZIRP

#define __CL_ENABLE_EXCEPTIONS
#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace DynamiCL
{

    char const* clErrorToStr(cl_int err);

    /**
     * Initializes the necessary handles to run OpenCL computations
     */
    struct ComputeContext
    {
        cl::Device device;
        cl::Context context;
        cl::CommandQueue queue;

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

    /**
     * Represents an OpenCL kernel of particular program,
     * with some auxiliary information to help composition.
     *
     * Allows easy instantiations of kernels for multiple uses.
     */
    struct Kernel
    {
        cl::Program& program;
        char const* name;
        size_t const taps;

        /**
         * Instantiate a new kernel with the given arguments
         */
        template <typename... Ts>
        cl::Kernel build(Ts&&... args)
        {
            cl::Kernel kernel(program, name);
            build_impl(kernel, 0, std::forward<Ts>(args)...);
            return kernel;
        }

    private:
        template <typename T, typename... Ts>
        void build_impl(cl::Kernel& kernel, size_t argIndex, T&& arg, Ts&&... rest)
        {
            kernel.setArg(argIndex, std::forward<T>(arg));
            build_impl(kernel, argIndex+1, std::forward<Ts>(rest)...);
        }

        // no arguments remaining
        void build_impl(cl::Kernel&, size_t) { }
    };

    namespace detail
    {
        template<typename Vec>
        void
        vector_constructor_impl(Vec& vec)
        { }

        template<typename Vec, typename U, typename... Us>
        void
        vector_constructor_impl(Vec& vec, U&& first, Us&&... rest)
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
}

#endif /* end of include guard: CL_ERRORS_H_GTNVZIRP */
