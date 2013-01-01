#ifndef CL_ERRORS_H_GTNVZIRP
#define CL_ERRORS_H_GTNVZIRP

#define __CL_ENABLE_EXCEPTIONS
#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <functional>

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
     *                        Kernel and Image helpers                         *
     ***************************************************************************/
    
    /**
     * Represents an OpenCL kernel of particular program,
     * with some auxiliary information to help composition.
     *
     * Allows easy instantiations of kernels for multiple uses.
     */
    struct Kernel
    {
        enum class Range
        {
            SOURCE,
            DESTINATION
        };

        cl::Program const& program;
        char const* name;
        size_t const taps;
        Range const range;

        /**
         * Instantiate a new kernel with the given arguments
         */
        template <typename... Ts>
        cl::Kernel build(Ts&&... args) const
        {
            cl::Kernel kernel(program, name);
            build_impl(kernel, 0, std::forward<Ts>(args)...);
            return kernel;
        }

    private:
        template <typename T, typename... Ts>
        static void build_impl(cl::Kernel& kernel, size_t argIndex, T&& arg, Ts&&... rest)
        {
            kernel.setArg(argIndex, std::forward<T>(arg));
            build_impl(kernel, argIndex+1, std::forward<Ts>(rest)...);
        }

        // no arguments remaining
        static void build_impl(cl::Kernel&, size_t) { }
    };

    /**
     * Represents an image that is currently being processed.
     * think of it as "std::future<Image>" but for OpenCL
     */
    struct PendingImage
    {
        ComputeContext const& context;
        cl::Image2D image;
        std::vector<cl::Event> events; // TODO: const?

        PendingImage(PendingImage&& other)
            : context(other.context),
              image(std::move(other.image)),
              events(std::move(other.events))
        {
            other.image = cl::Image2D();
        }

        PendingImage& operator =(PendingImage&& other)
        {
            image = std::move(other.image);
            events = std::move(other.events);
            other.image = cl::Image2D();
        }

        PendingImage(ComputeContext const& c)
            : context(c),
              image(),
              events()
        { }

        PendingImage(ComputeContext const& c, cl::Image2D const& im)
            : context(c),
              image(im),
              events()
        { }

        PendingImage process(Kernel const& kernel, size_t width, size_t height);
        PendingImage process(Kernel const& kernel, cl::Image2D const& reuseImage);

        /**
         * Read image into host memory
         */
        void read(void* hostPtr);
    };

    /***************************************************************************
     *                             Pyramid Helpers                             *
     ***************************************************************************/
    
    /**
     * Represents an image pyramid, with methods to construct it from
     * a source image.
     *
     * Manages caching OpenCL images on the host, as well as chunking
     * them appropriately for operations to fit in GPU memory.
     */
    class ImagePyramid
    {
    public:
        /**
         * An image pair, of two levels of a pyramid
         */
        struct LevelPair
        {
            PendingImage upper;
            PendingImage lower;
        }

    private:
        // TODO: make these typedefs.
        // define a static function for fusing?
        /**
         * Creates a new level
         */
        std::function< LevelPair(PendingImage const&) >
            nextLevelFunc;

        /**
         * Collapses two levels
         */
        std::function< PendingImage(LevelPair const&) >
            collapseLevelFunc;

        /**
         * Fuses several pyramids at a single layer
         */
        std::function< PendingImage(std::vector<PendingImage> const&) >
            collapseLevelFunc;
    };

    /***************************************************************************
     *                           cl::Vector helpers                            *
     ***************************************************************************/

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
