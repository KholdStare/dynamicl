#ifndef CL_ERRORS_H_GTNVZIRP
#define CL_ERRORS_H_GTNVZIRP

#define __CL_ENABLE_EXCEPTIONS
#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <functional>
#include <memory>

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


        size_t width() const { return this->image.getImageInfo<CL_IMAGE_WIDTH>(); }
        size_t height() const { return this->image.getImageInfo<CL_IMAGE_HEIGHT>(); }

        PendingImage process(Kernel const& kernel, size_t width, size_t height);
        PendingImage process(Kernel const& kernel, cl::Image2D const& reuseImage);

        /**
         * Read image into host memory
         */
        void read(void* hostPtr) const;
        
    };

    /**
     * A simple RGBA pixel, of a particular component type
     */
    template <typename ComponentType>
    union RGBA
    {
        typedef ComponentType component_type;
        typedef component_type* iterator;
        typedef component_type const* const_iterator;

        struct
        {
            ComponentType r;
            ComponentType g;
            ComponentType b;
            ComponentType a;
        };
        component_type components[4];

        component_type&       operator[]( size_t i )       { return components[i]; }
        component_type const& operator[]( size_t i ) const { return components[i]; }
    };

    template <typename PixType>
    class HostImage
    {
        size_t width_;
        size_t height_;
        PixType* pixArray_;

        void dealloc()
        {
            delete[] pixArray_;
        }

        void invalidate()
        {
            width_ =  0;
            height_ = 0;
            pixArray_ = nullptr; // don't delete
        }

    public:
        typedef PixType* iterator;
        typedef PixType const* const_iterator;
        typedef PixType pixel_type;

        HostImage(size_t width, size_t height)
            : width_(width),
              height_(height),
              pixArray_(new PixType[width*height])
        { }

        ~HostImage() { dealloc(); }

        HostImage()
            : width_(0),
              height_(0),
              pixArray_(nullptr)
        { }

        // disable copying because expensive
        HostImage(HostImage const& other) = delete;
        HostImage& operator =(HostImage const& other) = delete;

        // move constructor
        HostImage(HostImage&& other)
            : width_(other.width_),
              height_(other.height_),
              pixArray_(other.pixArray_)
        {
            other.invalidate();
        }

        // move assignment
        HostImage& operator =(HostImage&& other)
        {
            width_ = other.width_;
            height_ = other.height_;
            pixArray_ = other.pixArray_;

            other.invalidate();
        }

        size_t width() const { return width_; }
        size_t height() const { return height_; }

        /**
         * Return whether the image is valid
         */
        bool valid()
        {
            return ( pixArray_ == nullptr )
                   || width_ == 0
                   || height_ == 0;
        }

        operator bool()
        {
            return valid();
        }

        // iterators

        iterator begin() { return pixArray_; }
        iterator end()   { return pixArray_ + (width_ * height_); }

        const_iterator begin() const { return pixArray_; }
        const_iterator end()   const { return pixArray_ + (width_ * height_); }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend()   const { return end(); }

        /**
         * Row major indexing. so image[y][x]
         */
        iterator       operator[]( size_t y )       { return pixArray_ + y*width_; }
        const_iterator operator[]( size_t y ) const { return pixArray_ + y*width_; }

    };

    template <typename PixType>
    std::shared_ptr<HostImage<PixType>>
    makeHostImage(PendingImage const& pending)
    {
        typedef HostImage<PixType> image_type;
        auto out = std::make_shared<image_type>( pending.width(), pending.height() );
        pending.read(out->begin());

        return out;
    }

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
        };

        /**
         * Creates a new level
         */
        typedef std::function< LevelPair(PendingImage const&) > NextLevelFunc;

        /**
         * Collapses two levels
         */
        typedef std::function< PendingImage(LevelPair const&) > CollapseLevelFunc;

        /**
         * Fuses several pyramids at a single layer
         */
        typedef std::function< PendingImage(std::vector<PendingImage> const&) > FuseLevelsFunc;


    private:
        NextLevelFunc nextLevelFunc;
        CollapseLevelFunc collapseLevelFunc;
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
