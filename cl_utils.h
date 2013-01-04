#ifndef CL_ERRORS_H_GTNVZIRP
#define CL_ERRORS_H_GTNVZIRP

#define __CL_ENABLE_EXCEPTIONS
#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <algorithm>
#include <functional>
#include <memory>
#include <array>
#include <type_traits>

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

        // TODO: can these be const?
        PendingImage process(Kernel const& kernel) const; // assumes same dimension
        PendingImage process(Kernel const& kernel, size_t width, size_t height) const;
        PendingImage process(Kernel const& kernel, cl::Image2D const& reuseImage) const;

        /**
         * Read image into host memory
         */
        void readInto(void* hostPtr) const;
        
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

    template <typename PixType, size_t N>
    class HostImage
    {
        static_assert( N >= 1, "An image has to have at least one dimension." );
        // TODO: add assert once availble in gcc
        //static_assert( std::is_trivially_copyable<PixType>,
                        //"An image must consist of trivially copyable pixels." );

        std::array<size_t, N> dims_;
        PixType* pixArray_;

        void dealloc()
        {
            delete[] pixArray_;
        }

        void invalidate()
        {
            std::fill_n(dims_.begin(), N, 0);
            pixArray_ = nullptr; // don't delete
        }


    public:
        typedef PixType* iterator;
        typedef PixType const* const_iterator;
        typedef PixType pixel_type;

        HostImage(size_t width, size_t height)
            : dims_({{width, height}}),
              pixArray_(new PixType[width*height])
        { }

        HostImage(size_t width, size_t height, size_t depth)
            : dims_({{width, height, depth}}),
              pixArray_(new PixType[width*height*depth])
        { }

        /**
         * Creates a single image of N+1 dimensions,
         * out of M images of dimension N.
         *
         * @note input images are deallocated.
         */
        HostImage(std::vector<HostImage<PixType, N-1>> const& subimages);

        ~HostImage() { dealloc(); }

        HostImage()
        { 
            invalidate();
        }

        // disable copying because expensive
        HostImage(HostImage const& other) = delete;
        HostImage& operator =(HostImage const& other) = delete;

        // move constructor
        HostImage(HostImage&& other)
            : dims_(std::move(other.dims_)),
              pixArray_(other.pixArray_)
        {
            other.invalidate();
        }

        // move assignment
        HostImage& operator =(HostImage&& other)
        {
            std::copy_n(other.dims_.begin(), N, dims_.begin());
            pixArray_ = other.pixArray_;

            other.invalidate();
        }

        size_t width() const { return dims_[0]; }
        size_t height() const { return dims_[1]; }
        size_t depth() const { return dims_[2]; }

        std::array<size_t, N> const& dimensions() const { return dims_; }

        /**
         * Return the total number of pixels in the image.
         */
        size_t totalSize() const
        {
            return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<size_t>());
        }

        /**
         * Return whether the image is valid
         */
        bool valid()
        {
            // check dimensions
            for (size_t n = 0; n < N; ++n)
            {
                if (dims_[n] == 0)
                {
                    return false;
                }
            }
            // check nullptr
            return pixArray_ != nullptr;
        }

        operator bool()
        {
            return valid();
        }

        // iterators

        iterator begin() { return pixArray_; }
        iterator end()   { return pixArray_ + totalSize(); }

        const_iterator begin() const { return pixArray_; }
        const_iterator end()   const { return pixArray_ + totalSize(); }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend()   const { return end(); }

        void const* rawData() const { return static_cast<void const*>(begin()); }
        void*       rawData()       { return static_cast<void*>(begin()); }

    };

    template <typename PixType, size_t N>
    HostImage<PixType, N>::HostImage(std::vector<HostImage<PixType, N-1>> const& subimages)
        : pixArray_(nullptr)
    {

        assert(subimages.size() > 0);

        dims_[N-1] = subimages.size();

        // copy the first image dimensions
        std::array<size_t, N-1> const& otherdims = subimages[0].dimensions();
        std::copy(otherdims.begin(), otherdims.end(), dims_.begin());

        // can now allocate space
        pixArray_ = new PixType[totalSize()];
        PixType* writePtr = pixArray_; // current write point

        typedef HostImage<PixType, N-1> subimage_type;
        for(subimage_type const& subimage : subimages)
        {
            // ensure all dimensions match
            assert( std::equal( subimage.dimensions().begin(),
                                subimage.dimensions().end(),
                                dims_.begin() ) );

            writePtr = std::copy( subimage.begin(), subimage.end(), writePtr );
        }
    }

    /**
     * Takes an image currently on the host, and transforms
     * it inplace using an OpenCL kernel.
     */
    template <typename PixType>
    void processImageInPlace(HostImage<PixType, 2>& image,
                              Kernel const& kernel,
                              ComputeContext const& context)
    {
        PendingImage clImage = makePendingImage(context, image);
        clImage.process(kernel).readInto(image.rawData());
    }

    template <typename PixType>
    std::shared_ptr<HostImage<PixType, 2>>
    makeHostImage(PendingImage const& pending)
    {
        typedef HostImage<PixType, 2> image_type;
        auto out = std::make_shared<image_type>( pending.width(), pending.height() );
        pending.readInto(out->rawData());

        return out;
    }

    template <typename PixType>
    PendingImage
    makePendingImage(ComputeContext const& context, HostImage<PixType, 2> const& image)
    {
        cl::Image2D clInputImage(context.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), // TODO: not hardcode?
                image.width(),
                image.height(),
                0,
                const_cast<void*>(image.rawData()));

        PendingImage out(context, clInputImage);

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
        typedef RGBA<float> pixel_type;
        typedef HostImage<pixel_type, 2> image_type;

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

        /**
         * Construct an image puramid with @a numLevels levels,
         * from the @a startImage, using a specified NextLevelFunc
         */
        ImagePyramid( ComputeContext const& context,
                      image_type& startImage,
                      size_t numLevels,
                      NextLevelFunc);

        /**
         * Create a pyramid from the guts of another.
         */
        ImagePyramid( ComputeContext const& context,
                      std::vector<image_type>&& levels )
            : context_(context),
              levels_(std::move(levels))
        { }

        // disable copying
        ImagePyramid( ImagePyramid const& other ) = delete;
        ImagePyramid& operator = ( ImagePyramid const& other ) = delete;

        ImagePyramid( ImagePyramid&& other )
            : context_(other.context_),
              levels_(std::move(other.levels_))
        { }

        ImagePyramid& operator = ( ImagePyramid&& other )
        {
            levels_ = std::move(other.levels_);
        }

        /**
         * Return a vector of all the levels in this image pyramid
         */
        std::vector<image_type> const& levels() const { return levels_; }

        /**
         * Move the vector of all the levels in this image pyramid out.
         *
         * This leaves the pyramid empty. Use this if you want to modify the
         * individual images in the pyramid.
         */
        std::vector<image_type> releaseLevels() { return std::move(levels_); }

        /**
         * Returns collapsed image from image pyramid.
         *
         * @note Pyramid is left empty (no levels), to save memory.
         */
        image_type collapse(CollapseLevelFunc);

        /**
         * Fuses passed-in pyramids into one.
         *
         * @note input pyramids are left empty: this frees up memory as soon as
         * it is not needed.
         */
        static ImagePyramid fuse(std::vector<ImagePyramid>& pyramids, FuseLevelsFunc);

    private:
        ComputeContext const& context_; ///< context for OpenCL operations
        std::vector<image_type> levels_;
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
