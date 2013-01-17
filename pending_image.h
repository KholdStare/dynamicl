#ifndef PENDING_IMAGE_H_PU6OO2YN
#define PENDING_IMAGE_H_PU6OO2YN

#include "cl_common.h"
#include "kernel.hpp"

namespace DynamiCL
{
    template <typename CLImage>
    struct PendingImage;

    /**
     * A type trait to see if a type is a pending image.
     */
    template <typename T>
    struct is_pending_image : std::false_type { };

    template <typename CLImage>
    struct is_pending_image<PendingImage<CLImage>> : std::true_type { };

    /**
     * Represents an image that is currently being processed.
     * Think of it as "std::future<Image>" but for OpenCL
     */
    template <typename CLImage>
    struct PendingImage
    {
        typedef typename detail::image_traits<CLImage>::climage_type climage_type;
        static const size_t N = detail::image_traits<climage_type>::N;

        ComputeContext const& context;
        climage_type image;
        std::vector<cl::Event> events;

        PendingImage(PendingImage&& other)
            : context(other.context),
              image(std::move(other.image)),
              events(std::move(other.events))
        {
            other.image = climage_type();
        }

        PendingImage& operator =(PendingImage&& other)
        {
            image = std::move(other.image);
            events = std::move(other.events);
            other.image = climage_type();
            return *this;
        }

        PendingImage(ComputeContext const& c)
            : context(c),
              image(),
              events()
        { }

        PendingImage(ComputeContext const& c, climage_type const& im)
            : context(c),
              image(im),
              events()
        { }

        size_t width() const {
            return this->image.template getImageInfo<
                        detail::image_traits<climage_type>::dim_info[0]
                    >();
        }

        size_t height() const {
            return this->image.template getImageInfo<
                        detail::image_traits<climage_type>::dim_info[1]
                    >();
        }

        size_t depth() const {
            return this->image.template getImageInfo<
                        detail::image_traits<climage_type>::dim_info[2]
                    >();
        }

        std::array<size_t, N> dimensions() const
        {
            return getDims(this->image);
        }

        /**
         * Process this image using the passed kernel,
         * storing the result in @a reuseImage, with global range
         * 
         * @note Most specified overload
         */
        template <typename CLImage2>
        PendingImage<CLImage2>
        process(Kernel const& kernel,
                CLImage2 const& reuseImage,
                cl::NDRange const& kernelRange) const
        {
            // create pending image
            PendingImage result(context, reuseImage);

            // create a kernel with that image
            cl::Kernel clkernel = kernel.build(this->image, result.image);

            // enqueue kernel computation
            cl::Event complete;
            context.queue.enqueueNDRangeKernel(clkernel,
                                       cl::NullRange,
                                       kernelRange,
                                       cl::NullRange, 
                                       &this->events,
                                       &complete);

            result.events.push_back(complete);

            return result;
        }

        /**
         * Process this image with the specified kernel, creating a new
         * image of specified dimensions for the result
         */
        PendingImage process(Kernel const& kernel,
                             std::array<size_t, N> const& dims,
                             cl::NDRange const& kernelRange) const;

        /**
         * Process this image with the specified kernel, reusing the specified
         * image @a reuseImage for the result.
         *
         * @note Range of kernel application is derived from the dimensions of
         * either the source or destination image (specified in the kernel).
         */
        template <typename CLImage2>
        PendingImage<CLImage2>
        process(Kernel const& kernel, CLImage2 const& reuseImage) const
        {
            // figure out range of kernel
            cl::NDRange kernelRange;
            if (kernel.range == Kernel::Range::SOURCE)
            {
                kernelRange = toNDRange(this->dimensions());
            }
            else
            {
                kernelRange = toNDRange(getDims(reuseImage));
            }

            return process(kernel, reuseImage, kernelRange);
        }

        /**
         * Process this image with the specified kernel, and create a new
         * image of type CLImage2, and dimensions @a dims.
         */
        template <typename CLImage2>
        PendingImage<CLImage2>
        process(Kernel const& kernel,
                std::array<size_t, detail::image_traits<CLImage2>::N> const& dims) const
        {
            // construct a new image
            auto image = createCLImage<CLImage2>(context, dims);

            return this->process(kernel, image);
        }

        /**
         * Process this image with the specified kernel, creating a new image
         * of the same dimensions.
         */
        PendingImage process(Kernel const& kernel) const;

        /**
         * Read image into host memory
         */
        void readInto(void* hostPtr) const;
        
    };

    namespace detail
    {

        inline void aggregate_events_impl(std::vector<cl::Event>& vec)
        { }

        template <typename T, typename... Ts>
        void aggregate_events_impl(std::vector<cl::Event>& vec, 
                                   PendingImage<T> const& first,
                                   PendingImage<Ts> const&... rest)
        {
            vec.insert(end(vec), begin(first.events), end(first.events));
            aggregate_events_impl(vec, rest...);
        }

    }

    template <typename... Ts>
    std::vector<cl::Event>
    aggregateEvents(PendingImage<Ts> const&... images)
    {
        std::vector<cl::Event> vec;
        detail::aggregate_events_impl(vec, images...);
        return vec;
    }

    /**
     * Some non-member functions to help with PendingImages
     */
    namespace Pending
    {

        /**
         * @note have to manually specify CLImage type for output
         */
        template <typename CLImage, typename... Ts >
        static PendingImage<CLImage>
        process(ComputeContext const& context,
                Kernel const& kernel,
                std::array<size_t, detail::image_traits<CLImage>::N> const& dims,
                cl::NDRange const& kernelRange,
                PendingImage<Ts> const&... inputs)
        {
            typedef PendingImage<CLImage> pending_type;

            auto result = createCLImage<CLImage>(context, dims);

            cl::Kernel clkernel = kernel.build(inputs.image... , result);
                        
            cl::Event complete;
            std::vector<cl::Event> waitfor = aggregateEvents(inputs...);
            context.queue.enqueueNDRangeKernel(clkernel,
                                       cl::NullRange,
                                       kernelRange,
                                       cl::NullRange, 
                                       &waitfor,
                                       &complete);

            pending_type pendingResult(context, result);
            pendingResult.events.push_back(complete);

            return pendingResult;
        }

    }


    template <typename CLImage>
    PendingImage<CLImage> PendingImage<CLImage>::process(Kernel const& kernel) const
    {
        return this->process<CLImage>(kernel, getDims<climage_type>(this->image));
    }

    template <typename CLImage>
    PendingImage<CLImage>
    PendingImage<CLImage>::process(Kernel const& kernel,
                         std::array<size_t, N> const& dims,
                         cl::NDRange const& kernelRange) const
    {
        // construct a new image
        auto image = createCLImage<climage_type>(context, dims);

        return this->process(kernel, image, kernelRange);
    }

    template <typename CLImage>
    void PendingImage<CLImage>::readInto(void* hostPtr) const
    {
        context.queue.enqueueReadImage(this->image,
                CL_TRUE,
                VectorConstructor<size_t>::construct(0, 0, 0),
                toSizeVector(getDims(this->image), 1),
                0,
                0,
                hostPtr,
                &this->events);
    }

    // some commonly used types
    typedef PendingImage<cl::Image2D> Pending2DImage;
    typedef PendingImage<cl::Image2DArray> Pending2DImageArray;

}

#endif /* end of include guard: PENDING_IMAGE_H_PU6OO2YN */
