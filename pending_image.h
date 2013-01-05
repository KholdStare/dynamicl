#ifndef PENDING_IMAGE_H_PU6OO2YN
#define PENDING_IMAGE_H_PU6OO2YN

#include "cl_common.h"
#include "kernel.hpp"

namespace DynamiCL
{

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
        static const size_t N = detail::image_traits<cl::Image2D>::N;

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

        size_t width() const { return this->image.template getImageInfo<CL_IMAGE_WIDTH>(); }
        size_t height() const { return this->image.template getImageInfo<CL_IMAGE_HEIGHT>(); }
        size_t depth() const { return this->image.template getImageInfo<CL_IMAGE_DEPTH>(); }

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
         * Process this image with the specified kernel, reusing the 
         * specified image @a reuseImage for the result.
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

    typedef PendingImage<cl::Image2D> Pending2DImage;

}

#endif /* end of include guard: PENDING_IMAGE_H_PU6OO2YN */
