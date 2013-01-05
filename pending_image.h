#ifndef PENDING_IMAGE_H_PU6OO2YN
#define PENDING_IMAGE_H_PU6OO2YN

#include "cl_common.h"
#include "kernel.hpp"

namespace DynamiCL
{

    /**
     * Represents an image that is currently being processed.
     * think of it as "std::future<Image>" but for OpenCL
     */
    template <typename CLImage>
    struct PendingImage
    {
        typedef typename detail::image_traits<CLImage>::climage_type climage_type;
        static const size_t N = detail::image_traits<cl::Image2D>::N;

        ComputeContext const& context;
        climage_type image;
        std::vector<cl::Event> events; // TODO: const?

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
        PendingImage process(Kernel const& kernel,
                             climage_type const& reuseImage,
                             cl::NDRange const& kernelRange) const;

        PendingImage process(Kernel const& kernel,
                             std::array<size_t, N> const& dims,
                             cl::NDRange const& kernelRange) const;

        PendingImage process(Kernel const& kernel,
                             climage_type const& reuseImage) const;

        PendingImage process(Kernel const& kernel,
                             std::array<size_t, N> const& dims) const;

        PendingImage process(Kernel const& kernel) const; // assumes same dimension


        /**
         * Read image into host memory
         */
        void readInto(void* hostPtr) const;
        
    };


    template <typename CLImage>
    PendingImage<CLImage> PendingImage<CLImage>::process(Kernel const& kernel) const
    {
        return this->process(kernel, getDims<climage_type>(this->image));
    }

    template <typename CLImage>
    PendingImage<CLImage> PendingImage<CLImage>::process(Kernel const& kernel, std::array<size_t, N> const& dims) const
    {
        // construct a new image
        auto image = createCLImage<climage_type>(context, dims);

        return this->process(kernel, image);
    }

    template <typename CLImage>
    PendingImage<CLImage> PendingImage<CLImage>::process(Kernel const& kernel,
                         std::array<size_t, N> const& dims,
                         cl::NDRange const& kernelRange) const
    {
        // construct a new image
        auto image = createCLImage<climage_type>(context, dims);

        return this->process(kernel, image, kernelRange);
    }

    template <typename CLImage>
    PendingImage<CLImage> PendingImage<CLImage>::process(Kernel const& kernel, climage_type const& reuseImage) const
    {
        // figure out range of kernel
        climage_type const* rangeGuide; // which image do we get the range from
        if (kernel.range == Kernel::Range::SOURCE)
        {
            rangeGuide = &this->image;
        }
        else
        {
            rangeGuide = &reuseImage;
        }

        cl::NDRange kernelRange = toNDRange(getDims(*rangeGuide));

        return process(kernel, reuseImage, kernelRange);
    }

    template <typename CLImage>
    PendingImage<CLImage> PendingImage<CLImage>::process(Kernel const& kernel,
                         climage_type const& reuseImage,
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
