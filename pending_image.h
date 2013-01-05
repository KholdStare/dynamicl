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
    struct PendingImage
    {
        typedef typename detail::image_traits<cl::Image2D>::climage_type climage_type;
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

        size_t width() const { return this->image.getImageInfo<CL_IMAGE_WIDTH>(); }
        size_t height() const { return this->image.getImageInfo<CL_IMAGE_HEIGHT>(); }

        // TODO: can these be const?
        PendingImage process(Kernel const& kernel) const; // assumes same dimension
        PendingImage process(Kernel const& kernel, std::array<size_t, N> const& dims) const;
        PendingImage process(Kernel const& kernel, climage_type const& reuseImage) const;

        /**
         * Read image into host memory
         */
        void readInto(void* hostPtr) const;
        
    };

}

#endif /* end of include guard: PENDING_IMAGE_H_PU6OO2YN */
