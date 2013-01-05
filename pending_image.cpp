#include "pending_image.h"

namespace DynamiCL
{
    PendingImage PendingImage::process(Kernel const& kernel) const
    {
        return this->process(kernel, getDims<climage_type>(this->image));
    }

    PendingImage PendingImage::process(Kernel const& kernel, std::array<size_t, N> const& dims) const
    {
        // construct a new image
        auto image = createCLImage<climage_type>(context, dims);

        return this->process(kernel, image);
    }

    PendingImage PendingImage::process(Kernel const& kernel, climage_type const& reuseImage) const
    {
        // create pending image
        PendingImage result(context, reuseImage);

        // create a kernel with that image
        cl::Kernel clkernel = kernel.build(this->image, result.image);

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

        //size_t width = rangeGuide->getImageInfo<CL_IMAGE_WIDTH>();
        //size_t height = rangeGuide->getImageInfo<CL_IMAGE_HEIGHT>();

        // enqueue kernel computation
        cl::Event complete;
        context.queue.enqueueNDRangeKernel(clkernel,
                                   cl::NullRange,
                                   toNDRange(getDims(*rangeGuide)),
                                   cl::NullRange, 
                                   &this->events,
                                   &complete);

        result.events.push_back(complete);

        return result;
    }

    void PendingImage::readInto(void* hostPtr) const
    {
        size_t width = this->width();
        size_t height = this->height();

        context.queue.enqueueReadImage(this->image,
                CL_TRUE,
                VectorConstructor<size_t>::construct(0, 0, 0),
                VectorConstructor<size_t>::construct(width, height, 1),
                0,
                0,
                hostPtr,
                &this->events);
    }

}
