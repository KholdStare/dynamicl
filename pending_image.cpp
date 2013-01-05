#include "pending_image.h"

namespace DynamiCL
{
    PendingImage PendingImage::process(Kernel const& kernel) const
    {
        size_t width = this->image.getImageInfo<CL_IMAGE_WIDTH>();
        size_t height = this->image.getImageInfo<CL_IMAGE_HEIGHT>();

        return this->process(kernel, width, height);
    }

    PendingImage PendingImage::process(Kernel const& kernel, size_t width, size_t height) const
    {
        // construct a new image
        cl::Image2D resultImage(context.context,
                CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), // TODO: get image format from this image
                width,
                height);

        return this->process(kernel, resultImage);
    }

    PendingImage PendingImage::process(Kernel const& kernel, cl::Image2D const& reuseImage) const
    {
        // create pending image
        PendingImage result(context, reuseImage);

        // create a kernel with that image
        cl::Kernel clkernel = kernel.build(this->image, result.image);

        // figure out range of kernel
        cl::Image2D const* rangeGuide; // which image do we get the range from
        if (kernel.range == Kernel::Range::SOURCE)
        {
            rangeGuide = &this->image;
        }
        else
        {
            rangeGuide = &reuseImage;
        }

        size_t width = rangeGuide->getImageInfo<CL_IMAGE_WIDTH>();
        size_t height = rangeGuide->getImageInfo<CL_IMAGE_HEIGHT>();

        // enqueue kernel computation
        cl::Event complete;
        context.queue.enqueueNDRangeKernel(clkernel,
                                   cl::NullRange,
                                   cl::NDRange(width, height),
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
