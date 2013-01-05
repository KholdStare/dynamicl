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

    PendingImage PendingImage::process(Kernel const& kernel,
                         std::array<size_t, N> const& dims,
                         cl::NDRange const& kernelRange) const
    {
        // construct a new image
        auto image = createCLImage<climage_type>(context, dims);

        return this->process(kernel, image, kernelRange);
    }

    PendingImage PendingImage::process(Kernel const& kernel, climage_type const& reuseImage) const
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

    PendingImage PendingImage::process(Kernel const& kernel,
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

    void PendingImage::readInto(void* hostPtr) const
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

}


