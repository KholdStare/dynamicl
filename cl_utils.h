#ifndef CL_UTILS_H_PZU0OYIC
#define CL_UTILS_H_PZU0OYIC

#include "cl_common.h"

#include "kernel.hpp"
#include "pending_image.h"
#include "host_image.hpp"

#include <memory>

namespace DynamiCL
{

    /***************************************************************************
     *                        Kernel and Image helpers                         *
     ***************************************************************************/

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

    //template <typename PixType>
    //PendingImage
    //makePendingImage(ComputeContext const& context, HostImage<PixType, 3> const& image)
    //{
        //cl::Image2D clInputImage(context.context,
                //CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                //cl::ImageFormat(CL_RGBA, CL_FLOAT), // TODO: not hardcode?
                //image.width(),
                //image.height(),
                //0,
                //const_cast<void*>(image.rawData()));

        //PendingImage out(context, clInputImage);

        //return out;
    //}

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

}

#endif /* end of include guard: CL_UTILS_H_PZU0OYIC */
