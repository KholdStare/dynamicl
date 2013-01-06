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

    namespace detail
    {

        template <size_t N>
        struct dimension_traits;

        template <>
        struct dimension_traits<1>
        {
            typedef cl::Image1D climage_type;
        };

        template <>
        struct dimension_traits<2>
        {
            typedef cl::Image2D climage_type;
        };

        template <>
        struct dimension_traits<3>
        {
            typedef cl::Image3D climage_type;
        };

    }

    template <typename CLImage, typename PixType, size_t N>
    PendingImage<CLImage>
    makePendingImage(ComputeContext const& context, HostImage<PixType, N> const& image)
    {
        typedef CLImage climage_type;
        typedef PendingImage<climage_type> pending_type;

        climage_type climage =
            createCLImage<climage_type>(context,
                          image.dimensions(),
                          const_cast<void*>(image.rawData()));

        pending_type out(context, climage);

        return out;
    }

    /**
     * Takes an image currently on the host, and transforms
     * it inplace using an OpenCL kernel.
     */
    template <typename PixType, size_t N>
    void processImageInPlace(HostImage<PixType, N>& image,
                              Kernel const& kernel,
                              ComputeContext const& context)
    {
        makePendingImage<typename detail::dimension_traits<N>::climage_type>(context, image)
            .process(kernel)
            .readInto(image.rawData());
    }

    template <typename PixType, typename CLImage>
    std::shared_ptr<HostImage<PixType, detail::image_traits<CLImage>::N>>
    makeHostImage(PendingImage<CLImage> const& pending)
    {
        static const size_t N = detail::image_traits<CLImage>::N;
        typedef HostImage<PixType, N> image_type;

        auto out = std::make_shared<image_type>(pending.dimensions());
        pending.readInto(out->rawData());

        return out;
    }

}

#endif /* end of include guard: CL_UTILS_H_PZU0OYIC */
