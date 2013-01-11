#ifndef PYRAMID_GROUP_H_BKS04HUH
#define PYRAMID_GROUP_H_BKS04HUH

#include <vector>

#include "cl_common.h"
#include "image_pyramid.h"
#include "host_image.hpp"

namespace DynamiCL
{

    class ImagePyramid;

    class MergeGroup
    {
        ComputeContext const& context_;
        cl::Program program_;
        size_t numLevels_;
        std::vector<ImagePyramid> pyramids_;
        // TODO: allocated arena goes here
        
        typedef RGBA<float> pixel_type;
        typedef HostImage<pixel_type, 2> image_type;
        typedef HostImageView<pixel_type, 2> view_type;
        typedef cl::Image2D climage_type;

    public:

        MergeGroup(ComputeContext const& context,
                cl::Program const& program);
                //size_t numLevels);

        //ImagePyramid& addImage(view_type const& image);
        void addImage(view_type const& image);

        /**
         * @Return the number of pyramids currently part of the group
         */
        size_t numImages() const { return pyramids_.size(); }

        bool empty() const { return numImages() == 0; }

        /**
         * Merge the images in this group into an HDR image
         */
        image_type merge();

    };
    
} /* DynamiCL */ 

#endif /* end of include guard: PYRAMID_GROUP_H_BKS04HUH */
