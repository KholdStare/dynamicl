#ifndef PYRAMID_GROUP_H_BKS04HUH
#define PYRAMID_GROUP_H_BKS04HUH

#include <vector>

#include "utils.h"
#include "cl_common.h"
#include "image_pyramid.h"
#include "host_image.hpp"

namespace DynamiCL
{

    class MergeGroup
    {
        typedef ImagePyramid pyramid_type;
        typedef pyramid_type::pixel_type pixel_type;
        typedef pyramid_type::image_type image_type;
        typedef pyramid_type::view_type view_type;
        typedef pyramid_type::climage_type climage_type;

        ComputeContext const& context_;
        cl::Program program_;
        size_t width_;      ///< width of images in merge
        size_t height_;     ///< height of images in merge
        size_t numLevels_;  ///< number of levels required to merge images
        size_t pixelsPerPyramid; ///< number of pixels for all levels of one pyramid
        std::vector<ImagePyramid> pyramids_;
        // TODO: create single reusable arena
        std::vector<array_ptr<pixel_type>> arenas_; ///< memory arenas for pyramid images

    public:

        MergeGroup(ComputeContext const& context,
                cl::Program const& program,
                size_t width,
                size_t height);
                //size_t numLevels);

        // move constructor
        MergeGroup(MergeGroup&& other);
        MergeGroup& operator = (MergeGroup&& other);

        // disable copy operations
        MergeGroup(MergeGroup const&) = delete;
        MergeGroup& operator = (MergeGroup const&) = delete;

        void addImage(view_type const& image);

        /**
         * @Return the number of pyramids currently part of the group
         */
        size_t numImages() const { return pyramids_.size(); }

        bool empty() const { return numImages() == 0; }

        /**
         * Merge the images in this group into an HDR image.
         *
         * This resets the images in this group.
         */
        void mergeInto(view_type &);

    };
    
} /* DynamiCL */ 

#endif /* end of include guard: PYRAMID_GROUP_H_BKS04HUH */
