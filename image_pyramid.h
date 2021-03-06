#ifndef IMAGE_PYRAMID_H_ZSOHJD6F
#define IMAGE_PYRAMID_H_ZSOHJD6F

#include "utils.h"
#include "cl_common.h"
#include "pending_image.h"
#include "host_image.hpp"

namespace DynamiCL
{

    /***************************************************************************
     *                             Pyramid Helpers                             *
     ***************************************************************************/
    
    /**
     * Represents an image pyramid, with methods to construct it from
     * a source image.
     *
     * Manages caching OpenCL images on the host, as well as chunking
     * them appropriately for operations to fit in GPU memory.
     */
    class ImagePyramid
    {
    public:
        typedef RGBA<float> pixel_type;
        typedef HostImage<pixel_type, 2> image_type;
        typedef HostImageView<pixel_type, 2> view_type;
        typedef HostImageView<pixel_type, 3> fuse_view_type;
        typedef cl::Image2D climage_type;

        /**
         * An image pair, of two levels of a pyramid
         */
        struct LevelPair
        {
            Pending2DImage upper;
            Pending2DImage lower;
        };

        /**
         * Given a length, produces new length
         * in the next level of pyramid
         */
        typedef std::function< size_t(size_t) > HalvingFunc;

        /**
         * Creates a new level
         */
        typedef std::function< LevelPair(Pending2DImage const&) > NextLevelFunc;

        /**
         * Collapses two levels
         */
        typedef std::function< Pending2DImage(LevelPair const&) > CollapseLevelFunc;

        /**
         * Fuses several pyramids at a single layer
         */
        typedef std::function< Pending2DImage(PendingImage<cl::Image2DArray> const&) > FuseLevelsFunc;

        /**
         * Construct an image puramid with @a numLevels levels,
         * from the @a startImage, using a specified NextLevelFunc
         */
        //ImagePyramid( ComputeContext const& context,
                      //view_type const& startImage,
                      //size_t numLevels,
                      //HalvingFunc const&,
                      //NextLevelFunc const&);

        ImagePyramid( ComputeContext const& context,
                      std::vector<view_type>&& levelViews,
                      NextLevelFunc const&);

        /**
         * Create a pyramid from the guts of another.
         */
        ImagePyramid( array_ptr<pixel_type>&& data,
                      ComputeContext const& context,
                      std::vector<view_type>&& views)
            : data_(std::move(data)),
              context_(context),
              views_(std::move(views))
        { }

        // disable copying
        ImagePyramid( ImagePyramid const& other ) = delete;
        ImagePyramid& operator = ( ImagePyramid const& other ) = delete;

        ImagePyramid( ImagePyramid&& other )
            : data_(std::move(other.data_)),
              context_(other.context_),
              views_(std::move(other.views_))
        { }

        ImagePyramid& operator = ( ImagePyramid&& other )
        {
            data_ = std::move(other.data_);
            views_ = std::move(other.views_);
            return *this;
        }

        /**
         * Return a vector of all the levels in this image pyramid
         */
        std::vector<view_type> const& levels() const { return views_; }

        /**
         * Returns collapsed image from image pyramid.
         *
         * TODO: view is first subimage in arena. 
         * @note Pyramid is left empty (no levels), to save memory.
         */
        void collapseInto(CollapseLevelFunc, view_type&);

        /**
         * Fuses passed-in pyramids into one.
         *
         * @note input pyramids are left empty: this frees up memory as soon as
         * it is not needed.
         */
        static ImagePyramid fuse(std::vector<ImagePyramid>& pyramids, FuseLevelsFunc);

        static void fuseInto(ComputeContext const& context,
                         std::vector<fuse_view_type>& fuseViews,
                         FuseLevelsFunc const&,
                         std::vector<view_type>& dest);

        static std::vector<view_type>
        createPyramidViews(size_t width,
                size_t height,
                size_t numLevels,
                HalvingFunc const& halve,
                pixel_type* array);

        /**
         * @Return total number of pixels required to store
         * an image pyramid of numLevels
         */
        static size_t pyramidSize(size_t width,
                size_t height,
                size_t numLevels,
                HalvingFunc const& halve);

    private:
        array_ptr<pixel_type> data_; ///< optionally manages own data
        ComputeContext const& context_; ///< context for OpenCL operations
        std::vector<view_type> views_;

        void initPyramid( NextLevelFunc const& createNext);
    };

}

#endif /* end of include guard: IMAGE_PYRAMID_H_ZSOHJD6F */
