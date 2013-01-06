#ifndef IMAGE_PYRAMID_H_ZSOHJD6F
#define IMAGE_PYRAMID_H_ZSOHJD6F

#include "cl_utils.h"

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
        ImagePyramid( ComputeContext const& context,
                      image_type& startImage,
                      size_t numLevels,
                      NextLevelFunc);

        /**
         * Create a pyramid from the guts of another.
         */
        ImagePyramid( ComputeContext const& context,
                      std::vector<image_type>&& levels )
            : context_(context),
              levels_(std::move(levels))
        { }

        // disable copying
        ImagePyramid( ImagePyramid const& other ) = delete;
        ImagePyramid& operator = ( ImagePyramid const& other ) = delete;

        ImagePyramid( ImagePyramid&& other )
            : context_(other.context_),
              levels_(std::move(other.levels_))
        { }

        ImagePyramid& operator = ( ImagePyramid&& other )
        {
            levels_ = std::move(other.levels_);
            return *this;
        }

        /**
         * Return a vector of all the levels in this image pyramid
         */
        std::vector<image_type> const& levels() const { return levels_; }

        /**
         * Move the vector of all the levels in this image pyramid out.
         *
         * This leaves the pyramid empty. Use this if you want to modify the
         * individual images in the pyramid.
         */
        std::vector<image_type> releaseLevels() { return std::move(levels_); }

        /**
         * Returns collapsed image from image pyramid.
         *
         * @note Pyramid is left empty (no levels), to save memory.
         */
        image_type collapse(CollapseLevelFunc);

        /**
         * Fuses passed-in pyramids into one.
         *
         * @note input pyramids are left empty: this frees up memory as soon as
         * it is not needed.
         */
        static ImagePyramid fuse(std::vector<ImagePyramid>& pyramids, FuseLevelsFunc);

    private:
        ComputeContext const& context_; ///< context for OpenCL operations
        std::vector<image_type> levels_;
    };

}

#endif /* end of include guard: IMAGE_PYRAMID_H_ZSOHJD6F */
