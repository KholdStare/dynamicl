#ifndef PYR_IMPL_H_MVWUEKDM
#define PYR_IMPL_H_MVWUEKDM

#include "image_pyramid.h"

namespace DynamiCL
{
    /**
     * Return how halving of a dimension happens
     * in the pyramid construction.
     */
    inline size_t halveDimension(size_t n)
    {
        return (n + 1) / 2;
    }

    ImagePyramid::LevelPair
    createPyramidLevel(Pending2DImage const& inputImage,
                       cl::Program const& program );

    Pending2DImage
    collapsePyramidLevel(ImagePyramid::LevelPair const& pair,
                         cl::Program const& program );

    Pending2DImage
    fusePyramidLevel(Pending2DImageArray const& array,
                         cl::Program const& program );

    /**
     * Given dimensions of an image determine the maximum
     * allowable levels for a laplacian pyramid
     */
    size_t calculateNumLevels(size_t width, size_t height);

    /**
     * @Return total number of pixels required to store
     * an image pyramid of numLevels
     */
    size_t pyramidSize(size_t width, size_t height, size_t numLevels);

}

#endif /* end of include guard: PYR_IMPL_H_MVWUEKDM */
