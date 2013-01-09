#include "image_pyramid.h"
#include "save_image.h"
#include <iostream>
#include <sstream>


namespace DynamiCL
{
    ImagePyramid::ImagePyramid( ComputeContext const& context,
                      image_type& startImage,
                      size_t numLevels,
                      NextLevelFunc createNext)
        : context_(context)
    {
        levels_.reserve(numLevels);

        Pending2DImage image = makePendingImage<climage_type>(context, startImage.view());

        // create levels one at a time
        for (size_t level = 1; level < numLevels; ++level)
        {
            LevelPair pair = createNext(image);

            levels_.push_back( std::move(makeHostImage<pixel_type>(pair.upper)) );

            image = std::move(pair.lower);
        }

        // last level is byproduct of last creation
        levels_.push_back( std::move(makeHostImage<pixel_type>(image)) );
    }

    ImagePyramid::image_type ImagePyramid::collapse(CollapseLevelFunc collapseLevel)
    {
        std::vector<image_type> levels = this->releaseLevels();

        // extracts next level
        auto nextLevel =
            [&]() -> image_type
            {
                if (levels.empty())
                {
                    return image_type();
                }
                image_type i = std::move(levels.back());
                levels.pop_back();
                return i;
            };

        image_type lower = nextLevel();
        Pending2DImage result = makePendingImage<climage_type>(context_, lower.view());
        image_type upper = nextLevel();

        // keep collapsing layers
        while(upper.valid())
        {
            Pending2DImage u = makePendingImage<climage_type>(context_, upper.view());
            // create pair to pass to the collapser
            LevelPair pair {std::move(u), std::move(result)};

            // collapse using passed function
            result = collapseLevel(pair);

            // save upper layer for later
            lower = std::move(upper);
            upper = nextLevel();
        }

        result.readInto(lower.view().rawData());
        return lower;
    }

    ImagePyramid ImagePyramid::fuse(std::vector<ImagePyramid>& pyramids,
                                    FuseLevelsFunc fuseLevels)
    {
        size_t numPyramids = pyramids.size();
        assert (numPyramids > 1); // need to merge more than one

        ComputeContext const& context = pyramids.at(0).context_;
        size_t numLevels = pyramids[0].levels().size();

        // TODO: make sure all pyramids have the same number of levels
        // and dimensions

        // outer vector: element is pyramid
        // inner vector: element is a level
        std::vector<std::vector<image_type>> pyramidGuts;

        // extract guts from pyramids
        for (ImagePyramid& pyramid : pyramids)
        {
            // ensure all pyramids have the same number of levels
            assert( pyramid.levels().size() == numLevels );

            pyramidGuts.push_back(pyramid.releaseLevels());
        }

        std::cout << "Extracted Guts" << std::endl;

        //return ImagePyramid(context, std::move(pyramidGuts[1]));

        // outer vector: represents collection of levels
        // inner vector: single level from all pyramids
        std::vector<std::vector<image_type>> levelCollection(numLevels);

        // rearrange so that each inner vector
        // has all images of a particular level from all pyramids
        for (size_t level = 0; level < numLevels; ++level)
        {
            for (std::vector<image_type>& pyramid : pyramidGuts)
            {
                levelCollection[level].push_back( std::move(pyramid.back()) );
                pyramid.pop_back();
            }
        }

        // pyramid guts should be empty
        for (std::vector<image_type>& pyramid : pyramidGuts)
        {
            assert( pyramid.empty() );
        }

        // levelCollection starts with last (smallest level),
        // and increases from there
        if ( levelCollection.size() > 1 )
        {
            assert( levelCollection[0][0].view().width() < levelCollection[1][0].view().width() );
        }

        std::cout << "Have to fuse " << levelCollection.size() << " levels" << std::endl;

        // ===================================================
        // now we can fuse each level individually
        std::vector<image_type> fusedLevels;

        // fuse all levels
        size_t l = 0;
        while(!levelCollection.empty())
        {
            // these image have to be fused
            std::vector<image_type> singleLevel = std::move(levelCollection.back());
            levelCollection.pop_back();

            // create contiguous image array in memory
            HostImage<pixel_type, 3> levelArray(singleLevel);
            singleLevel.clear(); // deallocate subimages

            Pending2DImageArray clarray =
                makePendingImage<cl::Image2DArray>(context, levelArray.view());

            //std::stringstream sstr;
            //sstr << "level_test" << l << ".tiff";
            ////saveTiff16(makeHostImage<RGBA<float>>(fused), sstr.str());
            //saveTiff16(collapseDimension(std::move(levelArray)), sstr.str());

            auto dims = clarray.dimensions();
            std::cout << "Dimensions: "
                      << dims[0] << " x "
                      << dims[1] << " x "
                      << dims[2] << std::endl;

            Pending2DImage fused = fuseLevels(clarray);

            fusedLevels.push_back(makeHostImage<RGBA<float>>(fused));
            ++l;
        }

        std::cout << "Fused " << fusedLevels.size() << " levels" << std::endl;

        return ImagePyramid(context, std::move(fusedLevels));
    }

} /* DynamiCL */ 

