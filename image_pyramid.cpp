#include "image_pyramid.h"
#include "cl_utils.h"
#include "save_image.h"
#include <iostream>
#include <sstream>


namespace DynamiCL
{
    void ImagePyramid::initPyramid( NextLevelFunc const& createNext )
    {
        Pending2DImage image = makePendingImage<climage_type>(context_, views_[0]);

        // create levels one at a time
        for (size_t level = 1; level < views_.size(); ++level)
        {
            LevelPair pair = createNext(image);

            pair.upper.readInto(views_[level-1].rawData());

            image = std::move(pair.lower);
        }

        image.readInto(views_.back().rawData()); // read last level

        //for (size_t level = 0; level < views_.size(); ++level)
        //{
            //std::stringstream sstr;
            //sstr << "pyramid_cons_test" << level << ".tiff";
            ////saveTiff16(makeHostImage<RGBA<float>>(fused), sstr.str());
            //saveTiff16(views_[level], sstr.str());
        //}
    }

    //ImagePyramid::ImagePyramid( ComputeContext const& context,
                      //view_type const& startImage,
                      //size_t numLevels,
                      //HalvingFunc const& halve,
                      //NextLevelFunc const& createNext)
        //: data_(pyramidSize(startImage.width(), startImage.height(), numLevels, halve)),
          //context_(context),
          //views_( createPyramidViews(startImage.width(),
                                      //startImage.height(),
                                      //numLevels,
                                      //halve,
                                      //data_.ptr()))
    //{
        //std::copy(startImage.begin(), startImage.end(), views_.at(0).begin());

        //initPyramid(createNext);
                                          
    //}

    ImagePyramid::ImagePyramid( ComputeContext const& context,
              std::vector<view_type>&& levelViews,
              NextLevelFunc const& createNext)
        : context_(context),
          views_(std::move(levelViews))
    {
        initPyramid(createNext);
    }

    void ImagePyramid::collapseInto(CollapseLevelFunc collapseLevel,
            view_type& dest)
    {
        //std::vector<image_type> levels = this->releaseLevels();
        std::vector<view_type> levels = std::move(views_);

        // extracts next level
        auto nextLevel =
            [&]() -> view_type
            {
                if (levels.empty())
                {
                    return view_type();
                }
                view_type i = std::move(levels.back());
                levels.pop_back();
                return i;
            };

        auto lower = nextLevel();
        Pending2DImage result = makePendingImage<climage_type>(context_, lower);
        auto upper = nextLevel();

        // keep collapsing layers
        while(upper.valid())
        {
            Pending2DImage u = makePendingImage<climage_type>(context_, upper);
            // create pair to pass to the collapser
            LevelPair pair {std::move(u), std::move(result)};

            // collapse using passed function
            result = collapseLevel(pair);

            // save upper layer for later
            lower = std::move(upper);
            upper = nextLevel();
        }

        result.readInto(dest.rawData());
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
        //std::vector<std::vector<image_type>> pyramidGuts;
        std::vector<std::vector<view_type>> pyramidGuts;

        // extract guts from pyramids
        for (ImagePyramid& pyramid : pyramids)
        {
            // ensure all pyramids have the same number of levels
            assert( pyramid.levels().size() == numLevels );

            pyramidGuts.push_back(pyramid.levels());
        }

        std::cout << "Extracted Guts" << std::endl;

        //return ImagePyramid(context, std::move(pyramidGuts[1]));

        // outer vector: represents collection of levels
        // inner vector: single level from all pyramids
        std::vector<std::vector<view_type>> levelCollection(numLevels);

        // rearrange so that each inner vector
        // has all images of a particular level from all pyramids
        for (size_t level = 0; level < numLevels; ++level)
        {
            for (std::vector<view_type>& pyramid : pyramidGuts)
            {
                levelCollection[level].push_back( std::move(pyramid.back()) );
                pyramid.pop_back();
            }
        }

        // pyramid guts should be empty
        for (std::vector<view_type>& pyramid : pyramidGuts)
        {
            assert( pyramid.empty() );
        }

        // levelCollection starts with last (smallest level),
        // and increases from there
        if ( levelCollection.size() > 1 )
        {
            assert( levelCollection[0][0].width() < levelCollection[1][0].width() );
        }

        std::cout << "Have to fuse " << levelCollection.size() << " levels" << std::endl;

        // ===================================================
        // now we can fuse each level individually
        // reuse space of first input pyramid for the result
        ImagePyramid fusedPyramid = std::move(pyramids[0]);

        // fuse all levels
        for (size_t level = 0; level < numLevels; ++level)
        {
            // these images have to be fused
            std::vector<view_type> singleLevel = std::move(levelCollection.back());
            levelCollection.pop_back();

            // create contiguous image array in memory
            HostImage<pixel_type, 3> levelArray(singleLevel);
            singleLevel.clear(); // deallocate subimages

            Pending2DImageArray clarray =
                makePendingImage<cl::Image2DArray>(context, levelArray.view());

            //std::stringstream sstr;
            //sstr << "level_test" << level << ".tiff";
            ////saveTiff16(makeHostImage<RGBA<float>>(fused), sstr.str());
            //saveTiff16(image_type(std::move(levelArray)).view(), sstr.str());

            auto dims = clarray.dimensions();
            std::cout << "Dimensions: "
                      << dims[0] << " x "
                      << dims[1] << " x "
                      << dims[2] << std::endl;

            Pending2DImage fused = fuseLevels(clarray);

            fused.readInto(fusedPyramid.views_[level].rawData());
            //fusedLevels.push_back(makeHostImage<RGBA<float>>(fused));
        }

        std::cout << "Fused " << numLevels << " levels" << std::endl;

        return fusedPyramid;
    }

    std::vector<ImagePyramid::view_type> ImagePyramid::createPyramidViews(
            size_t width,
            size_t height,
            size_t numLevels,
            HalvingFunc const& halve,
            pixel_type* array)
    {

        size_t levelWidth = width;
        size_t levelHeight = height;

        std::vector<view_type> views;

        for (size_t level = numLevels; level > 0; --level)
        {
            views.push_back(view_type({{levelWidth, levelHeight}}, array));
            array += levelWidth*levelHeight;
            levelWidth = halve(levelWidth);
            levelHeight = halve(levelHeight);
        }

        return views;
    }

    size_t ImagePyramid::pyramidSize(size_t width,
                size_t height,
                size_t numLevels,
                HalvingFunc const& halve)
    {

        size_t levelWidth = width;
        size_t levelHeight = height;

        size_t numPixels = 0;
        
        for (size_t level = numLevels; level > 0; --level)
        {
            numPixels += levelWidth*levelHeight;
            levelWidth = halve(levelWidth);
            levelHeight = halve(levelHeight);
        }

        return numPixels;
    }

} /* DynamiCL */ 

