#include "merge_group.h"
#include "image_pyramid.h"
#include "pyr_impl.h"

namespace DynamiCL
{

    MergeGroup::MergeGroup(ComputeContext const& context,
                cl::Program const& program)
                //size_t numLevels)
        : context_(context),
          program_(program),
          numLevels_(0)
          //numLevels_(numLevels)
    { }

    //ImagePyramid& MergeGroup::addImage(view_type const& image)
    void MergeGroup::addImage(view_type const& image)
    {
        if (!numLevels_)
        {
            numLevels_ = calculateNumLevels(image.width(), image.height());
        }

        std::cout << "========================\n"
                     "Creating Pyramid.\n"
                     "========================"
                  << std::endl;
        ImagePyramid pyramid(context_, image, numLevels_,
                [=](Pending2DImage const& im)
                {
                    return createPyramidLevel(im, program_);
                });

        pyramids_.push_back(std::move(pyramid));
        //return pyramids_.back();
    }

    MergeGroup::image_type MergeGroup::merge()
    {
        std::cout << "========================\n"
            "Fusing Pyramids.\n"
            "========================"
            << std::endl;

        ImagePyramid fused =
            ImagePyramid::fuse(pyramids_,
                [&](Pending2DImageArray const& im)
                {
                    return fusePyramidLevel(im, program_);
                }
            );
        pyramids_.clear();

        std::cout << "========================\n"
            "Collapsing Pyramid.\n"
            "========================"
            << std::endl;

        image_type collapsed =
            fused.collapse(
                [&](ImagePyramid::LevelPair const& pair)
                {
                    return collapsePyramidLevel(pair, program_);
                }
            );

        numLevels_ = 0; // reset numlevels, to re-detect it later TODO: ewww
        return collapsed;
    }
    
} /* DynamiCL */ 
