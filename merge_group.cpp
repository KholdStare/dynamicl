#include "merge_group.h"
#include "pyr_impl.h"

namespace DynamiCL
{

    MergeGroup::MergeGroup(ComputeContext const& context,
                cl::Program const& program,
                size_t width,
                size_t height)
                //size_t numLevels)
        : context_(context),
          program_(program),
          width_(width),
          height_(height),
          numLevels_(calculateNumLevels(width, height))
    { }

    MergeGroup::MergeGroup(MergeGroup&& other)
        : context_(other.context_),
          program_(other.program_),
          width_(other.width_),
          height_(other.height_),
          numLevels_(other.numLevels_),
          pyramids_(std::move(other.pyramids_))
    {

    }

    MergeGroup& MergeGroup::operator = (MergeGroup&& other)
    {
        //context_ = other.context_;
        program_ = other.program_;
        width_ = other.width_;
        height_ = other.height_;
        numLevels_ = other.numLevels_;
        pyramids_ = std::move(other.pyramids_);

        return *this;
    }

    void MergeGroup::addImage(view_type const& image)
    {
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

        return collapsed;
    }
    
} /* DynamiCL */ 
