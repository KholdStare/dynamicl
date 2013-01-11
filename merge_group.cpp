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
          numLevels_(calculateNumLevels(width, height)),
          pixelsPerPyramid(pyramidSize(width, height, numLevels_))
    { 
        // calculate space needed for merging
    }

    MergeGroup::MergeGroup(MergeGroup&& other)
        : context_(other.context_),
          program_(other.program_),
          width_(other.width_),
          height_(other.height_),
          numLevels_(other.numLevels_),
          pixelsPerPyramid(other.pixelsPerPyramid),
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
        pixelsPerPyramid = other.pixelsPerPyramid;
        pyramids_ = std::move(other.pyramids_);

        return *this;
    }

    void MergeGroup::addImage(view_type const& image)
    {
        if (image.width() != width_ || image.height() != height_)
        {
            throw std::invalid_argument("Dimensions of image passed in differ to others in the sequence.");
        }

        //arenas_.emplace_back(pyramidSize);

        std::cout << "========================\n"
                     "Creating Pyramid.\n"
                     "========================"
                  << std::endl;
        ImagePyramid pyramid(context_, image, numLevels_, halveDimension,
                [=](Pending2DImage const& im)
                {
                    return createPyramidLevel(im, program_);
                });

        pyramids_.push_back(std::move(pyramid));
    }

    void MergeGroup::mergeInto(view_type& dest)
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

        view_type collapsed =
            fused.collapse(
                [&](ImagePyramid::LevelPair const& pair)
                {
                    return collapsePyramidLevel(pair, program_);
                }
            );

        std::copy(collapsed.begin(), collapsed.end(), dest.begin());
    }
    
} /* DynamiCL */ 
