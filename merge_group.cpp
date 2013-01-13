#include "merge_group.h"
#include "pyr_impl.h"

namespace DynamiCL
{

    MergeGroup::MergeGroup(ComputeContext const& context,
                cl::Program const& program,
                size_t width,
                size_t height,
                size_t groupSize)
        : context_(context),
          program_(program),
          width_(width),
          height_(height),
          numLevels_(calculateNumLevels(width, height)),
          pixelsPerPyramid_(pyramidSize(width, height, numLevels_)),
          groupSize_(groupSize),
          arena_(pixelsPerPyramid_ * groupSize_) // total pixel count of all pyramids for merge
    { 
        // Have to create views into memory arena that will be used by
        // the image pyramids
        size_t levelWidth = width_;
        size_t levelHeight = height_;
        pixel_type* dataptr = arena_.ptr();

        // create views level by level
        for (size_t level = 0; level < numLevels_; ++level)
        {
            fuseViews_.emplace_back(std::array<size_t, 3>{{levelWidth, levelHeight, groupSize_}},
                                    dataptr);

            // move the data ptr forward in the arena
            dataptr += fuseViews_.back().totalSize();
            // halve the dimensions for the next level
            levelWidth = halveDimension(levelWidth);
            levelHeight = halveDimension(levelHeight);
        }
    }

    MergeGroup::MergeGroup(MergeGroup&& other)
        : context_(other.context_),
          program_(other.program_),
          width_(other.width_),
          height_(other.height_),
          numLevels_(other.numLevels_),
          pixelsPerPyramid_(other.pixelsPerPyramid_),
          groupSize_(other.groupSize_),
          arena_(std::move(other.arena_)),
          fuseViews_(std::move(other.fuseViews_)),
          pyramids_(std::move(other.pyramids_))
    {
        // TODO: invalidate other
    }


    void MergeGroup::addImage(view_type const& image)
    {
        if (image.width() != width_ || image.height() != height_)
        {
            throw std::invalid_argument("Dimensions of image passed in differ to others in the sequence.");
        }

        if (pyramids_.size() == groupSize_)
        {
            throw std::invalid_argument("Group already contains enough images to fuse. Cannot add another.");
        }

        // TODO: do quality mask here, then create pyramid from Pending image

        // which image in the group is this
        size_t imageNum = pyramids_.size();

        // create subviews from arena for a single pyramid
        std::vector< view_type > subviews;
        for (auto& fuseView : fuseViews_)
        {
            subviews.push_back(fuseView[imageNum]);
        }

        // transfer input image into first level of pyramid
        std::copy(image.begin(), image.end(), subviews.front().begin());

        std::cout << "========================\n"
                     "Creating Pyramid.\n"
                     "========================"
                  << std::endl;
        ImagePyramid pyramid(context_, std::move(subviews),
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

        // "borrow" first pyramid for destination
        ImagePyramid fused( std::move(pyramids_[0]) );

        ImagePyramid::fuseInto(context_, fuseViews_,
            [&](Pending2DImageArray const& im)
            {
                return fusePyramidLevel(im, program_);
            },
            // TODO: get rid of hack
            const_cast<std::vector<view_type>&>(fused.levels())
        );

        std::cout << "========================\n"
                     "Collapsing Pyramid.\n"
                     "========================"
                  << std::endl;

        fused.collapseInto(
                [&](ImagePyramid::LevelPair const& pair)
                {
                    return collapsePyramidLevel(pair, program_);
                },
                dest
            );
        pyramids_.clear();
    }
    
} /* DynamiCL */ 
