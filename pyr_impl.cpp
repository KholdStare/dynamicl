#include "pyr_impl.h"
#include <iostream>

namespace
{


}

namespace DynamiCL
{

    ImagePyramid::LevelPair
    createPyramidLevel(Pending2DImage const& inputImage,
                       cl::Program const& program )
    {
        ComputeContext const& gpu = inputImage.context;
        size_t width = inputImage.width();
        size_t height = inputImage.height();

        /********************
         *  Downsample row  *
         ********************/

        // calculate width of next image
        size_t halfWidth = halveDimension(width);

        // row downsampling kernel
        Kernel row = {program, "downsample_row", Kernel::Range::DESTINATION};

        // process image with kernel
        Pending2DImage pendingInterImage =
            inputImage.process<cl::Image2D>(row, {{ halfWidth, height }});

        std::cout << "Downsampled Rows" << std::endl;

        /********************
         *  Downsample col  *
         ********************/

        // Create an output image in compute device
        size_t halfHeight = halveDimension(height);

        Kernel col = {program, "downsample_col", Kernel::Range::DESTINATION};
        Pending2DImage downsampled =
            pendingInterImage.process<cl::Image2D>(col, {{halfWidth, halfHeight}});

        std::cout << "Downsampled Cols" << std::endl;

        /******************
         *  Upsample col  *
         ******************/

        Kernel upcol = {program, "upsample_col", Kernel::Range::SOURCE};
        Pending2DImage pendingUpCol =
            downsampled.process(upcol, pendingInterImage.image);

        std::cout << "Upsampled Cols" << std::endl;

        /******************
         *  Upsample row  *
         ******************/

        Kernel uprow = {program, "upsample_row", Kernel::Range::SOURCE};
        Pending2DImage pendingUpRow =
            pendingUpCol.process<cl::Image2D>(uprow, {{width, height}});

        std::cout << "Upsampled Rows" << std::endl;

        /**********************
         *  Create Laplacian  *
         **********************/

        Kernel createLaplacian = {program, "create_laplacian", Kernel::Range::SOURCE};

        Pending2DImage pendingResult = 
            Pending::process<cl::Image2D>
            (
                    gpu,
                    createLaplacian,
                    inputImage.dimensions(), // dimensions
                    toNDRange(inputImage.dimensions()), // problem range
                    inputImage, pendingUpRow // input images
            );

        std::cout << "Created Laplacian" << std::endl;

        return {std::move(pendingResult), std::move(downsampled)};
    }

    Pending2DImage
    collapsePyramidLevel(ImagePyramid::LevelPair const& pair,
                         cl::Program const& program )
    {
        ComputeContext const& context = pair.upper.context;

        // get all the dimensions
        size_t upperWidth  = pair.upper.width();
        size_t upperHeight = pair.upper.height();
        size_t lowerWidth  = pair.lower.width();

        /******************
         *  Upsample col  *
         ******************/

        Kernel upcol = {program, "upsample_col", Kernel::Range::SOURCE};
        Pending2DImage pendingUpCol =
            pair.lower.process<cl::Image2D>(upcol, {{lowerWidth, upperHeight}});

        std::cout << "Upsampled Cols" << std::endl;

        /******************
         *  Upsample row  *
         ******************/

        Kernel uprow = {program, "upsample_row", Kernel::Range::SOURCE};
        Pending2DImage pendingUpRow =
            pendingUpCol.process<cl::Image2D>(uprow, {{upperWidth, upperHeight}});

        std::cout << "Upsampled Rows" << std::endl;

        /**********************
         *  Create Laplacian  *
         **********************/

        Kernel collapse= {program, "collapse_level", Kernel::Range::SOURCE};

        auto pendingResult =
            Pending::process<cl::Image2D>
            (
                context,
                collapse,
                pair.upper.dimensions(),
                toNDRange(pair.upper.dimensions()),
                pendingUpRow, pair.upper
            );

        std::cout << "Created Laplacian" << std::endl;

        return pendingResult;
    }

    Pending2DImage
    fusePyramidLevel(Pending2DImageArray const& array,
                         cl::Program const& program )
    {
        ComputeContext const& context = array.context;

        // get all the dimensions
        size_t width  = array.width();
        size_t height = array.height();
        std::cout << "DEPTH :" << array.depth() << std::endl;

        /********************
         *  Fuse the level  *
         ********************/

        cl::Image2D resultImage(context.context,
                CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                width,
                height);

        Kernel kernel = {program, "fuse_level", Kernel::Range::DESTINATION};
        cl::Kernel clkernel = kernel.build(array.image, resultImage);

        //Pending2DImage fused =
            //array.process(fuse, width, height);

        std::cout << "MERGING LEVEL :"
                  << width  << " x "
                  << height << std::endl;

        cl::Event complete;
        context.queue.enqueueNDRangeKernel(clkernel,
                                   cl::NullRange,
                                   cl::NDRange(width, height, 1),
                                   cl::NullRange, 
                                   &array.events,
                                   &complete);

        Pending2DImage fused(context, resultImage);
        fused.events.push_back(complete);

        return fused;
    }

    size_t calculateNumLevels(size_t width, size_t height)
    {
        size_t shortDim = std::min(width, height);

        size_t levels = 1;
        while (shortDim > 8)
        {
            shortDim = halveDimension(shortDim);
            ++levels;
        }

        return levels;
    }

    size_t pyramidSize(size_t width, size_t height, size_t numLevels)
    {

        size_t levelWidth = width;
        size_t levelHeight = height;

        size_t numPixels = 0;
        
        for (size_t level = numLevels; level > 0; --level)
        {
            numPixels += levelWidth*levelHeight;
            levelWidth = halveDimension(levelWidth);
            levelHeight = halveDimension(levelHeight);
        }

        return numPixels;
    }

}
