#define ARRAY_SIZE 64

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>

#include <vigra/impex.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/transformimage.hxx>

#include "cl_utils.h"
#include "utils.h"

#include "plumbingplusplus/plumbing.hpp"

namespace DynamiCL
{
    typedef HostImage<RGBA<float>> FloatImage;
        
    std::shared_ptr< vigra::BasicImage< vigra::RGBValue< vigra::UInt8 >>>
    loadImage(std::string const& path)
    {
        typedef vigra::BasicImage< vigra::RGBValue< vigra::UInt8 >> ImgType;
        vigra::ImageImportInfo info(path.c_str());

        if(info.isGrayscale())
        {
            throw std::runtime_error( "Could not open grayscale image. Only RGB images supported." );
        }

        // TODO: check pixel type is uint8

        auto img = std::make_shared<ImgType>( info.width(), info.height() );

        // import the image just read
        importImage(info, destImage(*img));

        return img;
    }

    template <typename InComponentType>
    inline RGBA<float>
    convertPixelToFloat4(vigra::RGBValue< InComponentType > const& in)
    {
        using namespace vigra;

        const float inMax = static_cast<float>(NumericTraits<InComponentType>::max());

        RGBA<float> out;

        // copy values from input and scale
        for (size_t i = 0; i < 3; ++i)
        {
            out.components[i] = static_cast<float>(in[i]) / inMax;
        }

        // set alpha
        out.a = 1.0f;

        return out;
    }

    template <typename OutComponentType>
    vigra::RGBValue< OutComponentType >
    convertPixelFromFloat4(RGBA<float> const& in)
    {
        using namespace vigra;

        const float outMax = static_cast<float>(NumericTraits<OutComponentType>::max());

        RGBValue< OutComponentType > out;

        // copy values from input, apply alpha, and scale
        for (size_t i = 0; i < 3; ++i)
        {
            //out[i] = static_cast<OutComponentType>(in[i] * outMax * in.a);
            out[i] = static_cast<OutComponentType>(in[i] * outMax);
        }

        return out;
    }

    template <typename InComponentType>
    std::shared_ptr< FloatImage >
    transformToFloat4(vigra::BasicImage< vigra::RGBValue< InComponentType >> const& in)
    {
        typedef vigra::TinyVector< float, 4 > OutPixelType;

        // Create output image
        auto out = std::make_shared<FloatImage>(in.width(), in.height());

        // transform using unary function
        std::transform(in.begin(), in.end(), out->begin(),
                       convertPixelToFloat4<InComponentType>);

        return out;
    }

    void saveTiff16(FloatImage const& in, std::string const& outPath)
    {
        typedef vigra::TinyVector< float, 4 > InPixelType;
        typedef vigra::UInt16 OutComponentType;
        typedef vigra::RGBValue< OutComponentType > OutPixelType;
        typedef vigra::BasicImage< OutPixelType > OutImgType;

        vigra::ImageExportInfo exportInfo(outPath.c_str());
        exportInfo.setFileType("TIFF");
        exportInfo.setPixelType("UINT16");
        //exportInfo.setCompression("LZW"); // TODO: major bottleneck

        OutImgType out(in.width(), in.height());

        // transform
        std::transform(in.begin(), in.end(), out.begin(),
                       convertPixelFromFloat4<OutComponentType>);

        // write the image to the file given as second argument
        // the file type will be determined from the file name's extension
        exportImage(srcImageRange(out), exportInfo);
    }

    template <typename T>
    void printN(T const* array, size_t n)
    {
        for (; n > 0; --n)
        {
            std::cout << static_cast<float>(*array++) << std::endl;
        }
    }

    inline size_t halveDimension(size_t n)
    {
        return (n + 1) / 2;
    }

    ImagePyramid::LevelPair
    createPyramidLevel(PendingImage const& inputImage,
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
        PendingImage pendingInterImage =
            inputImage.process(row, halfWidth, height);

        std::cout << "Downsampled Rows" << std::endl;

        /********************
         *  Downsample col  *
         ********************/

        // Create an output image in compute device
        size_t halfHeight = halveDimension(height);

        Kernel col = {program, "downsample_col", Kernel::Range::DESTINATION};
        PendingImage downsampled =
            pendingInterImage.process(col, halfWidth, halfHeight);

        std::cout << "Downsampled Cols" << std::endl;

        /******************
         *  Upsample col  *
         ******************/

        Kernel upcol = {program, "upsample_col", Kernel::Range::SOURCE};
        PendingImage pendingUpCol =
            downsampled.process(upcol, pendingInterImage.image);

        std::cout << "Upsampled Cols" << std::endl;

        /******************
         *  Upsample row  *
         ******************/

        Kernel uprow = {program, "upsample_row", Kernel::Range::SOURCE};
        PendingImage pendingUpRow =
            pendingUpCol.process(uprow, width, height);

        std::cout << "Upsampled Rows" << std::endl;

        /**********************
         *  Create Laplacian  *
         **********************/

        Kernel createLaplacian = {program, "create_laplacian", Kernel::Range::SOURCE};

        cl::Image2D laplacianImage(gpu.context,
                CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                width,
                height);

        cl::Kernel clkernel =
            createLaplacian.build(inputImage.image,
                                  pendingUpRow.image,
                                  laplacianImage);
                    
        cl::Event complete;
        gpu.queue.enqueueNDRangeKernel(clkernel,
                                   cl::NullRange,
                                   cl::NDRange(width, height),
                                   cl::NullRange, 
                                   &pendingUpRow.events,
                                   &complete);

        PendingImage finalResult(gpu, laplacianImage);
        finalResult.events.push_back(complete);

        std::cout << "Created Laplacian" << std::endl;

        return {std::move(finalResult), std::move(downsampled)};
    }

    
    PendingImage
    collapsePyramidLevel(ImagePyramid::LevelPair const& pair,
                         cl::Program const& program )
    {
        ComputeContext const& context = pair.upper.context;

        // get all the dimensions
        size_t upperWidth  = pair.upper.width();
        size_t upperHeight = pair.upper.height();
        size_t lowerWidth  = pair.lower.width();
        size_t lowerHeight = pair.lower.height();

        /******************
         *  Upsample col  *
         ******************/

        Kernel upcol = {program, "upsample_col", Kernel::Range::SOURCE};
        PendingImage pendingUpCol =
            pair.lower.process(upcol, lowerWidth, upperHeight);

        std::cout << "Upsampled Cols" << std::endl;

        /******************
         *  Upsample row  *
         ******************/

        Kernel uprow = {program, "upsample_row", Kernel::Range::SOURCE};
        PendingImage pendingUpRow =
            pendingUpCol.process(uprow, upperWidth, upperHeight);

        std::cout << "Upsampled Rows" << std::endl;

        /**********************
         *  Create Laplacian  *
         **********************/

        Kernel collapse= {program, "collapse_level", Kernel::Range::SOURCE};

        // TODO: add utility function for processing several image
        // into one output image.

        cl::Image2D collapsedImage(context.context,
                CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                upperWidth,
                upperHeight);

        cl::Kernel clkernel =
            collapse.build(pendingUpRow.image,
                                  pair.upper.image,
                                  collapsedImage);
                    
        cl::Event complete;
        pendingUpRow.events.insert(end(pendingUpRow.events),
                                   begin(pair.upper.events),
                                   end(pair.upper.events));

        context.queue.enqueueNDRangeKernel(clkernel,
                                   cl::NullRange,
                                   cl::NDRange(upperWidth, upperHeight),
                                   cl::NullRange, 
                                   &pendingUpRow.events,
                                   &complete);

        PendingImage finalResult(context, collapsedImage);
        finalResult.events.push_back(complete);

        std::cout << "Created Laplacian" << std::endl;

        return finalResult;
    }

        // TODO: size may be too large for device
        // TODO: have to check CL_DEVICE_MAX_MEM_ALLOC_SIZE from getDeviceInfo?
    PendingImage calculateQualityCL(PendingImage const& inputImage, cl::Program const& program )
    {
        Kernel quality = {program, "compute_quality", Kernel::Range::SOURCE};

        return inputImage.process(quality);
    }

    /**
     * Function object for merging exposures
     */
    struct mergeHDR
    {
        const size_t numExposures;
        ComputeContext const& context;
        cl::Program const& program;

        // from shared_ptr image to shared_ptr of image
        template <typename InputIt, typename OutputIt>
        void operator() (InputIt cur, InputIt last, OutputIt dest)
        {
            Kernel quality = {program, "compute_quality", Kernel::Range::SOURCE};

            while(cur != last)
            {
                std::shared_ptr<FloatImage> in = *cur++;
                
                // create quality mask in image
                processImageInPlace(*in, quality, context);

                // build pyramid
                ImagePyramid pyramid(context, *in, 8,
                                     [&](PendingImage const& im)
                                     {
                                        return createPyramidLevel(im, program);
                                     });

                auto collapsed = pyramid.collapse(
                                     [&](ImagePyramid::LevelPair const& pair)
                                     {
                                        return collapsePyramidLevel(pair, program);
                                     });

                *dest = std::make_shared<FloatImage>(std::move(collapsed));
                dest++;

                //std::vector<FloatImage> levels = pyramid.releaseLevels();
                
                //for (auto&& level : levels)
                //{
                    //*dest = std::make_shared<FloatImage>(std::move(level));
                    //dest++;
                //}
            }
        }

    };

} /* DynamiCL */ 

int main(int argc, char const *argv[])
{
    // for io efficiency:
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);

    using namespace DynamiCL;

    // create device, context, and queue
    ComputeContext gpu;

    // Build program 
    cl::Program program = buildProgram(gpu.context, gpu.device, "expocl.cl");

    // get image paths
    std::vector<std::string> paths;
    std::copy_n( &argv[1], argc-1, std::back_inserter(paths) );

    auto transformImage =
        [&]( std::shared_ptr<vigra::BRGBImage> im )
        {
            return transformToFloat4(*im);
        };

    auto calcQuality =
        [&]( std::shared_ptr<FloatImage> in ) -> std::shared_ptr<FloatImage>
        {
            PendingImage inputImage = makePendingImage(gpu, *in);

            PendingImage outputImage = calculateQualityCL(inputImage, program);
            return makeHostImage<FloatImage::pixel_type>(outputImage);
        };

    int currentIndex = 1;
    auto saveImage =
        [&]( std::shared_ptr<FloatImage> im )
        {
            std::stringstream sstr;
            sstr << "out" << currentIndex << ".tiff";

            saveTiff16(*im, sstr.str());
            ++currentIndex;
        };

    std::future<void> fut =
          Plumbing::makeSource(paths)
          >> loadImage
          >> transformImage
          >> Plumbing::makeIteratorFilter<std::shared_ptr<FloatImage>,
                                          std::shared_ptr<FloatImage>>(mergeHDR{ 3, gpu, program })
          //>> calcQuality
          >> saveImage;

    try
    {
        fut.get();
    }
    catch (cl::Error& e)
    {
        std::cout << "Encountered OpenCL Error!\n" << e.what() << ": " << clErrorToStr(e.err()) << std::endl;
        exit(1);
    }

    return 0;
}

