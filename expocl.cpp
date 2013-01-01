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
    typedef vigra::BasicImage< vigra::TinyVector< float, 4 >> FloatImage;
        
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
    inline vigra::TinyVector<float, 4>
    convertPixelToFloat4(vigra::RGBValue< InComponentType > const& in)
    {
        using namespace vigra;

        const float inMax = static_cast<float>(NumericTraits<InComponentType>::max());

        TinyVector<float, 4> out;
        auto inIt = in.begin();
        auto inEnd = in.end();
        auto outIt = out.begin();
        auto outEnd = out.end();

        // copy values from input and scale
        for (; ( inIt != inEnd ) && ( outIt != outEnd ); ++inIt, ++ outIt)
        {
            *outIt = static_cast<float>(*inIt) / inMax;
        }

        // fill any remaining values in output with ones
        for (; outIt != outEnd; ++ outIt)
        {
            *outIt = 1.0f;
        }

        return out;
    }

    template <typename OutComponentType>
    vigra::RGBValue< OutComponentType >
    convertPixelFromFloat4(vigra::TinyVector<float, 4> const& in)
    {
        using namespace vigra;

        const float outMax = static_cast<float>(NumericTraits<OutComponentType>::max());

        RGBValue< OutComponentType > out;
        auto inIt = in.begin();
        auto outIt = out.begin();
        auto outEnd = out.end();

        // copy values from input, apply alpha, and scale
        for (; outIt != outEnd; ++inIt, ++outIt)
        {
            *outIt = static_cast<OutComponentType>(*inIt * outMax * in[3]);
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
        vigra::transformImage(in.upperLeft(), in.lowerRight(), in.accessor(),
                out->upperLeft(), out->accessor(), convertPixelToFloat4<InComponentType>);

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
        //exportInfo.setCompression("LZW");

        OutImgType out(in.width(), in.height());

        // transform
        vigra::transformImage(in.upperLeft(), in.lowerRight(), in.accessor(),
                out.upperLeft(), out.accessor(),
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

    std::shared_ptr< FloatImage >
    transformWithKernel( std::shared_ptr< FloatImage > in,
                       ComputeContext const& gpu,
                       cl::Program const& program )
    {
        using namespace vigra;
        typedef float InComponentType;
        typedef TinyVector< InComponentType, 4 > PixelType;
        typedef BasicImage< PixelType > ImageType;

        // get raw component array from input image
        InComponentType const* inArray = reinterpret_cast<const InComponentType*>(in->data());
        std::cout << "In array:" << std::endl;
        printN(inArray, 12);

        /********************
         *  Downsample row  *
         ********************/

        // create input buffer from input image
        // TODO: size may be too large for device
        // TODO: have to check CL_DEVICE_MAX_MEM_ALLOC_SIZE from getDeviceInfo?
        cl::Image2D clInputImage(gpu.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                in->width(),
                in->height(),
                0,
                const_cast<InComponentType*>(inArray));

        PendingImage inputImage(gpu, clInputImage);

        // calculate width of next image
        size_t halfWidth = halveDimension(in->width());

        // row downsampling kernel
        Kernel row = {program, "downsample_row", 5, Kernel::Range::DESTINATION};

        // process image with kernel
        PendingImage pendingInterImage =
            inputImage.process(row, halfWidth, in->height());

        std::cout << "Downsampled Rows" << std::endl;

        /********************
         *  Downsample col  *
         ********************/

        // Create an output image in compute device
        size_t halfHeight = halveDimension(in->height());

        Kernel col = {program, "downsample_col", 5, Kernel::Range::DESTINATION};
        PendingImage pendingSmallImage =
            pendingInterImage.process(col, halfWidth, halfHeight);

        std::cout << "Downsampled Cols" << std::endl;

        /******************
         *  Upsample col  *
         ******************/

        Kernel upcol = {program, "upsample_col", 5, Kernel::Range::SOURCE};
        PendingImage pendingUpCol =
            pendingSmallImage.process(upcol, pendingInterImage.image);

        std::cout << "Upsampled Cols" << std::endl;

        /******************
         *  Upsample row  *
         ******************/

        Kernel uprow = {program, "upsample_row", 5, Kernel::Range::SOURCE};
        PendingImage pendingUpRow =
            pendingUpCol.process(uprow, in->width(), in->height());

        std::cout << "Upsampled Rows" << std::endl;

        /**********************
         *  Create Laplacian  *
         **********************/

        Kernel createLaplacian = {program, "create_laplacian", 1, Kernel::Range::SOURCE};

        cl::Image2D laplacianImage(gpu.context,
                CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                in->width(),
                in->height());

        cl::Kernel clkernel =
            createLaplacian.build(inputImage.image,
                                  pendingUpRow.image,
                                  laplacianImage);
                    
        cl::Event complete;
        gpu.queue.enqueueNDRangeKernel(clkernel,
                                   cl::NullRange,
                                   cl::NDRange(in->width(), in->height()),
                                   cl::NullRange, 
                                   &pendingUpRow.events,
                                   &complete);

        PendingImage finalResult(gpu, laplacianImage);
        finalResult.events.push_back(complete);

        std::cout << "Created Laplacian" << std::endl;

        /***********************
         *  Read final output  *
         ***********************/

        //// Create new output vigra image
        //auto outImg = std::make_shared<ImageType>( in->width(), in->height() );
        //InComponentType const* outArray = reinterpret_cast<const InComponentType*>(outImg->data());

        // Read the kernel's output
        finalResult.read(const_cast<InComponentType*>(inArray));

        std::cout << "Out array:" << std::endl;
        printN(inArray, 12);

        return in;
    }

    std::shared_ptr< FloatImage >
    calculateQualityCL(std::shared_ptr< FloatImage > in,
                       ComputeContext const& gpu,
                       cl::Program const& program )
    {
        using namespace vigra;
        typedef float InComponentType;
        typedef TinyVector< InComponentType, 4 > PixelType;
        typedef BasicImage< PixelType > ImageType;

        // get raw component array from input image
        InComponentType const* inArray = reinterpret_cast<const InComponentType*>(in->data());
        std::cout << "In array:" << std::endl;
        printN(inArray, 12);

        // create input buffer from input image
        // TODO: size may be too large for device
        // TODO: have to check CL_DEVICE_MAX_MEM_ALLOC_SIZE from getDeviceInfo?
        cl::Image2D inputImage(gpu.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY,
                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                in->width(),
                in->height(),
                0,
                const_cast<InComponentType*>(inArray));

        // Create an output image in compute device
        cl::Image2D outputImage(gpu.context,
                CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                cl::ImageFormat(CL_RGBA, CL_FLOAT),
                in->width(),
                in->height());

        // Kernel for downsampling columns
        cl::Kernel qualityKernel(program, "compute_quality");
        qualityKernel.setArg(0, inputImage);
        qualityKernel.setArg(1, outputImage);

        // Enqueue col kernel
        cl::Event kernelComplete;
        gpu.queue.enqueueNDRangeKernel(qualityKernel,
                                   cl::NullRange,
                                   cl::NDRange(in->width(), in->height()),
                                   cl::NullRange, 
                                   nullptr,
                                   &kernelComplete);

        // Read the kernel's output back into input image
        std::vector<cl::Event> waitfor = {kernelComplete};
        gpu.queue.enqueueReadImage(outputImage,
                CL_TRUE,
                VectorConstructor<size_t>::construct(0, 0, 0),
                VectorConstructor<size_t>::construct(in->width(), in->height(), 1),
                0,
                0,
                const_cast<InComponentType*>(inArray),
                &waitfor);

        std::cout << "Out array:" << std::endl;
        printN(inArray, 12);

        return in;
    }


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
        [&]( std::shared_ptr<FloatImage> im )
        {
            return calculateQualityCL(im, gpu, program);
        };

    auto constructPyramid =
        [&]( std::shared_ptr<FloatImage> im )
        {
            return transformWithKernel(im, gpu, program);
        };

    int currentIndex = 1;
    auto saveImage =
        [&]( std::shared_ptr<FloatImage> im )
        {
            std::string outPath(DynamiCL::stripExtension(argv[currentIndex]));
            outPath += ".tiff";

            saveTiff16(*im, outPath);
            ++currentIndex;
        };

    std::future<void> fut =
          Plumbing::makeSource(paths)
          >> loadImage
          >> transformImage
          //>> calcQuality
          >> constructPyramid
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

