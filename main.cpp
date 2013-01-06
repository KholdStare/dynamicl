#include <iostream>
#include <memory>

#include <vigra/impex.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/transformimage.hxx>

#include "cl_utils.h"
#include "utils.h"
#include "image_pyramid.h"
#include "pyr_impl.h"
#include "save_image.h"

#include "plumbingplusplus/plumbing.hpp"

namespace DynamiCL
{
    typedef HostImage<RGBA<float>, 2> FloatImage;
        
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
        // TODO: huge bottleneck! must improve

        return out;
    }

    template <typename T>
    void printN(T const* array, size_t n)
    {
        for (; n > 0; --n)
        {
            std::cout << static_cast<float>(*array++) << std::endl;
        }
    }

        // TODO: size may be too large for device
        // TODO: have to check CL_DEVICE_MAX_MEM_ALLOC_SIZE from getDeviceInfo?
        //

    /**
     * Function object for merging exposures
     */
    struct mergeHDR
    {
        const size_t numExposures;
        ComputeContext const& context;
        cl::Program const& program;

        FloatImage fuseGroup(std::vector<ImagePyramid>&& group)
        {
            std::cout << "========================\n"
                         "Fusing Pyramids.\n"
                         "========================"
                      << std::endl;

            ImagePyramid fused =
                ImagePyramid::fuse(group,
                    [&](Pending2DImageArray const& im)
                    {
                        return fusePyramidLevel(im, program);
                    }
                );
            group.clear();

            std::cout << "========================\n"
                         "Collapsing Pyramid.\n"
                         "========================"
                      << std::endl;

            FloatImage collapsed =
                fused.collapse(
                    [&](ImagePyramid::LevelPair const& pair)
                    {
                        return collapsePyramidLevel(pair, program);
                    }
                );

            return collapsed;
        }

        // from shared_ptr image to shared_ptr of image
        template <typename InputIt, typename OutputIt>
        void operator() (InputIt cur, InputIt last, OutputIt dest)
        {
            Kernel quality = {program, "compute_quality", Kernel::Range::SOURCE};

            std::vector<ImagePyramid> subpyramids;
            size_t width = 1;
            size_t height = 1;
            size_t maxLevels = 1;
            while(cur != last)
            {
                std::shared_ptr<FloatImage> in = *cur++;

                // determine pyramid depth if this is a first image in sequence
                if (subpyramids.empty())
                {
                    width = in->width();
                    height = in->height();

                    maxLevels = calculateNumLevels(width, height);
                }
                // if subsequent images in sequence, check that sizes match
                else if (width != in->width() || height != in->height()) {
                    throw std::runtime_error("Image dimensions in sequence are not equal!");
                }
                
                // create quality mask in image
                std::cout << "========================\n"
                             "Creating Quality Mask.\n"
                             "========================"
                          << std::endl;
                processImageInPlace(*in, quality, context);

                // build pyramid
                std::cout << "========================\n"
                             "Creating Pyramid.\n"
                             "========================"
                          << std::endl;
                ImagePyramid pyramid(context, *in, maxLevels,
                                     [&](Pending2DImage const& im)
                                     {
                                        return createPyramidLevel(im, program);
                                     });

                // move pyramid into local collection
                subpyramids.push_back(std::move(pyramid));

                // as soon as we can merge, do so
                if (subpyramids.size() == 3)
                {
                    FloatImage collapsed = fuseGroup(std::move(subpyramids));
                    subpyramids.clear();

                    std::cout << "========================\n"
                                 "HDR Merge complete.\n"
                                 "========================"
                              << std::endl;
                    *dest = std::make_shared<FloatImage>(std::move(collapsed));
                    dest++;
                    std::cout << std::endl;
                }

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
    cl::Program program = buildProgram(gpu.context, gpu.device, "kernels.cl");

    // get image paths
    std::vector<std::string> paths;
    std::copy_n( &argv[1], argc-1, std::back_inserter(paths) );

    // set up transformation functions

    auto toFloatImage =
        [&]( std::shared_ptr<vigra::BRGBImage> im )
        {
            return transformToFloat4(*im);
        };

    int currentIndex = 1;
    auto saveImage =
        [&]( std::shared_ptr<FloatImage> im )
        {
            // create output filename
            std::stringstream sstr;
            sstr << "out" << currentIndex << ".tiff";

            // save image
            saveTiff16(*im, sstr.str());
            ++currentIndex;
        };

    // create pipeline
    std::future<void> fut =
          Plumbing::makeSource(paths)
          >> loadImage
          >> toFloatImage
          >> Plumbing::makeIteratorFilter<std::shared_ptr<FloatImage>,
                                          std::shared_ptr<FloatImage>>(mergeHDR{ 3, gpu, program })
          >> saveImage;

    // wait for pipeline to complete
    // and report any errors
    try
    {
        fut.get();
    }
    catch (cl::Error& e)
    {
        std::cout << "Encountered OpenCL Error!\n"
                  << e.what() << ": "
                  << clErrorToStr(e.err()) << std::endl;
        exit(1);
    }

    return 0;
}

