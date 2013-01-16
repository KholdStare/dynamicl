#include "save_image.h"

#include <vigra/impex.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/transformimage.hxx>

namespace
{
    template <typename OutComponentType>
    vigra::RGBValue< OutComponentType >
    convertPixelFromFloat4(DynamiCL::RGBA<float> const& in)
    {
        using namespace vigra;

        const float outMax = static_cast<float>(NumericTraits<OutComponentType>::max());

        RGBValue< OutComponentType > out;

        // copy values from input, apply alpha, and scale
        for (size_t i = 0; i < 3; ++i)
        {
            //out[i] = static_cast<OutComponentType>(outMax * in.a);
            //out[i] = static_cast<OutComponentType>(in[i] * outMax * in.a);
            out[i] = static_cast<OutComponentType>(in[i] * outMax);
        }

        return out;
    }
}

namespace DynamiCL
{
    
    void saveTiff16(FloatImageView const& in, std::string const& outPath)
    {
        typedef vigra::TinyVector< float, 4 > InPixelType;
        typedef vigra::UInt16 OutComponentType;
        typedef vigra::RGBValue< OutComponentType > OutPixelType;
        typedef vigra::BasicImage< OutPixelType > OutImgType;

        vigra::ImageExportInfo exportInfo(outPath.c_str());
        exportInfo.setFileType("TIFF");
        exportInfo.setPixelType("UINT16");
        exportInfo.setCompression("LZW"); // TODO: major bottleneck

        OutImgType out(in.width(), in.height());

        // transform
        std::transform(in.begin(), in.end(), out.begin(),
                       convertPixelFromFloat4<OutComponentType>);

        // write the image to the file given as second argument
        // the file type will be determined from the file name's extension
        exportImage(srcImageRange(out), exportInfo);
    }

} /* DynamiCL */ 

