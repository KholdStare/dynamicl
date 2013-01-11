#ifndef SAVE_IMAGE_H_RNXX0VQG
#define SAVE_IMAGE_H_RNXX0VQG

#include <string>
#include "host_image.hpp"

namespace DynamiCL
{
    typedef HostImage<RGBA<float>, 2> FloatImage;
    typedef HostImageView<RGBA<float>, 2> FloatImageView;

    void saveTiff16(FloatImageView const& in, std::string const& outPath);
}

#endif /* end of include guard: SAVE_IMAGE_H_RNXX0VQG */
